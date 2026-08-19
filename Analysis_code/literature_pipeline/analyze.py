from __future__ import annotations

import html
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from path_config import ANALYSIS_DIR, MODEL_DIR, MPL_CONFIG_DIR, PARSED_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict


INPUT = PARSED_DIR / "all_records_raw.pkl"
OUT = ANALYSIS_DIR
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODEL_DIR
MODEL_OUT.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def flag(series: pd.Series, expressions: list[str]) -> pd.Series:
    pattern = "(?:" + "|".join(expressions) + ")"
    return series.str.contains(pattern, regex=True, na=False)


df = pd.read_pickle(INPUT).copy()

# Prefer the final, richer record when WoS supplies multiple UTs for the same DOI.
df["metadata_richness"] = (
    df["AB"].fillna("").str.len()
    + df["DE"].fillna("").str.len()
    + df["ID"].fillna("").str.len()
    + df["CR"].fillna("").str.len().clip(upper=20000)
)
df["is_early_access"] = df["DT"].fillna("").str.contains("Early Access", case=False)
df["source_key"] = df["SO"].map(normalize_text).str.replace(r"[^a-z0-9]+", "", regex=True)
df = df.sort_values(
    ["is_early_access", "metadata_richness", "times_cited"],
    ascending=[True, False, False],
    kind="stable",
)
df["duplicate_group_doi"] = df["DOI_normalized"].where(df["DOI_normalized"].ne(""), "")
df["duplicate_group_title_source"] = (
    df["title_normalized_key"] + "|" + df["source_key"]
).where(df["title_normalized_key"].ne("") & df["source_key"].ne(""), "")
doi_dup = df["duplicate_group_doi"].ne("") & df.duplicated("duplicate_group_doi", keep="first")
title_source_dup = (
    df["duplicate_group_title_source"].ne("")
    & df.duplicated("duplicate_group_title_source", keep="first")
)
df["excluded_duplicate"] = doi_dup | title_source_dup
df = df.sort_values("dedupe_priority", kind="stable").reset_index(drop=True)
df.insert(0, "record_id", [f"WCC{i:05d}" for i in range(1, len(df) + 1)])

for col in ["TI", "AB", "DE", "ID"]:
    df[f"{col}_norm"] = df[col].map(normalize_text)
df["core_text"] = (df["TI_norm"] + ". " + df["AB_norm"] + ". " + df["DE_norm"]).str.strip(". ")
df["all_text"] = (df["core_text"] + ". " + df["ID_norm"]).str.strip(". ")

fire_terms = [
    r"\bwildfires?\b", r"\bwildland fires?\b", r"\bforest fires?\b",
    r"\bbushfires?\b", r"\bbush fires?\b", r"\bvegetation fires?\b",
    r"\blandscape fires?\b", r"\bpeat(?:land)? fires?\b", r"\bgrassland fires?\b",
    r"\bsavanna fires?\b", r"\bboreal fires?\b", r"\bfire regimes?\b",
    r"\bwildland[ -]urban (?:interface )?fires?\b",
]
climate_strong = [
    r"\bclimat(?:e|ic) chang\w*\b", r"\bglobal warm\w*\b",
    r"\banthropogenic (?:climat\w*|warm\w*|forcing)\b",
    r"\bhuman[ -]induced (?:climat\w*|warm\w*)\b",
    r"\bclimat(?:e|ic) (?:trend|projection|scenario|forcing|attribution|feedback|impact)s?\b",
    r"\bclimat(?:e|ic) variab\w*\b", r"\bclimat(?:e|ic) fluctuat\w*\b",
    r"\bfire[ -]climate feedbacks?\b", r"\bclimate[ -]fire feedbacks?\b",
    r"\bcarbon[ -]climate feedbacks?\b", r"\bclimate[ -]driven\b",
    r"\bradiative forcing\b", r"\btemperature responses?\b",
]
climate_weak = [
    r"\bfire weather\b", r"\bfire danger\b", r"\bfuel aridity\b",
    r"\bvapou?r[ -]pressure deficit\b", r"\bdrought\b", r"\bheatwaves?\b",
]

for field_name in ["TI_norm", "AB_norm", "DE_norm", "ID_norm", "core_text", "all_text"]:
    df[f"fire_{field_name}"] = flag(df[field_name], fire_terms)
    df[f"climate_strong_{field_name}"] = flag(df[field_name], climate_strong)
    df[f"climate_weak_{field_name}"] = flag(df[field_name], climate_weak)

df["fire_core"] = df["fire_core_text"]
df["climate_core"] = df["climate_strong_core_text"]
df["weak_climate_only"] = ~df["climate_core"] & df["climate_weak_core_text"]
df["kp_only_bridge"] = (
    df["fire_all_text"] & df["climate_strong_all_text"]
    & ~(df["fire_core"] & df["climate_core"])
)
df["high_conf_positive"] = (
    (df["fire_TI_norm"] & df["climate_strong_core_text"])
    | (df["climate_strong_TI_norm"] & df["fire_core_text"])
    | (df["fire_AB_norm"] & df["climate_strong_AB_norm"])
    | (df["fire_DE_norm"] & df["climate_strong_DE_norm"])
)
df["high_conf_negative"] = df["kp_only_bridge"] | (
    df["fire_core"] & ~df["climate_core"] & df["weak_climate_only"]
)

# Weakly supervised relevance model. Seed rules define high-confidence training labels;
# predictions prioritize manual review and do not constitute final systematic eligibility.
seed = df.loc[df["high_conf_positive"] | df["high_conf_negative"]].copy()
seed["seed_label"] = seed["high_conf_positive"].astype(int)
model_stop = sorted(set(ENGLISH_STOP_WORDS) | {
    "study", "studies", "results", "result", "data", "using", "based", "analysis",
    "model", "models", "year", "years", "area", "areas", "research", "paper",
})
rel_vectorizer = TfidfVectorizer(
    stop_words=model_stop,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=30000,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
)
X_all_rel = rel_vectorizer.fit_transform(df["all_text"])
seed_idx = seed.index.to_numpy()
X_seed = X_all_rel[seed_idx]
y_seed = seed["seed_label"].to_numpy()
clf = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0, random_state=20260810)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260810)
cv_prob = cross_val_predict(clf, X_seed, y_seed, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
cv_pred = (cv_prob >= 0.5).astype(int)
clf.fit(X_seed, y_seed)
joblib.dump(rel_vectorizer, MODEL_OUT / "relevance_vectorizer.joblib")
joblib.dump(clf, MODEL_OUT / "relevance_classifier.joblib")
df["ml_relevance_probability"] = clf.predict_proba(X_all_rel)[:, 1]

df["relevance_tier"] = "Peripheral_screening_candidate"
direct = df["high_conf_positive"] | (
    df["fire_core"] & df["climate_core"] & df["ml_relevance_probability"].ge(0.50)
)
context = ~direct & (
    (df["fire_core"] & (df["climate_core"] | df["climate_weak_core_text"]))
    | df["ml_relevance_probability"].ge(0.55)
)
df.loc[context, "relevance_tier"] = "Contextual_or_indirect"
df.loc[direct, "relevance_tier"] = "Core_direct"
df["eligible_document_type"] = df["DT"].fillna("").str.contains(r"Article|Review", case=False, regex=True)
df["analysis_status"] = "retained_for_screening"
df.loc[df["excluded_duplicate"], "analysis_status"] = "excluded_duplicate"
df.loc[df["retracted_or_withdrawn"], "analysis_status"] = "excluded_retracted"
df.loc[~df["eligible_document_type"], "analysis_status"] = "excluded_document_type"
primary_mask = (
    ~df["excluded_duplicate"] & ~df["retracted_or_withdrawn"] & df["eligible_document_type"]
    & df["complete_year_1900_2025"] & df["abstract_available"]
    & df["relevance_tier"].ne("Peripheral_screening_candidate")
)
df.loc[primary_mask, "analysis_status"] = "primary_text_analysis"

cv_summary = {
    "seed_n": int(len(seed)),
    "positive_seed_n": int(y_seed.sum()),
    "negative_seed_n": int((1 - y_seed).sum()),
    "pseudo_label_cv_auc": float(roc_auc_score(y_seed, cv_prob)),
    "confusion_matrix": confusion_matrix(y_seed, cv_pred).tolist(),
    "classification_report": classification_report(y_seed, cv_pred, output_dict=True),
    "interpretation": "Cross-validation measures recovery of deterministic seed rules, not independent human-validated accuracy.",
}

# Manuscript-aligned SFCA dictionary.
concept_patterns: dict[str, list[str]] = {
    "source_explicit": [
        r"\bignition (?:source|cause)s?\b", r"\bfire causes?\b", r"\bcause of (?:the )?(?:wild)?fire\b",
        r"\b(?:lightning|human|anthropogenic|natural)[ -](?:caused|ignited|started) (?:wild)?fires?\b",
        r"\b(?:wild)?fires? (?:caused|ignited|started) by (?:lightning|human|people|anthropogenic)\b",
        r"\bnatural ignition\b", r"\bhuman ignition\b", r"\barson\b", r"\bescaped (?:prescribed|managed) fire\b",
    ],
    "natural_ignition": [
        r"\blightning[ -](?:caused|ignited|started)\b", r"\b(?:wild)?fires? (?:caused|ignited|started) by lightning\b",
        r"\bnatural ignition\b", r"\blightning ignition\b",
    ],
    "human_ignition": [
        r"\b(?:human|anthropogenic)[ -](?:caused|ignited|started)\b",
        r"\b(?:wild)?fires? (?:caused|ignited|started) by (?:human|people)\b", r"\bhuman ignition\b",
        r"\banthropogenic ignition\b", r"\barson\b", r"\baccidental ignition\b",
        r"\binfrastructure[ -](?:caused|ignited)\b", r"\bescaped (?:prescribed|managed) fire\b",
    ],
    "climate_attribution": [
        r"\b(?:wildfire|forest fire|fire weather|fire risk|fire season|burned area)s?\b.{0,120}\battribut(?:e|ed|able|ion)\b.{0,120}\b(?:climate change|global warming|anthropogenic warming|human emissions)\b",
        r"\b(?:climate change|global warming|anthropogenic warming|human emissions)\b.{0,120}\battribut(?:e|ed|able|ion)\b.{0,120}\b(?:wildfire|forest fire|fire weather|fire risk|fire season|burned area)s?\b",
        r"\banthropogenic (?:climate change|warming|forcing)\b.{0,100}\b(?:increase|drive|driven|cause|exacerbate|contribution|effect|impact|likelihood|odds)\w*\b.{0,100}\b(?:wildfire|forest fire|fire weather|fire risk|fire season|burned area)s?\b",
        r"\b(?:climate|event|extreme event) attribution\b.{0,120}\b(?:wildfire|forest fire|fire weather|fire risk|fire season|burned area)s?\b",
        r"\bfraction of attributable risk\b", r"\bprobability ratio\b", r"\brisk ratio\b.{0,100}\bclimate\b",
    ],
    "counterfactual": [
        r"\bcounterfactual\b", r"\bwithout (?:anthropogenic )?climate change\b",
        r"\bworld without (?:anthropogenic )?(?:climate change|warming|fire)\b",
        r"\bno[ -]fire (?:scenario|simulation|experiment|world)\b", r"\bfire[ -]off (?:scenario|simulation|experiment)\b",
        r"\bpreindustrial (?:climate|forcing|scenario)\b", r"\ball[ -]forcing\b", r"\bnatural[ -]forcing\b",
    ],
    "fuel_legacy": [
        r"\bfuel (?:load|loading|continuity|accumulation|treatment|management)s?\b", r"\bfire exclusion\b",
        r"\bland[ -]use (?:legacy|history|change)\b", r"\blandscape (?:legacy|fragmentation)\b",
    ],
    "suppression_control": [
        r"\bfire suppression\b", r"\bwildfire suppression\b", r"\bfirefighting\b", r"\binitial attack\b",
        r"\bcontainment\b", r"\bprescribed (?:fire|burn)\b", r"\bcontrolled burn\b",
    ],
    "burned_area": [r"\bburn(?:ed|t) area\b", r"\barea burned\b", r"\bfire extent\b"],
    "fire_severity": [
        r"\bfire severity\b", r"\bburn severity\b", r"\bhigh[ -]severity fire\b",
        r"\bfire intensity\b", r"\bcombustion completeness\b",
    ],
    "emissions_carbon": [
        r"\b(?:fire|wildfire|biomass burning) emissions?\b", r"\bcarbon (?:emission|loss|release|flux|balance|budget|debt|stock)s?\b",
        r"\bco2 emissions?\b", r"\bgreenhouse gas(?:es)?\b", r"\bmethane\b", r"\bblack carbon\b",
    ],
    "nonco2_biophysical": [
        r"\baerosols?\b", r"\bblack carbon\b", r"\bmethane\b", r"\balbedo\b",
        r"\bsurface energy(?: balance)?\b", r"\bevapotranspiration\b", r"\bbiophysical (?:effect|feedback|impact)s?\b",
    ],
    "radiative_temperature": [
        r"\bradiative (?:forcing|effect|impact)s?\b", r"\bclimate forcing\b", r"\btemperature response\b",
        r"\bland surface (?:warming|cooling|temperature)\b", r"\bsurface (?:warming|cooling)\b",
    ],
    "postfire_recovery": [
        r"\bpost[ -]fire (?:recovery|regeneration|regrowth|succession|carbon uptake)\b",
        r"\bvegetation recovery\b", r"\becosystem recovery\b", r"\bcarbon recovery\b", r"\bcarbon debt\b",
    ],
    "feedback_closure": [
        r"\bfire[ -]climate feedbacks?\b", r"\bclimate[ -]fire feedbacks?\b", r"\bfire-mediated climate feedback\b",
        r"\b(?:wild)?fire\b.{0,80}\b(?:radiative forcing|temperature response)\b",
        r"\b(?:radiative forcing|temperature response)\b.{0,80}\b(?:wild)?fire\b",
    ],
    "fire_weather": [
        r"\bfire weather\b", r"\bfire danger\b", r"\bfuel aridity\b",
        r"\bvapou?r[ -]pressure deficit\b", r"\bhot and dry\b", r"\bcompound (?:heat|hot|drought|climate)\b",
    ],
}

for name, patterns in concept_patterns.items():
    df[name] = flag(df["core_text"], patterns)

df["ignition_class"] = "Not_resolved"
df.loc[df["natural_ignition"] & ~df["human_ignition"], "ignition_class"] = "Natural_only"
df.loc[df["human_ignition"] & ~df["natural_ignition"], "ignition_class"] = "Human_only"
df.loc[df["natural_ignition"] & df["human_ignition"], "ignition_class"] = "Both_or_comparative"
df["process_layer"] = df["fuel_legacy"] | df["suppression_control"] | df["fire_weather"]
df["fire_response_layer"] = df["burned_area"] | df["fire_severity"]
df["emission_biophysical_layer"] = df["emissions_carbon"] | df["nonco2_biophysical"] | df["radiative_temperature"]
df["recovery_layer"] = df["postfire_recovery"]
df["counterfactual_feedback_layer"] = df["counterfactual"] | df["feedback_closure"]
layer_cols = [
    "source_explicit", "climate_attribution", "process_layer", "fire_response_layer",
    "emission_biophysical_layer", "recovery_layer", "counterfactual_feedback_layer",
]
df["sfca_component_count_0_7"] = df[layer_cols].sum(axis=1)

def broad_discipline(wc: object, sc: object) -> str:
    text = normalize_text(f"{wc}; {sc}")
    rules = [
        ("Health", r"public.*health|environmental.*health|toxicology|respiratory|cardiac|medicine|epidemiology"),
        ("Atmospheric and climate", r"meteorology|atmospheric|climatology|climate"),
        ("Forestry and land management", r"forestry|agriculture|land management|planning|management"),
        ("Ecology and biology", r"ecology|biodiversity|biology|plant sciences|zoology|conservation"),
        ("Earth observation and geosciences", r"remote sensing|geosciences|geography physical|imaging|geology"),
        ("Engineering, policy and society", r"engineering|economics|social sciences|policy|sociology|urban studies"),
        ("Environmental sciences", r"environmental sciences|green.*sustainable|water resources"),
    ]
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return "Other or multidisciplinary"

df["broad_discipline"] = [broad_discipline(w, s) for w, s in zip(df["WC"], df["SC"])]
df["analysis_period"] = pd.cut(
    df["publication_year"].astype(float),
    bins=[1899, 1989, 1999, 2009, 2019, 2025, 2026],
    labels=["1900-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2025", "2026 partial"],
)

primary = df.loc[primary_mask].copy()
topic_text = primary["core_text"].fillna("")
domain_stop = {
    "wildfire", "wildfires", "fire", "fires", "forest", "forests", "study", "studies",
    "result", "results", "data", "using", "based", "analysis", "model", "models",
    "year", "years", "area", "areas", "global", "change", "changes", "effect", "effects",
    "different", "high", "low", "large", "used", "use", "new", "research", "climate",
}
topic_stop = sorted(set(ENGLISH_STOP_WORDS) | domain_stop)
topic_vectorizer = TfidfVectorizer(
    stop_words=topic_stop,
    ngram_range=(1, 2),
    min_df=12,
    max_df=0.78,
    max_features=18000,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
)
X = topic_vectorizer.fit_transform(topic_text)
topic_terms = np.asarray(topic_vectorizer.get_feature_names_out())


def fit_nmf(k: int, seed_value: int):
    model = NMF(
        n_components=k, init="random", random_state=seed_value, max_iter=1000,
        tol=2e-4, solver="cd", beta_loss="frobenius",
    )
    W = model.fit_transform(X)
    return model, W, model.components_


def matched_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    sim = cosine_similarity(reference, candidate)
    rows, cols = linear_sum_assignment(-sim)
    return float(sim[rows, cols].mean())


grid_rows = []
models_by_k = {}
for k in range(10, 17):
    fits = [fit_nmf(k, seed_value) for seed_value in (17, 41, 73)]
    ref_model, ref_w, ref_h = fits[0]
    stability = float(np.mean([matched_similarity(ref_h, item[2]) for item in fits[1:]]))
    top_sets = [set(topic_terms[row.argsort()[::-1][:10]]) for row in ref_h]
    diversity = len(set().union(*top_sets)) / (10 * k)
    grid_rows.append({
        "k": k,
        "stability": stability,
        "topic_diversity": float(diversity),
        "reconstruction_error": float(ref_model.reconstruction_err_),
    })
    models_by_k[k] = (ref_model, ref_w, ref_h)
grid = pd.DataFrame(grid_rows)
err_range = grid["reconstruction_error"].max() - grid["reconstruction_error"].min()
grid["error_gain_scaled"] = (
    (grid["reconstruction_error"].max() - grid["reconstruction_error"]) / err_range
    if err_range > 0 else 0.0
)
grid["selection_score"] = 0.55 * grid["stability"] + 0.30 * grid["topic_diversity"] + 0.15 * grid["error_gain_scaled"]
selected_k = int(grid.sort_values(["selection_score", "stability"], ascending=False).iloc[0]["k"])
topic_model, W, H = models_by_k[selected_k]
primary["topic_id"] = W.argmax(axis=1)
primary["topic_loading"] = W.max(axis=1)
topic_rows = []
for topic_id, row in enumerate(H):
    order = row.argsort()[::-1][:20]
    topic_rows.append({
        "topic_id": topic_id,
        "top_terms": "; ".join(topic_terms[order]),
        "top_10_terms": "; ".join(topic_terms[order[:10]]),
        "record_n": int((primary["topic_id"] == topic_id).sum()),
        "share": float((primary["topic_id"] == topic_id).mean()),
    })
topics = pd.DataFrame(topic_rows).sort_values("record_n", ascending=False)

# Two-dimensional latent semantic projection for a deterministic, capped visual sample.
svd = TruncatedSVD(n_components=2, random_state=20260810)
coords = svd.fit_transform(X)
joblib.dump(topic_vectorizer, MODEL_OUT / "topic_vectorizer.joblib")
joblib.dump(topic_model, MODEL_OUT / "nmf_model.joblib")
joblib.dump(svd, MODEL_OUT / "semantic_projection_svd.joblib")
primary["lsa_x"] = coords[:, 0]
primary["lsa_y"] = coords[:, 1]
sample_n = min(6000, len(primary))
semantic_sample = primary.sample(sample_n, random_state=20260810) if len(primary) > sample_n else primary.copy()

# Push primary model assignments back to the full record table.
df["topic_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
df["topic_loading"] = np.nan
df.loc[primary.index, "topic_id"] = primary["topic_id"].astype("Int64")
df.loc[primary.index, "topic_loading"] = primary["topic_loading"]

analysis_base = df.loc[primary_mask].copy()
concept_cols = list(concept_patterns)
concept_summary = pd.DataFrame([
    {"concept": c, "record_n": int(analysis_base[c].sum()), "share": float(analysis_base[c].mean())}
    for c in concept_cols
]).sort_values("record_n", ascending=False)

ladder_masks = [
    ("Primary wildfire-climate corpus", pd.Series(True, index=analysis_base.index)),
    ("Ignition/source explicitly resolved", analysis_base["source_explicit"]),
    ("+ climate-attribution language", analysis_base["source_explicit"] & analysis_base["climate_attribution"]),
    ("+ fire-response endpoint", analysis_base["source_explicit"] & analysis_base["climate_attribution"] & analysis_base["fire_response_layer"]),
    ("+ emission/biophysical endpoint", analysis_base["source_explicit"] & analysis_base["climate_attribution"] & analysis_base["fire_response_layer"] & analysis_base["emission_biophysical_layer"]),
    ("+ explicit counterfactual", analysis_base["source_explicit"] & analysis_base["climate_attribution"] & analysis_base["fire_response_layer"] & analysis_base["emission_biophysical_layer"] & analysis_base["counterfactual"]),
    ("+ feedback closure", analysis_base["source_explicit"] & analysis_base["climate_attribution"] & analysis_base["fire_response_layer"] & analysis_base["emission_biophysical_layer"] & analysis_base["counterfactual"] & analysis_base["feedback_closure"]),
]
evidence_ladder = pd.DataFrame({
    "stage": [name for name, _ in ladder_masks],
    "record_n": [int(mask.sum()) for _, mask in ladder_masks],
})
evidence_ladder["share"] = evidence_ladder["record_n"] / len(analysis_base)

jaccard = pd.DataFrame(index=concept_cols, columns=concept_cols, dtype=float)
for a in concept_cols:
    for b in concept_cols:
        union = (analysis_base[a] | analysis_base[b]).sum()
        jaccard.loc[a, b] = float((analysis_base[a] & analysis_base[b]).sum() / union) if union else np.nan

annual = (
    df.loc[~df["excluded_duplicate"] & ~df["retracted_or_withdrawn"]]
    .groupby(["publication_year", "relevance_tier"], observed=True)
    .size().unstack(fill_value=0).reset_index()
)
for name in ["Core_direct", "Contextual_or_indirect", "Peripheral_screening_candidate"]:
    if name not in annual:
        annual[name] = 0
annual["total"] = annual[["Core_direct", "Contextual_or_indirect", "Peripheral_screening_candidate"]].sum(axis=1)
annual["core_share"] = annual["Core_direct"] / annual["total"].replace(0, np.nan)

period_topic = (
    primary.groupby(["analysis_period", "topic_id"], observed=True).size().rename("record_n").reset_index()
)
period_topic["period_total"] = period_topic.groupby("analysis_period", observed=True)["record_n"].transform("sum")
period_topic["share_within_period"] = period_topic["record_n"] / period_topic["period_total"]

discipline_summary = (
    analysis_base.groupby("broad_discipline").size().rename("record_n").reset_index().sort_values("record_n", ascending=False)
)
discipline_summary["share"] = discipline_summary["record_n"] / len(analysis_base)

discipline_concept = []
for discipline, group in analysis_base.groupby("broad_discipline"):
    for concept in concept_cols:
        # Jeffreys smoothing stabilizes rare markers.
        within = (group[concept].sum() + 0.5) / (len(group) + 1.0)
        overall = (analysis_base[concept].sum() + 0.5) / (len(analysis_base) + 1.0)
        discipline_concept.append({
            "discipline": discipline, "concept": concept, "record_n": int(group[concept].sum()),
            "group_n": int(len(group)), "prevalence": float(group[concept].mean()),
            "log2_prevalence_ratio": float(np.log2(within / overall)),
        })
discipline_concept = pd.DataFrame(discipline_concept)

ignition_endpoint = (
    analysis_base.groupby("ignition_class")[
        ["climate_attribution", "fire_response_layer", "emission_biophysical_layer", "recovery_layer", "counterfactual_feedback_layer"]
    ].agg(["sum", "mean"])
)
ignition_endpoint.columns = [f"{a}_{b}" for a, b in ignition_endpoint.columns]
ignition_endpoint = ignition_endpoint.reset_index()

topic_concept = (
    primary.groupby("topic_id")[concept_cols].mean().reset_index()
)

relevance_summary = (
    df.groupby("relevance_tier").size().rename("record_n").reset_index().sort_values("record_n", ascending=False)
)
relevance_summary["share"] = relevance_summary["record_n"] / len(df)

data_quality = {
    "records_received": int(len(df)),
    "duplicate_rows_excluded": int(df["excluded_duplicate"].sum()),
    "retracted_rows_excluded": int(df["retracted_or_withdrawn"].sum()),
    "eligible_article_review_records": int(df["eligible_document_type"].sum()),
    "complete_year_records_1900_2025": int(df["complete_year_1900_2025"].sum()),
    "partial_2026_records": int(df["partial_year_2026"].sum()),
    "abstract_missing": int((~df["abstract_available"]).sum()),
    "core_direct": int(df["relevance_tier"].eq("Core_direct").sum()),
    "contextual_or_indirect": int(df["relevance_tier"].eq("Contextual_or_indirect").sum()),
    "peripheral_screening_candidate": int(df["relevance_tier"].eq("Peripheral_screening_candidate").sum()),
    "primary_text_analysis_n": int(primary_mask.sum()),
    "selected_nmf_k": selected_k,
    "nmf_vocabulary_n": int(X.shape[1]),
    "nmf_stability": float(grid.loc[grid["k"].eq(selected_k), "stability"].iloc[0]),
    "nmf_topic_diversity": float(grid.loc[grid["k"].eq(selected_k), "topic_diversity"].iloc[0]),
    "lsa_explained_variance_2d": float(svd.explained_variance_ratio_.sum()),
}

df.to_pickle(OUT / "records_analyzed.pkl")
df.to_csv(OUT / "records_analyzed.csv", index=False)
grid.to_csv(OUT / "nmf_k_selection.csv", index=False)
topics.to_csv(OUT / "nmf_topics_unlabelled.csv", index=False)
period_topic.to_csv(OUT / "nmf_topic_period.csv", index=False)
semantic_sample[["record_id", "publication_year", "TI", "relevance_tier", "topic_id", "topic_loading", "lsa_x", "lsa_y"]].to_csv(OUT / "semantic_map_sample.csv", index=False)
concept_summary.to_csv(OUT / "sfca_concept_summary.csv", index=False)
evidence_ladder.to_csv(OUT / "sfca_evidence_ladder.csv", index=False)
jaccard.to_csv(OUT / "sfca_concept_jaccard.csv")
annual.to_csv(OUT / "annual_relevance_counts.csv", index=False)
discipline_summary.to_csv(OUT / "discipline_summary.csv", index=False)
discipline_concept.to_csv(OUT / "discipline_concept_index.csv", index=False)
ignition_endpoint.to_csv(OUT / "ignition_endpoint_matrix.csv", index=False)
topic_concept.to_csv(OUT / "topic_concept_matrix.csv", index=False)
relevance_summary.to_csv(OUT / "relevance_summary.csv", index=False)
with (OUT / "ml_relevance_validation.json").open("w", encoding="utf-8") as f:
    json.dump(cv_summary, f, indent=2, ensure_ascii=False)
with (OUT / "analysis_summary.json").open("w", encoding="utf-8") as f:
    json.dump(data_quality, f, indent=2, ensure_ascii=False)

print(json.dumps(data_quality, indent=2, ensure_ascii=False))
print("\nTOPICS\n", topics.to_string(index=False))
print("\nRELEVANCE\n", relevance_summary.to_string(index=False))
print("\nEVIDENCE LADDER\n", evidence_ladder.to_string(index=False))
