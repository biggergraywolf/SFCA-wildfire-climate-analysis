from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from path_config import MODEL_DIR


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def flag(series: pd.Series, expressions: list[str]) -> pd.Series:
    return series.str.contains("(?:" + "|".join(expressions) + ")", regex=True, na=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild fitted text models from the processed record workbook."
    )
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    records = pd.read_excel(args.workbook, sheet_name="Processed_records")
    if len(records) != 13012:
        raise RuntimeError(f"Expected 13,012 records, found {len(records):,}")

    title = records["Title"].map(normalize_text)
    abstract = records["Abstract"].map(normalize_text)
    author_keywords = records["Author_keywords"].map(normalize_text)
    keywords_plus = records["Keywords_Plus"].map(normalize_text)
    core_text = (title + ". " + abstract + ". " + author_keywords).str.strip(". ")
    all_text = (core_text + ". " + keywords_plus).str.strip(". ")

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

    fire_title = flag(title, fire_terms)
    fire_abstract = flag(abstract, fire_terms)
    fire_keywords = flag(author_keywords, fire_terms)
    fire_core = flag(core_text, fire_terms)
    fire_all = flag(all_text, fire_terms)
    climate_title = flag(title, climate_strong)
    climate_abstract = flag(abstract, climate_strong)
    climate_keywords = flag(author_keywords, climate_strong)
    climate_core = flag(core_text, climate_strong)
    climate_all = flag(all_text, climate_strong)
    climate_weak_core = flag(core_text, climate_weak)
    kp_only_bridge = fire_all & climate_all & ~(fire_core & climate_core)
    high_conf_positive = (
        (fire_title & climate_core)
        | (climate_title & fire_core)
        | (fire_abstract & climate_abstract)
        | (fire_keywords & climate_keywords)
    )
    high_conf_negative = kp_only_bridge | (fire_core & ~climate_core & climate_weak_core)
    seed_mask = high_conf_positive | high_conf_negative
    seed_labels = high_conf_positive.loc[seed_mask].astype(int).to_numpy()

    model_stop = sorted(set(ENGLISH_STOP_WORDS) | {
        "study", "studies", "results", "result", "data", "using", "based", "analysis",
        "model", "models", "year", "years", "area", "areas", "research", "paper",
    })
    relevance_vectorizer = TfidfVectorizer(
        stop_words=model_stop, ngram_range=(1, 2), min_df=5, max_df=0.90,
        max_features=30000, sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )
    relevance_matrix = relevance_vectorizer.fit_transform(all_text)
    classifier = LogisticRegression(
        max_iter=1500, class_weight="balanced", C=2.0, random_state=20260810
    )
    classifier.fit(relevance_matrix[seed_mask.to_numpy()], seed_labels)
    rebuilt_probability = classifier.predict_proba(relevance_matrix)[:, 1]
    stored_probability = records["ML_relevance_probability"].to_numpy(dtype=float)
    probability_max_abs_error = float(np.nanmax(np.abs(rebuilt_probability - stored_probability)))

    primary_mask = records["Analysis_status"].eq("primary_text_analysis")
    topic_text = core_text.loc[primary_mask]
    if int(primary_mask.sum()) != 9563:
        raise RuntimeError(f"Expected 9,563 primary records, found {int(primary_mask.sum()):,}")
    domain_stop = {
        "wildfire", "wildfires", "fire", "fires", "forest", "forests", "study", "studies",
        "result", "results", "data", "using", "based", "analysis", "model", "models",
        "year", "years", "area", "areas", "global", "change", "changes", "effect", "effects",
        "different", "high", "low", "large", "used", "use", "new", "research", "climate",
    }
    topic_vectorizer = TfidfVectorizer(
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | domain_stop), ngram_range=(1, 2),
        min_df=12, max_df=0.78, max_features=18000, sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
    )
    topic_matrix = topic_vectorizer.fit_transform(topic_text)
    topic_model = NMF(
        n_components=16, init="random", random_state=17, max_iter=1000,
        tol=2e-4, solver="cd", beta_loss="frobenius",
    )
    topic_weights = topic_model.fit_transform(topic_matrix)
    rebuilt_topics = topic_weights.argmax(axis=1)
    stored_topics = records.loc[primary_mask, "Topic_ID"].to_numpy(dtype=int)
    topic_assignment_agreement = float((rebuilt_topics == stored_topics).mean())
    projection = TruncatedSVD(n_components=2, random_state=20260810).fit(topic_matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(relevance_vectorizer, args.output_dir / "relevance_vectorizer.joblib")
    joblib.dump(classifier, args.output_dir / "relevance_classifier.joblib")
    joblib.dump(topic_vectorizer, args.output_dir / "topic_vectorizer.joblib")
    joblib.dump(topic_model, args.output_dir / "nmf_model.joblib")
    joblib.dump(projection, args.output_dir / "semantic_projection_svd.joblib")
    verification = {
        "records": int(len(records)),
        "seed_records": int(seed_mask.sum()),
        "primary_records": int(primary_mask.sum()),
        "relevance_probability_max_abs_error": probability_max_abs_error,
        "topic_assignment_agreement": topic_assignment_agreement,
        "selected_nmf_k": 16,
    }
    (args.output_dir / "model_rebuild_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
