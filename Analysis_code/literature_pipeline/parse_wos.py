from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from path_config import PARSED_DIR, WOS_TEXT_DIR

RAW = WOS_TEXT_DIR
OUT = PARSED_DIR
OUT.mkdir(parents=True, exist_ok=True)

LIST_FIELDS = {"AU", "AF", "C1", "C3", "CR"}


def parse_file(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, list[str]] | None = None
    active_tag: str | None = None
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if line == "ER":
            if current:
                record: dict[str, str] = {}
                for tag, values in current.items():
                    sep = "; " if tag in LIST_FIELDS else " "
                    record[tag] = sep.join(v.strip() for v in values if v.strip()).strip()
                record["source_batch"] = path.name
                records.append(record)
            current = None
            active_tag = None
            continue
        if line in {"EF", "FN Clarivate Analytics Web of Science", "VR 1.0"}:
            continue
        if re.match(r"^[A-Z0-9]{2} ", line):
            tag, value = line[:2], line[3:]
            if tag == "PT":
                current = {}
            if current is None:
                continue
            current.setdefault(tag, []).append(value)
            active_tag = tag
        elif line.startswith("   ") and current is not None and active_tag:
            current.setdefault(active_tag, []).append(line[3:])
    return records


def norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = html.unescape(str(value))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_doi(value: object) -> str:
    s = norm_text(value)
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip().rstrip(".,;)")


def norm_title(value: object) -> str:
    s = norm_text(value)
    return re.sub(r"[^a-z0-9]+", "", s)


paths = sorted(RAW.glob("*.txt"), key=lambda p: int(re.search(r"_(\d+)_\d+\.txt$", p.name).group(1)))
if not paths:
    raise FileNotFoundError(f"No Web of Science plain-text exports found in {RAW}")
all_records: list[dict[str, str]] = []
batch_rows = []
for path in paths:
    rows = parse_file(path)
    start, end = map(int, re.search(r"_(\d+)_(\d+)\.txt$", path.name).groups())
    batch_rows.append({
        "source_batch": path.name,
        "expected_start": start,
        "expected_end": end,
        "expected_n": end - start + 1,
        "parsed_n": len(rows),
        "batch_complete": len(rows) == end - start + 1,
    })
    all_records.extend(rows)

df = pd.DataFrame(all_records)
for col in ["UT", "DI", "TI", "AB", "DE", "ID", "PY", "DT", "WC", "SC", "SO", "AU", "AF", "LA", "TC", "Z9", "OA", "WE", "DA", "PT", "CR", "C1"]:
    if col not in df:
        df[col] = ""

df["UT_normalized"] = df["UT"].fillna("").str.strip().str.upper()
df["DOI_normalized"] = df["DI"].map(norm_doi)
df["title_normalized_key"] = df["TI"].map(norm_title)
df["publication_year"] = pd.to_numeric(df["PY"], errors="coerce").astype("Int64")
df["times_cited"] = pd.to_numeric(df["TC"], errors="coerce").fillna(0).astype(int)
df["title_abstract_keywords"] = (
    df["TI"].map(norm_text) + ". " + df["AB"].map(norm_text) + ". " + df["DE"].map(norm_text)
).str.strip(". ")
df["searchable_text_with_kp"] = (
    df["title_abstract_keywords"] + ". " + df["ID"].map(norm_text)
).str.strip(". ")
df["abstract_available"] = df["AB"].fillna("").str.strip().ne("")
df["author_keywords_available"] = df["DE"].fillna("").str.strip().ne("")
df["keywords_plus_available"] = df["ID"].fillna("").str.strip().ne("")
df["partial_year_2026"] = df["publication_year"].eq(2026)
df["complete_year_1900_2025"] = df["publication_year"].between(1900, 2025)
df["retracted_or_withdrawn"] = (
    df["DT"].fillna("").str.contains(r"retract|withdraw", case=False, regex=True)
    | df["TI"].fillna("").str.contains(r"^retract|withdrawn", case=False, regex=True)
)

# Exact duplicate checks. UT is authoritative; DOI/title checks reveal export or metadata issues.
df["duplicate_UT"] = df["UT_normalized"].ne("") & df.duplicated("UT_normalized", keep=False)
df["duplicate_DOI"] = df["DOI_normalized"].ne("") & df.duplicated("DOI_normalized", keep=False)
df["duplicate_title"] = df["title_normalized_key"].ne("") & df.duplicated("title_normalized_key", keep=False)
df["dedupe_priority"] = np.arange(len(df))
keep = ~df.duplicated("UT_normalized", keep="first") | df["UT_normalized"].eq("")
dedup = df.loc[keep].copy()
dedup.insert(0, "record_id", [f"WCC{i:05d}" for i in range(1, len(dedup) + 1)])

field_summary = []
for col in df.columns:
    if col in {"title_abstract_keywords", "searchable_text_with_kp"}:
        continue
    nonempty = df[col].notna() & df[col].astype(str).str.strip().ne("")
    field_summary.append({
        "field": col,
        "nonempty_n": int(nonempty.sum()),
        "nonempty_share": float(nonempty.mean()),
    })

quality = {
    "zip_expected_records": 13012,
    "parsed_records": int(len(df)),
    "deduplicated_records": int(len(dedup)),
    "exact_duplicate_ut_rows": int(df["duplicate_UT"].sum()),
    "exact_duplicate_doi_rows": int(df["duplicate_DOI"].sum()),
    "exact_duplicate_title_rows": int(df["duplicate_title"].sum()),
    "missing_title": int(df["TI"].fillna("").str.strip().eq("").sum()),
    "missing_abstract": int((~df["abstract_available"]).sum()),
    "missing_author_keywords": int((~df["author_keywords_available"]).sum()),
    "missing_year": int(df["publication_year"].isna().sum()),
    "complete_year_1900_2025": int(df["complete_year_1900_2025"].sum()),
    "partial_2026": int(df["partial_year_2026"].sum()),
    "retracted_or_withdrawn": int(df["retracted_or_withdrawn"].sum()),
    "document_types": Counter(df["DT"].fillna("").astype(str)).most_common(),
}

pd.DataFrame(batch_rows).to_csv(OUT / "batch_audit.csv", index=False)
pd.DataFrame(field_summary).to_csv(OUT / "field_completeness.csv", index=False)
df.to_pickle(OUT / "all_records_raw.pkl")
dedup.to_pickle(OUT / "records_deduplicated.pkl")
with (OUT / "quality_summary.json").open("w", encoding="utf-8") as f:
    json.dump(quality, f, indent=2, ensure_ascii=False)

print(json.dumps(quality, indent=2, ensure_ascii=False))
