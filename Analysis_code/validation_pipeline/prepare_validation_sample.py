from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from path_config import ANALYSIS_DIR, VALIDATION_SAMPLE_DIR


TIERS = ["Core_direct", "Contextual_or_indirect", "Peripheral_screening_candidate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the probability-stratified validation sample.")
    parser.add_argument(
        "--records",
        type=Path,
        default=ANALYSIS_DIR / "records_analyzed.pkl",
        help="Analyzed record table produced by analyze.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VALIDATION_SAMPLE_DIR,
    )
    args = parser.parse_args()

    records = pd.read_pickle(args.records)
    population = records.loc[
        (~records["excluded_duplicate"])
        & (~records["retracted_or_withdrawn"])
        & records["eligible_document_type"]
        & records["complete_year_1900_2025"]
        & records["abstract_available"]
    ].copy()
    if len(population) != 11625:
        raise RuntimeError(f"Expected validation population n=11,625, found {len(population):,}")

    rng = np.random.default_rng(20260812)
    parts = []
    for tier in TIERS:
        group = population.loc[population["relevance_tier"].eq(tier)].copy()
        chosen = rng.choice(group.index.to_numpy(), size=100, replace=False)
        part = group.loc[chosen].copy()
        part["stratum_population_n"] = len(group)
        part["stratum_sample_n"] = 100
        part["inclusion_probability"] = 100 / len(group)
        part["design_weight"] = len(group) / 100
        parts.append(part)

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1, random_state=20260813).reset_index(drop=True)
    sample.insert(0, "validation_id", [f"VAL{i:03d}" for i in range(1, 301)])

    manifest = sample[[
        "validation_id", "record_id", "UT", "DI", "TI", "AB", "DE",
        "publication_year", "SO", "relevance_tier", "ml_relevance_probability",
        "stratum_population_n", "stratum_sample_n", "inclusion_probability", "design_weight",
    ]].copy()
    blinded = manifest[[
        "validation_id", "record_id", "TI", "AB", "DE", "publication_year", "SO"
    ]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "validation_manifest.csv", index=False)
    blinded.sample(frac=1, random_state=20260814).to_csv(
        args.output_dir / "reviewer_a_blinded.csv", index=False
    )
    blinded.sample(frac=1, random_state=20260815).to_csv(
        args.output_dir / "reviewer_b_blinded.csv", index=False
    )
    print(population["relevance_tier"].value_counts().reindex(TIERS).to_string())


if __name__ == "__main__":
    main()
