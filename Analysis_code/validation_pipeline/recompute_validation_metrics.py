from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from path_config import VALIDATION_DIR, VALIDATION_METRICS_FILE


ORDER = ["Core_direct", "Contextual_or_indirect", "Peripheral_screening_candidate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute independent-review agreement statistics.")
    parser.add_argument(
        "--relevance",
        type=Path,
        default=VALIDATION_DIR / "relevance_validation_labels.csv",
    )
    parser.add_argument(
        "--nmf",
        type=Path,
        default=VALIDATION_DIR / "nmf_semantic_audit_labels.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_METRICS_FILE,
        help="JSON file for the recomputed validation metrics",
    )
    args = parser.parse_args()

    relevance = pd.read_csv(args.relevance)
    a = relevance["Human_relevance_label_A"]
    b = relevance["Human_relevance_label_B"]
    a_binary = a.map(lambda value: "Exclude" if value == "Peripheral_screening_candidate" else "Include")
    b_binary = b.map(lambda value: "Exclude" if value == "Peripheral_screening_candidate" else "Include")

    nmf = pd.read_csv(args.nmf)
    nmf_a = nmf["Label_fit_0_2_A"]
    nmf_b = nmf["Label_fit_0_2_B"]

    result = {
        "relevance_n": int(len(relevance)),
        "relevance_agreement_n": int((a == b).sum()),
        "relevance_raw_agreement": float((a == b).mean()),
        "relevance_kappa": float(cohen_kappa_score(a, b, labels=ORDER)),
        "relevance_matrix": confusion_matrix(a, b, labels=ORDER).tolist(),
        "binary_agreement_n": int((a_binary == b_binary).sum()),
        "binary_raw_agreement": float((a_binary == b_binary).mean()),
        "binary_kappa": float(cohen_kappa_score(a_binary, b_binary, labels=["Include", "Exclude"])),
        "binary_matrix": confusion_matrix(a_binary, b_binary, labels=["Include", "Exclude"]).tolist(),
        "nmf_n": int(len(nmf)),
        "nmf_agreement_n": int((nmf_a == nmf_b).sum()),
        "nmf_raw_agreement": float((nmf_a == nmf_b).mean()),
        "nmf_kappa": float(cohen_kappa_score(nmf_a, nmf_b, labels=[0, 1, 2])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
