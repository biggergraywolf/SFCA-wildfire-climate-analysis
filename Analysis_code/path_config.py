"""Central path configuration for the SFCA literature-analysis workflows.

The defaults below match the Windows folder layout used for this project.
Environment variables are optional and only provide a portable override for
testing or for moving the project later; normal Windows use requires no setup.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(
    os.environ.get("SFCA_PROJECT_DIR", r"C:\Users\win10\Desktop\43_analysis_code")
)
LITERATURE_PIPELINE_DIR = Path(
    os.environ.get(
        "SFCA_LITERATURE_PIPELINE_DIR",
        r"C:\Users\win10\Desktop\43_analysis_code\Analysis_code\literature_pipeline",
    )
)
VALIDATION_PIPELINE_DIR = Path(
    os.environ.get(
        "SFCA_VALIDATION_PIPELINE_DIR",
        r"C:\Users\win10\Desktop\43_analysis_code\Analysis_code\validation_pipeline",
    )
)
RAW_DATA_DIR = Path(
    os.environ.get("SFCA_RAW_DATA_DIR", r"C:\Users\win10\Desktop\43_analysis_code\raw_data")
)
OUTPUT_DIR = Path(
    os.environ.get("SFCA_OUTPUT_DIR", r"C:\Users\win10\Desktop\43_analysis_code\output")
)
PROCESS_DATA_DIR = Path(
    os.environ.get(
        "SFCA_PROCESS_DATA_DIR",
        r"C:\Users\win10\Desktop\43_analysis_code\process_data",
    )
)

# Input data and intermediate products.
WOS_TEXT_DIR = RAW_DATA_DIR / "V2.0"
PARSED_DIR = PROCESS_DATA_DIR / "parsed"
ANALYSIS_DIR = PROCESS_DATA_DIR / "analysis"
MODEL_DIR = PROCESS_DATA_DIR / "models" / "fitted"
VALIDATION_DIR = PROCESS_DATA_DIR / "validation"
VALIDATION_SAMPLE_DIR = PROCESS_DATA_DIR / "validation_sample"
FIGURE_SOURCE_DIR = PROCESS_DATA_DIR / "figure_source_data"
MPL_CONFIG_DIR = PROCESS_DATA_DIR / "mplconfig"

# Submission-ready analysis outputs.
FIGURE_DIR = OUTPUT_DIR / "figures"
VALIDATION_METRICS_FILE = OUTPUT_DIR / "validation_metrics.json"
