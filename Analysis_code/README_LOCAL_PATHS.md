# Local path configuration

This code package is configured for the following Windows project layout:

```text
C:\Users\win10\Desktop\43_analysis_code\
├── Analysis_code\
│   ├── literature_pipeline\
│   ├── validation_pipeline\
│   ├── model_spec.json
│   └── path_config.py
├── raw_data\
│   └── V2.0\                 # 14 Web of Science plain-text batches
├── process_data\             # parsed data, analytical tables, models and validation files
└── output\                   # final figures and validation summary
```

The supplied `raw_data.zip` should be extracted into
`C:\Users\win10\Desktop\43_analysis_code\raw_data`. After extraction, the
14 text files should be directly under `raw_data\V2.0`.

## Standard run order

Run the commands below in a Windows Command Prompt or PowerShell after the
packages in `requirements.txt` have been installed.

```bat
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\literature_pipeline\parse_wos.py
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\literature_pipeline\analyze.py
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\literature_pipeline\make_figures.py
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\validation_pipeline\prepare_validation_sample.py
```

The first three commands read the 14 raw batches, write parsed and analytical
process files to `process_data`, fit the models under
`process_data\models\fitted`, and write final PNG/TIFF/SVG figures to
`output\figures`. Figure source tables and method metadata are copied to
`process_data\figure_source_data`.

After the two independently completed validation label files have been placed
under `process_data\validation`, recompute the agreement statistics with:

```bat
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\validation_pipeline\recompute_validation_metrics.py
```

The resulting JSON file is written to `output\validation_metrics.json`.

The model-rebuild script still requires the processed Excel workbook as its
positional input. Its default model output is the same fitted-model folder:

```bat
python C:\Users\win10\Desktop\43_analysis_code\Analysis_code\literature_pipeline\rebuild_models_from_processed_workbook.py "C:\path\to\processed_workbook.xlsx"
```

All default paths are defined once in `path_config.py`. Optional environment
variable overrides are included for testing or for moving the project later;
they are not needed for the folder layout above.
