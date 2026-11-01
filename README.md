# Hospital readmission prediction

Predict recorded 30-day readmission following a hospital encounter involving a patient with diabetes, using information available at discharge.

## Current milestone

Dataset acquisition and initial understanding are complete. Open `notebooks/01_data_understanding.ipynb` for executed outputs, or `reports/initial_data_audit.md` for findings. `reports/column_inventory.csv` documents every column.

Target: `<30` = 1; `>30` and `NO` = 0. Retain `patient_nbr` for patient-disjoint splitting, excluding identifiers and `readmitted` from predictors.

## Reproduce

Install `requirements.txt`, download the archive using the URL in `data/raw/source.json` to `data/raw/uci_diabetes_130.zip`, then run `python scripts/understand_data.py` from the project root. This script executes inspection cells and writes a standard notebook with saved outputs; Jupyter is only needed for interactive notebook use.

## Dataset attribution

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. Licensed under CC BY 4.0. Raw data is excluded from Git; source URLs and checksums are retained locally in `data/raw/source.json`.

## Cohort and splits

Run `python scripts/create_cohort_splits.py` to regenerate the eligible cohort and fixed patient-disjoint partitions. See `reports/cohort_and_splits.md` for the exclusion policy, partition counts, and leakage checks. Assignments are stored in `data/processed/partition_assignments.csv` and excluded encounters in `data/processed/excluded_encounters.csv`; these local data files are excluded from Git.

## Baseline models

Run `python scripts/train_baselines.py` to fit full preprocessing/model pipelines for a prior-probability dummy, logistic regression, and class-weighted logistic regression. Results are saved to `reports/baselines/baseline_report.md` and `validation_metrics.csv`; pipelines are stored locally under `models/`. The reserved test partition is not evaluated. The run metadata records package versions and input checksums.

## Engineered model comparison

Run `python scripts/compare_models.py` to compare engineered logistic regression, Random Forest, and histogram gradient boosting on identical three-fold patient-grouped training splits. The winner by mean average precision is refitted on training data and evaluated on validation data, including threshold tradeoffs. See `reports/model_comparison/comparison_report.md`. The full pipeline is saved to `models/selected_engineered_model.joblib`; keep the `src` package importable when loading it from Python.

## Calibration and demographic audit

Run `python scripts/calibrate_and_audit.py` to fit training-only patient-grouped sigmoid calibration and compare it with the original model on validation Brier score and log loss. It uses a provisional educational objective of at least 70% validation recall, selecting the highest threshold meeting that target. Results, reliability plots, demographic tables, and limitations are in `reports/calibration/calibration_and_fairness_report.md`. The selected model path and exact threshold are saved in `models/operating_policy.json`.

## Explainability and frozen evaluation

`python scripts/explain_and_freeze.py` generates exact tree SHAP explanations for a fixed validation sample and six score-selected examples, verifies prediction reconstruction, and writes `models/frozen_manifest.json` before test access. This command refuses to replace an existing freeze.

`python scripts/final_test_evaluation.py` verifies the frozen hashes and evaluates the held-out test partition using the unchanged model and threshold. Saved metrics prevent fresh scoring on rerun; an exclusive start marker guards incomplete runs. Results are in `reports/final_test/final_test_report.md`, with calibration, demographic tables, patient-cluster bootstrap intervals, and plots. The test set is consumed after this evaluation and must not guide further tuning.

See `reports/explainability/explainability_report.md` for SHAP findings and the operating-policy rationale. Model associations are not causal explanations.

## Local Streamlit demonstration

Install app dependencies with `python -m pip install -r requirements-app.txt`. The original frozen `requirements.txt` is preserved. Run from the project root:

```text
python -m streamlit run app/app.py --server.address 127.0.0.1
```

Open the localhost URL printed by Streamlit (usually http://127.0.0.1:8501). The three pages provide a raw-input predictor, live SHAP explanations of the last submitted encounter, and saved final-test performance/calibration/demographic results. Every model input is editable; hidden expanders retain the loaded training example's values. No arbitrary low/medium/high medical risk bands are assigned. The frozen threshold and model stay unchanged.

`app/assets/input_schema.json` supplies choices, bounds, and three ID-free examples derived only from the training partition; regenerate with `python scripts/prepare_app_assets.py`. The app does not need the raw dataset to run once these assets, model, frozen manifest, source feature code, and saved reports are present. Inputs live in Streamlit session state and are not written to a patient database. Only load the trusted model distributed with this project.

Run `python scripts/check_streamlit_app.py` for automated interaction and frozen-pipeline consistency checks. Results are saved in `reports/app_verification.json`.

## Next step

Prepare repository documentation and deployment packaging. No public hosting or deployment has been performed by the local app workflow.

This is an educational project using historical US data and is not validated for clinical use.
