# Validation baseline comparison

Trained on 69,651 encounters; evaluated on 14,806 validation encounters. Validation readmission prevalence: 11.14%. Reserved test partition was not used for fitting, feature selection, or evaluation.

```text
                       model  average_precision  roc_auc  precision  recall     f1  brier_score  log_loss  threshold    tn   fp   fn  tp
                 dummy_prior             0.1114   0.5000     0.0000  0.0000 0.0000       0.0990    0.3496        0.5 13156    0 1650   0
         logistic_regression             0.2135   0.6473     0.5051  0.0303 0.0572       0.0955    0.3347        0.5 13107   49 1600  50
logistic_regression_balanced             0.2099   0.6422     0.1693  0.5352 0.2572       0.2202    0.6335        0.5  8824 4332  767 883
```

## Method

All three full pipelines fit their preprocessing only on training data. Numerical features: median imputation and standard scaling. String categories: explicit `Missing` category and one-hot encoding. Admission/discharge integer codes: one-hot encoding, preserving the distinct source unknown/unavailable codes as explicit categories. Unseen validation categories are ignored by the encoder. Literal laboratory `None` categories remain intact. All 47 source predictors are retained for this initial baseline, including sparse weight and detailed diagnosis codes; any later feature simplification should be based on training data.

Identifiers, source outcome, partition, and target are excluded from predictors. Models: prior-probability dummy, ordinary logistic regression, and balanced-class logistic regression. Logistic regression uses liblinear, C=1, max_iter=2000, random_state=42. No resampling, tuning, threshold selection, or calibration was performed.

Average precision is the non-interpolated precision-recall summary from sklearn, not trapezoidal PR-AUC. Precision/recall/F1 and confusion matrices use a fixed 0.50 threshold; tn/fp/fn/tp are shown above. Undefined precision for zero predicted positives is reported as zero. The dummy predicts the training prevalence and ranks every case equally.

Balanced weights change the error tradeoff and can distort probability calibration; their raw output should not be presented as calibrated patient risk. Compare ranking metrics separately from threshold-dependent metrics. Brier score and log loss are included as initial probability-quality diagnostics, not a calibration validation.

## Verification and artifacts

Verified source checksum, unique assignment IDs, patient-disjoint partitions, complete training/validation joins, convergence, finite probabilities, and saved-model reload prediction agreement. Full preprocessing/model pipelines are saved under `models/*.joblib`. Only load trusted joblib artifacts. Runtime package versions and feature lists are in `run_metadata.json`.

Next: investigate training-only feature engineering and patient-grouped cross-validation, compare additional models, and use validation for model/threshold selection. Keep the test partition reserved until the methodology is fixed. These historical-data models are educational and not clinically validated.
