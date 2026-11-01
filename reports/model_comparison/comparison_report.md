# Engineered features and grouped model comparison

## Training-only comparison

All candidates used identical three-fold StratifiedGroupKFold splits of the reserved training partition (69,651 encounters). Every patient's encounters stay in one fold. Feature transforms are deterministic; each fold fits its own imputation, scaling, encoding, and model on the fitting subset only. No validation or test outcome was used to rank models.

```text
                                   average_precision_mean  average_precision_std  roc_auc_mean  roc_auc_std  brier_score_mean  brier_score_std  log_loss_mean  log_loss_std
model                                                                                                                                                                      
hist_gradient_boosting_engineered                  0.2321                 0.0093        0.6753       0.0022            0.0963           0.0020         0.3343        0.0057
random_forest_engineered                           0.2183                 0.0118        0.6722       0.0002            0.0974           0.0022         0.3381        0.0056
logistic_engineered                                0.2150                 0.0155        0.6607       0.0036            0.0974           0.0017         0.3385        0.0049
```

Selected **hist_gradient_boosting_engineered** by highest mean fold average precision. Standard deviations describe fold variation, not confidence intervals. Average precision is the non-interpolated precision-recall summary, not trapezoidal PR-AUC. This is a fixed-parameter comparison, not an exhaustive tuning study; fold differences do not establish statistical superiority.

## Feature changes

Add total previous outpatient/emergency/inpatient visits, count of medication columns marked Steady/Up/Down, and count marked Up/Down. Replace each detailed diagnosis with project-defined broad numeric ICD-9 ranges: circulatory [390,460), respiratory [460,520), digestive [520,580), genitourinary [580,630), musculoskeletal [710,740), injury [800,1000), neoplasm [140,240), diabetes [250,251); all other codes (including V/E prefixes) map to Other, with Missing separate. These simple groups are not a comprehensive clinical ontology. Medication counts count active columns, not individual active ingredients.

Retain the remaining raw predictors, including sparse weight, for this fixed experiment. Admission/discharge codes remain categorical. Numerical fields receive median imputation/scaling; categorical missing values receive an explicit category; unseen categories are ignored. Original lab None categories are preserved. IDs and outcomes never enter predictors. See training_feature_audit.csv for training-only column statistics. This comparison does not isolate the effect of engineering: an ablation would compare raw and engineered features on identical CV folds.

## Selected model on validation

Refit the CV-selected pipeline on all training encounters, then score the reserved validation partition only.

```text
                            model  average_precision  roc_auc  brier_score  log_loss  threshold  precision  recall     f1    tn  fp   fn  tp
hist_gradient_boosting_engineered             0.2315   0.6682        0.094     0.329        0.5     0.5763  0.0206 0.0398 13131  25 1616  34
```

Validation threshold tradeoffs:

```text
 threshold  precision  recall     f1    fp   fn   tp
      0.05     0.1177  0.9667 0.2099 11953   55 1595
      0.10     0.1608  0.6764 0.2598  5824  534 1116
      0.15     0.2136  0.3915 0.2764  2378 1004  646
      0.20     0.2768  0.2315 0.2521   998 1268  382
      0.25     0.3596  0.1467 0.2084   431 1408  242
      0.30     0.4131  0.0994 0.1602   233 1486  164
      0.40     0.5474  0.0455 0.0839    62 1575   75
      0.50     0.5763  0.0206 0.0398    25 1616   34
```

No operating threshold has been selected. A recall target or intervention capacity/cost criterion is needed to choose one; these thresholds are exploratory project values, not clinical standards. Validation is now development data, so its threshold metrics are not an unbiased final performance estimate. Test data remains reserved. Probability calibration has not been validated.

## Artifacts and verification

Saved the full raw-feature-to-probability pipeline as models/selected_engineered_model.joblib. From the project root load it with joblib (the src package must be importable). Input columns are recorded in run_metadata.json. Only load trusted artifacts.

Verified source checksum, assignment uniqueness, patient-disjoint outer and CV partitions, complete train/validation joins, diagnosis grouping boundaries, convergence, finite probabilities, and prediction agreement after model reload. CSV files contain fold metrics, summaries, validation metrics, and threshold tradeoffs. Test outcomes were not evaluated.

Next: review the validation precision/recall tradeoff and define an operating objective, then assess calibration and demographic performance. Any further model tuning should use training-only grouped CV. Educational historical-data experiment; not clinically validated.
