# Calibration, threshold, and demographic audit

## Probability calibration

The selected gradient boosting pipeline was cloned and calibrated with sigmoid scaling using pooled out-of-fold scores from three patient-grouped training folds (seed 42, ensemble=False). All preprocessing and base-model fits are inside those folds. The calibrated estimator then refits the base model on all 69,651 training encounters. Validation outcomes did not fit the calibrator. The previously saved full-training base pipeline provides the uncalibrated comparison. Both candidates are evaluated on 14,806 validation encounters.

```text
     variant  average_precision  roc_auc  brier_score  log_loss  mean_probability  readmission_rate
uncalibrated            0.23146  0.66816      0.09399   0.32904           0.11416           0.11144
     sigmoid            0.23146  0.66816      0.09400   0.32909           0.11468           0.11144
```

Selection rule, fixed before inspecting results: use sigmoid only if both validation Brier score and log loss improve; otherwise retain uncalibrated. Selected: **uncalibrated**. Lower Brier/log loss is better, but both measure overall probabilistic accuracy and cannot alone establish perfect calibration. See calibration_curve.png and calibration_bins.csv for approximately equal-count reliability bins; tied probabilities stay together. Calibration selection used development data, so these are not unbiased final performance estimates. Monotone sigmoid calibration should preserve ranking; it changes the probability scale, not clinical discrimination.

## Provisional operating threshold

Because no intervention capacity or cost criterion was supplied, use the explicit educational assumption: recall at least 70%, choosing the highest validation score threshold that achieves it. This minimizes flagged encounters under that recall constraint. It does not optimize clinical utility or establish a medically validated cutoff. Equal-to-threshold scores are flagged; tied scores cannot be split.

- Selected variant: uncalibrated
- Threshold: 0.0973794101 (use the full value in models/operating_policy.json)
- Recall: 70.00%; precision: 15.93%
- Flagged: 7,251 / 14,806 encounters (48.97%)
- True positives: 1,155; false positives: 6,096
- False negatives: 495; true negatives: 7,060

```text
 recall_objective  threshold  precision  recall  flagged_rate    fp  fn
              0.5    0.12732    0.19213     0.5       0.29002  3469 825
              0.6    0.11110    0.17284     0.6       0.38687  4738 660
              0.7    0.09738    0.15929     0.7       0.48973  6096 495
              0.8    0.08193    0.14474     0.8       0.61597  7800 330
              0.9    0.06419    0.12745     0.9       0.78698 10167 165
```

The alert workload is an essential tradeoff. This provisional policy needs revision if the intended intervention cannot support that volume. The recall target is a development criterion, not a guarantee on future or test data. Do not change it after inspecting final test performance.

## Demographic performance

Audit uses the same selected model and global threshold for every age, gender, and race group. subgroup_metrics.csv reports encounter and unique-patient counts, positive counts, prevalence, precision, recall, false-positive rate, flagged rate, mean predicted probability, calibration gap (mean probability minus observed rate), Brier score, log loss, AP, and ROC-AUC. subgroup_calibration_bins.csv provides reliability bins for sufficiently large groups.

Descriptive recall ranges among groups with at least 100 encounters, 30 positives, and 30 negatives:

```text
attribute  included_groups  recall_min  recall_max  recall_gap  precision_min  precision_max  false_positive_rate_min  false_positive_rate_max
      age                8     0.60345     0.84848     0.24504        0.10744        0.30108                  0.32835                  0.62974
   gender                2     0.64876     0.74026     0.09150        0.15038        0.16606                  0.43644                  0.48668
     race                4     0.50000     0.84783     0.34783        0.13559        0.20968                  0.33010                  0.47348
```

Small groups are flagged, not silently removed; undefined metrics are blank rather than zero. These are encounter-weighted descriptive estimates, with repeated encounters per patient and no confidence intervals. Group gaps may reflect sampling variation, prevalence, coding, and case mix; they do not prove discrimination, fairness, or equal clinical benefit. Calibration-gap averages can hide within-group miscalibration; examine the bins as well. Do not interpret small-group differences as reliable or use them to create demographic-specific thresholds without further study.

## Artifacts and verification

models/sigmoid_calibrated_model.joblib stores the full calibration/preprocessing/model estimator. models/operating_policy.json points to the selected model, its checksum, required input features, exact threshold, and decision rule. Existing uncalibrated and comparison artifacts are preserved. Keep the src package importable when loading trusted joblib files.

Verified source/assignment checksums, unique assignments, patient-disjoint outer and calibration folds, complete partition loading, threshold tie handling and minimal flagging at the target, finite probabilities, safe undefined subgroup metrics, and calibrated model prediction equality after reload. Test outcomes have not been evaluated.

Next: review the provisional workload and freeze the intended operating policy before final test evaluation. Explainability can be added before opening the test set. This historical-data project is educational and is not clinically validated.
