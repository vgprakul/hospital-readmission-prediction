# Final held-out test evaluation

The model, calibration choice, features, source code, data/partition checksums, and exact threshold were frozen in models/frozen_manifest.json before this test evaluation. No model fitting, calibration fitting, threshold adjustment, or model selection was performed on test outcomes. All outputs derive from one model-scoring pass.

## Results

- Encounters: 14,862; unique patients: 10,497; prevalence: 11.30%
- Average precision: 0.2291; ROC-AUC: 0.6705
- Brier score: 0.09536; log loss: 0.33221
- Frozen threshold: 0.097379410080
- Recall: 69.58%; precision: 15.89%; F1: 0.2587
- Flagged encounters: 7,359 (49.52%)
- TN: 6,992; FP: 6,190; FN: 511; TP: 1,169

Approximate 95% patient-cluster bootstrap intervals (300 draws, seed 42):

```text
           metric  estimate  lower_95  upper_95
average_precision   0.22913   0.20512   0.24906
          roc_auc   0.67050   0.65324   0.68509
           recall   0.69583   0.67159   0.71698
        precision   0.15885   0.14912   0.16774
      brier_score   0.09536   0.09121   0.09925
     flagged_rate   0.49516   0.48654   0.50498
```

The validation recall target of 70% was a development objective, not a guarantee on held-out data. These results do not trigger threshold changes. Bootstrap intervals describe sampling variation under patient resampling, not distribution shifts or model-selection uncertainty. AP is average precision, not trapezoidal PR-AUC.

## Calibration and demographic results

See test_calibration_bins.csv and test_evaluation.png. Reliability bins are approximately equal-count. Mean predicted probability: 11.39%; observed readmission rate: 11.30%. Brier score and log loss measure probabilistic accuracy and do not establish perfect calibration.

test_subgroup_metrics.csv reports age/gender/race counts, unique patients, prevalence, calibration gaps, Brier score, log loss, recall, precision, and false-positive rates at the single frozen threshold. Small groups (fewer than 100 encounters or 30 positive/negative outcomes) are flagged; undefined metrics remain blank. These descriptive subgroup estimates have no subgroup confidence intervals and do not prove fairness or clinical utility.

## Interpretation and limitations

This educational model uses historical 1999–2008 US diabetes encounter records and source readmission labels, which do not establish complete cross-hospital follow-up or distinguish all planned readmissions. Cohort exclusions and retained transfers are documented in reports/cohort_and_splits.md. Patient-disjoint splitting prevents patient overlap but does not establish temporal or hospital-level external validity. Flagging a large share of encounters is an important capacity limitation. Explanations identify model associations, not causal treatment effects. This is not clinically validated.

## Verification and next work

Verified frozen hashes, unique encounter assignments, patient-disjoint partitions, complete test membership, and finite probabilities. The evaluation marker and saved metrics prevent accidental fresh test evaluations on rerun. Original validation artifacts and the frozen policy are preserved; this test is now consumed and cannot serve as an untouched holdout for later tuning.

Next: build the Streamlit demonstration around the frozen raw-input pipeline and threshold, showing the actual model performance and educational limitations. New modeling experiments require a new evaluation design rather than optimizing against these test results.
