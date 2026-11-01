# Validation model explanations

Exact tree SHAP explains the selected uncalibrated gradient boosting model. Global importance is computed from a fixed-seed random sample of 1,000 validation encounters. Six individual examples are selected at evenly spaced score ranks, without using their outcomes. No test data was used.

One-hot contributions are summed into their source engineered fields before computing mean absolute importance. Contributions are additive in **log-odds**, not percentage points. The model's training tree-path counts define the background; these are model explanations, not causal effects or clinical recommendations. Correlated raw and derived variables can share attribution. A grouped field's value is an attribution aggregation, not a separate independently trained explanation model.

Top features:

```text
                    feature  mean_absolute_shap_log_odds
           number_inpatient                      0.22756
   discharge_disposition_id                      0.20638
      total_previous_visits                      0.08662
           time_in_hospital                      0.05877
           number_diagnoses                      0.05576
               diag_1_group                      0.05474
                 payer_code                      0.04466
                        age                      0.03448
          medical_specialty                      0.03352
                    insulin                      0.03137
active_diabetes_medications                      0.02887
            num_medications                      0.02874
                  metformin                      0.02769
           number_emergency                      0.02535
         num_lab_procedures                      0.02479
```

Individual predictions:

```text
 case  encounter_id  probability  flagged  base_log_odds  sum_shap_log_odds
    1     175164420      0.02342    False       -2.18583           -1.54476
    2     168270876      0.06278    False       -2.18583           -0.51743
    3     240132540      0.08387    False       -2.18583           -0.20505
    4      38495142      0.10923     True       -2.18583            0.08724
    5     149663454      0.15157     True       -2.18583            0.46350
    6     243967338      0.71777     True       -2.18583            3.11925
```

For each case, base_log_odds + all contributions = model log-odds; applying the logistic function reproduces the full pipeline probability. Both equalities were verified numerically. Individual plots show the eight largest contributions only; full contributions are in the CSV.

## Operating-policy review before test access

Retain the existing illustrative objective of 70% validation recall. It flags 48.97% of validation encounters at 15.93% precision. Alternatives at 60% and 50% recall flag 38.69% and 29.00% respectively. No real intervention-capacity constraint has been provided, so the 70% policy remains an educational demonstration rather than an operational recommendation. No threshold or model changes were made based on explanations.

Freeze model, source code, calibration choice, feature list, threshold, source data, and partition assignments in models/frozen_manifest.json before evaluating the test partition. The test script verifies those hashes and uses this frozen policy without tuning.
