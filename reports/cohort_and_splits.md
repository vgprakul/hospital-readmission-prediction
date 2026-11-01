# Eligible cohort and reserved partitions

## Policy

Prediction is made at discharge for a recorded readmission within 30 days. Exclude death codes 11/19/20/21, hospice codes 13/14, and codes 9/12 that do not clearly represent a completed discharge. Code 12 is ambiguous (still patient or expected outpatient return); excluding all of it is a conservative project policy. Hospice exclusion defines a cohort outside end-of-life care; it does not imply hospice patients cannot be readmitted.

Retain other transfers, rehabilitation, nursing-facility discharges, unknown discharge codes, and unknown/invalid gender for now. Transfer records may represent continued care, so this cohort must not be described as exclusively community discharges or unplanned readmissions. The source does not establish complete cross-hospital follow-up. Future alternative cohort policies should be evaluated on training data and documented before any test evaluation.

Raw encounters: 101,766. Excluded: 2,447. Eligible: 99,319, representing 69,980 patients.

See `cohort_exclusions.csv` for code-level exclusion counts and descriptions. No missing-value imputation or feature fitting was performed.

## Split results

```text
 partition  encounters  encounter_percent  patients  patient_percent  positive_count  readmission_percent
     train       69651            70.1286     48986             70.0            7973              11.4471
validation       14806            14.9075     10497             15.0            1650              11.1441
      test       14862            14.9639     10497             15.0            1680              11.3040
```

Two-stage GroupShuffleSplit, random_state=42. Target ratios apply approximately to unique patients; encounter ratios and label rates can differ. Assignments depend on patient groups, not on optimizing label balance. No seed search was performed.

Verified: zero shared patients across every pair of partitions; each eligible encounter assigned exactly once; excluded records absent; both target classes present in all partitions.

## Coded unknown values

Admission type: 5/6/8. Discharge disposition: 18/25/26. Admission source: 9/15/17/20/21. Preserve raw values and treat these as explicit unknown categories during the next preprocessing step. Integer codes are categorical, not ordered quantities. `None` lab categories remain intact and `?` denotes missing.

## Use these partitions

Join `data/processed/partition_assignments.csv` to the raw file on `encounter_id`, validating a one-to-one join. Keep `patient_nbr` only as grouping metadata. Exclude `encounter_id`, `patient_nbr`, `readmitted`, and `partition` from predictors; `target` is the outcome. Only the training partition should guide cleaning decisions, feature engineering, and cross-validation. Use patient-grouped cross-validation inside training; fit imputers/encoders/resampling inside each fold. Reserve validation for model/threshold selection and test for final evaluation.

Regenerate using `python scripts/create_cohort_splits.py`. This overwrites assignments deterministically; do not change cohort policy or seed after examining test performance.
