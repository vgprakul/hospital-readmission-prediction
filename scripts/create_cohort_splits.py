"""Define the discharge cohort and reserve deterministic patient-disjoint splits."""
from pathlib import Path
import csv
import json
import hashlib

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EXCLUSIONS = {
    9: 'not a completed discharge: admitted as inpatient',
    12: 'not a clear completed discharge: still patient / expected outpatient return',
    11: 'death', 19: 'death', 20: 'death', 21: 'death',
    13: 'hospice', 14: 'hospice',
}
UNKNOWN_CODES = {
    'admission_type_id': [5, 6, 8],
    'discharge_disposition_id': [18, 25, 26],
    'admission_source_id': [9, 15, 17, 20, 21],
}


def main():
    source = ROOT / 'data/raw/diabetic_data.csv'
    df = pd.read_csv(source, keep_default_na=False, na_values=['?'], low_memory=False)
    assert df.encounter_id.is_unique
    assert df.patient_nbr.notna().all()
    assert df.readmitted.isin(['<30', '>30', 'NO']).all()
    # Parse the three separate tables in the mapping CSV without treating NULL as missing.
    mappings = {}
    section = None
    with (ROOT / 'data/raw/IDS_mapping.csv').open(newline='') as handle:
        for row in csv.reader(handle):
            if not row or not row[0]:
                continue
            if row[0].endswith('_id'):
                section = row[0]
                mappings[section] = {}
            else:
                mappings[section][int(row[0])] = row[1].strip()
    for column, mapping in mappings.items():
        assert set(df[column].unique()).issubset(mapping), f'Unmapped codes in {column}'

    reason = df.discharge_disposition_id.map(EXCLUSIONS)
    cohort = df.loc[reason.isna()].copy()
    cohort['target'] = cohort.readmitted.eq('<30').astype('int8')
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, held_idx = next(first.split(cohort, groups=cohort.patient_nbr))
    held = cohort.iloc[held_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(second.split(held, groups=held.patient_nbr))
    parts = {'train': cohort.iloc[train_idx], 'validation': held.iloc[val_idx],
             'test': held.iloc[test_idx]}
    patient_sets = {name: set(part.patient_nbr) for name, part in parts.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert patient_sets[a].isdisjoint(patient_sets[b]), f'Patient overlap: {a}/{b}'
    assignments = pd.concat([
        part[['encounter_id', 'patient_nbr', 'target']].assign(partition=name)
        for name, part in parts.items()
    ]).sort_values('encounter_id')
    assert assignments.encounter_id.is_unique
    assert set(assignments.encounter_id) == set(cohort.encounter_id)
    assert len(assignments) == len(cohort)
    assert all(part.target.nunique() == 2 for part in parts.values())

    out = ROOT / 'data/processed'
    reports = ROOT / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    assignments.to_csv(out / 'partition_assignments.csv', index=False)
    df.loc[reason.notna(), ['encounter_id', 'patient_nbr', 'discharge_disposition_id']].assign(
        exclusion_reason=reason[reason.notna()]).to_csv(out / 'excluded_encounters.csv', index=False)
    exclusions = []
    for code, why in EXCLUSIONS.items():
        exclusions.append({'code': code, 'description': mappings['discharge_disposition_id'][code],
                           'reason': why, 'encounters': int(df.discharge_disposition_id.eq(code).sum())})
    pd.DataFrame(exclusions).to_csv(reports / 'cohort_exclusions.csv', index=False)
    unknowns = []
    for column, codes in UNKNOWN_CODES.items():
        for code in codes:
            unknowns.append({'column': column, 'code': code, 'description': mappings[column][code],
                             'eligible_encounters': int(cohort[column].eq(code).sum())})
    pd.DataFrame(unknowns).to_csv(reports / 'coded_unknown_values.csv', index=False)
    stats = pd.DataFrame([
        {'partition': name, 'encounters': len(part),
         'encounter_percent': 100 * len(part) / len(cohort),
         'patients': part.patient_nbr.nunique(),
         'patient_percent': 100 * part.patient_nbr.nunique() / cohort.patient_nbr.nunique(),
         'positive_count': int(part.target.sum()), 'readmission_percent': 100 * part.target.mean()}
        for name, part in parts.items()
    ])
    stats.to_csv(reports / 'partition_summary.csv', index=False)
    metadata = {'seed': SEED, 'method': 'two-stage GroupShuffleSplit; 70/15/15 of patients approximately',
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'raw_encounters': len(df), 'excluded_encounters': int(reason.notna().sum()),
                'eligible_encounters': len(cohort), 'eligible_patients': int(cohort.patient_nbr.nunique()),
                'excluded_discharge_codes': EXCLUSIONS, 'coded_unknown_values': UNKNOWN_CODES,
                'patient_overlap': 0, 'partition_summary': stats.to_dict(orient='records')}
    (reports / 'cohort_split_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = f'''# Eligible cohort and reserved partitions

## Policy

Prediction is made at discharge for a recorded readmission within 30 days. Exclude death codes 11/19/20/21, hospice codes 13/14, and codes 9/12 that do not clearly represent a completed discharge. Code 12 is ambiguous (still patient or expected outpatient return); excluding all of it is a conservative project policy. Hospice exclusion defines a cohort outside end-of-life care; it does not imply hospice patients cannot be readmitted.

Retain other transfers, rehabilitation, nursing-facility discharges, unknown discharge codes, and unknown/invalid gender for now. Transfer records may represent continued care, so this cohort must not be described as exclusively community discharges or unplanned readmissions. The source does not establish complete cross-hospital follow-up. Future alternative cohort policies should be evaluated on training data and documented before any test evaluation.

Raw encounters: {len(df):,}. Excluded: {reason.notna().sum():,}. Eligible: {len(cohort):,}, representing {cohort.patient_nbr.nunique():,} patients.

See `cohort_exclusions.csv` for code-level exclusion counts and descriptions. No missing-value imputation or feature fitting was performed.

## Split results

```text
{stats.round(4).to_string(index=False)}
```

Two-stage GroupShuffleSplit, random_state=42. Target ratios apply approximately to unique patients; encounter ratios and label rates can differ. Assignments depend on patient groups, not on optimizing label balance. No seed search was performed.

Verified: zero shared patients across every pair of partitions; each eligible encounter assigned exactly once; excluded records absent; both target classes present in all partitions.

## Coded unknown values

Admission type: 5/6/8. Discharge disposition: 18/25/26. Admission source: 9/15/17/20/21. Preserve raw values and treat these as explicit unknown categories during the next preprocessing step. Integer codes are categorical, not ordered quantities. `None` lab categories remain intact and `?` denotes missing.

## Use these partitions

Join `data/processed/partition_assignments.csv` to the raw file on `encounter_id`, validating a one-to-one join. Keep `patient_nbr` only as grouping metadata. Exclude `encounter_id`, `patient_nbr`, `readmitted`, and `partition` from predictors; `target` is the outcome. Only the training partition should guide cleaning decisions, feature engineering, and cross-validation. Use patient-grouped cross-validation inside training; fit imputers/encoders/resampling inside each fold. Reserve validation for model/threshold selection and test for final evaluation.

Regenerate using `python scripts/create_cohort_splits.py`. This overwrites assignments deterministically; do not change cohort policy or seed after examining test performance.
'''
    (reports / 'cohort_and_splits.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
