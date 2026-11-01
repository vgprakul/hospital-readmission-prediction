# Initial data audit

- Encounters: 101,766
- Raw columns: 50
- Unique patients: 71,518
- Patients with multiple encounters: 16,773
- Duplicate encounter IDs: 0
- Readmitted within 30 days: 11,357 (11.16%)

## Largest missing-value percentages

weight               96.86
medical_specialty    49.08
payer_code           39.56
race                  2.23
diag_3                1.40
diag_2                0.35
diag_1                0.02

`None` lab categories are preserved; only `?` is parsed as missing. No cleaning, imputation, or training has been performed.

Next: decide discharge eligibility (including death/hospice records), interpret mapping codes, and reserve patient-disjoint splits. Exclude IDs and the source target from model features. Fit all learned preprocessing on training data only.
