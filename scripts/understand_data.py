"""Extract the official dataset and generate an executed inspection notebook."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
raw = ROOT / 'data' / 'raw'
with zipfile.ZipFile(raw / 'uci_diabetes_130.zip') as archive:
    for name in ('diabetic_data.csv', 'IDS_mapping.csv'):
        matches = [entry for entry in archive.namelist() if Path(entry).name == name]
        if len(matches) != 1:
            raise ValueError(f'Expected one archive entry for {name}: {matches}')
        (raw / name).write_bytes(archive.read(matches[0]))

cells = []
def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})

scope = {'__name__': '__main__'}
def code(source):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(source, '<notebook>', 'exec'), scope)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(True),
                  'execution_count': sum(c['cell_type'] == 'code' for c in cells) + 1,
                  'outputs': [{'output_type': 'stream', 'name': 'stdout',
                               'text': stream.getvalue().splitlines(True)}]})

markdown('''# 01 — Dataset acquisition and understanding

## Prediction task
Using information available at discharge, predict whether a hospital encounter involving a patient with diabetes is followed by a recorded readmission within 30 days.

- Unit: one hospital encounter.
- Target: `<30` = 1; `>30` and `NO` = 0. This follows the source labels; `NO` does not establish complete follow-up across all hospitals.
- Prediction time: discharge.
- Retain `patient_nbr` for grouped splitting; exclude both identifiers and the original `readmitted` column from model inputs.
- Keep every encounter for this initial audit. Decide the eligible discharge cohort before modeling, including how to handle deaths and hospice discharges.
- Later, reserve patient-disjoint train/validation/test partitions before fitting imputers, encoders, feature selection, or resampling. Training-only exploration should guide modeling decisions.

## Source
[UCI Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008).
Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). DOI: https://doi.org/10.24432/C5230J. License: CC BY 4.0.
Historical US diabetes encounter data; this project is educational and is not clinically validated.
''')
code('''from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / 'data' / 'raw' / 'diabetic_data.csv').exists():
    ROOT = ROOT.parent
df = pd.read_csv(ROOT / 'data' / 'raw' / 'diabetic_data.csv',
                 keep_default_na=False, na_values=['?'], low_memory=False)
assert df['readmitted'].isin(['<30', '>30', 'NO']).all()
assert df['patient_nbr'].notna().all()
print('Raw shape:', df.shape)
print('Unique patients:', df['patient_nbr'].nunique())
print('Duplicate encounter IDs:', df['encounter_id'].duplicated().sum())
print(df.head().to_string(index=False))
''')
code('''print('Column types and completeness:')
df.info()
schema = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'unique_values': df.nunique(dropna=False),
    'missing_count': df.isna().sum(),
    'missing_percent': df.isna().mean().mul(100),
})
print(schema.to_string())
print('\\nNumeric summaries (identifier summaries are not predictive features):')
print(df.describe().to_string())
''')
markdown('''## Missing-value interpretation
Only `?` is converted to missing on load. In `A1Cresult` and `max_glu_serum`, the literal category `None` means no test result recorded and is preserved. Admission/discharge mapping codes also include unknown or unavailable categories; inspect these before preprocessing. No imputation or column removal is performed here.
''')
code('''print('Missing values (%):')
print(schema['missing_percent'].sort_values(ascending=False).to_string())
for column in ['A1Cresult', 'max_glu_serum', 'gender']:
    print('\\n' + column)
    print(df[column].value_counts(dropna=False).to_string())
print('\\nAdmission/discharge mapping file:')
print((ROOT / 'data' / 'raw' / 'IDS_mapping.csv').read_text())
''')
code('''target = df['readmitted'].eq('<30').astype('int8').rename('target')
label_counts = df['readmitted'].value_counts()
print('Original readmission labels:')
print(label_counts.to_string())
print('\\nBinary target:')
print(pd.DataFrame({'count': target.value_counts().sort_index(),
                    'percent': target.value_counts(normalize=True).sort_index() * 100}).to_string())
patient_counts = df.groupby('patient_nbr').size()
print('\\nPatients with multiple encounters:', int(patient_counts.gt(1).sum()))
print('Additional encounters beyond one per patient:', len(df) - len(patient_counts))
print('Maximum encounters for one patient:', int(patient_counts.max()))
print('\\nDischarge disposition counts (review eligibility before modeling):')
print(df['discharge_disposition_id'].value_counts().sort_index().to_string())
''')
code('''reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
schema.to_csv(reports / 'column_inventory.csv', index_label='column')
summary = {
    'encounters': len(df), 'raw_columns': len(df.columns),
    'unique_patients': int(df['patient_nbr'].nunique()),
    'patients_with_multiple_encounters': int(patient_counts.gt(1).sum()),
    'duplicate_encounter_ids': int(df['encounter_id'].duplicated().sum()),
    'readmission_labels': {str(k): int(v) for k, v in label_counts.items()},
    'positive_target_count': int(target.sum()),
    'positive_target_percent': float(target.mean() * 100),
}
(reports / 'data_summary.json').write_text(__import__('json').dumps(summary, indent=2))
print(__import__('json').dumps(summary, indent=2))
''')
markdown('''## First milestone completed
The raw data is loaded, columns and missingness are inventoried, the target is defined, and repeated patients are measured. Raw files remain unchanged. No model has been trained and no preprocessing has been fitted.

Next: review discharge eligibility and coded missing categories, then create reproducible patient-disjoint partitions. The raw CSV contains 50 columns (including identifiers and target), while UCI lists 47 features.
''')
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
             'language_info': {'name': 'python', 'version': __import__('platform').python_version()}},
            'nbformat': 4, 'nbformat_minor': 4}
(ROOT / 'notebooks' / '01_data_understanding.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
summary = scope['summary']
report = f'''# Initial data audit

- Encounters: {summary['encounters']:,}
- Raw columns: {summary['raw_columns']}
- Unique patients: {summary['unique_patients']:,}
- Patients with multiple encounters: {summary['patients_with_multiple_encounters']:,}
- Duplicate encounter IDs: {summary['duplicate_encounter_ids']}
- Readmitted within 30 days: {summary['positive_target_count']:,} ({summary['positive_target_percent']:.2f}%)

## Largest missing-value percentages

{scope['schema']['missing_percent'].sort_values(ascending=False).head(7).round(2).to_string()}

`None` lab categories are preserved; only `?` is parsed as missing. No cleaning, imputation, or training has been performed.

Next: decide discharge eligibility (including death/hospice records), interpret mapping codes, and reserve patient-disjoint splits. Exclude IDs and the source target from model features. Fit all learned preprocessing on training data only.
'''
(ROOT / 'reports' / 'initial_data_audit.md').write_text(report, encoding='utf-8')
provenance = {'source': 'https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008',
              'download_url': 'https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip',
              'license': 'CC BY 4.0', 'download_date': '2026-10-31',
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir() if p.suffix in ('.csv', '.zip')}}
(raw / 'source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
print(report)
