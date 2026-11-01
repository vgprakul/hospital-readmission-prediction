"""Create input choices/defaults and example inputs from training records only."""
from pathlib import Path
import csv
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    frozen = json.loads((ROOT / 'models/frozen_manifest.json').read_text())
    policy = frozen['policy']
    assignments = pd.read_csv(ROOT / 'data/processed/partition_assignments.csv')
    ids = set(assignments.loc[assignments.partition.eq('train'), 'encounter_id'])
    chunks = []
    for chunk in pd.read_csv(ROOT / 'data/raw/diabetic_data.csv', keep_default_na=False,
                             na_values=['?'], low_memory=False, chunksize=5000):
        chunks.append(chunk.loc[chunk.encounter_id.isin(ids), policy['features']])
    data = pd.concat(chunks, ignore_index=True)
    mappings, section = {}, None
    with (ROOT / 'data/raw/IDS_mapping.csv').open(newline='') as handle:
        for row in csv.reader(handle):
            if not row or not row[0]:
                continue
            if row[0].endswith('_id'):
                section = row[0]
                mappings[section] = {}
            else:
                mappings[section][row[0]] = row[1].strip()
    numeric = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
               'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
    fields = {}
    for column in policy['features']:
        if column in numeric:
            fields[column] = {'kind': 'integer', 'min': int(data[column].min()),
                              'max': int(data[column].max()), 'default': int(data[column].median())}
        else:
            options = sorted(data[column].dropna().unique().tolist(), key=str)
            options = [int(x) if column in mappings else str(x) for x in options]
            if data[column].isna().any():
                options.append('__MISSING__')
            fields[column] = {'kind': 'category', 'options': options,
                              'default': int(data[column].mode().iloc[0]) if column in mappings else str(data[column].mode().iloc[0])}
            if column in mappings:
                fields[column]['labels'] = {str(x): mappings[column].get(str(x), str(x)) for x in options}
    examples = []
    for _, row in data.sample(n=3, random_state=42).iterrows():
        examples.append({c: ('__MISSING__' if pd.isna(row[c]) else int(row[c])
                              if fields[c]['kind'] == 'integer' or c in mappings else str(row[c]))
                         for c in policy['features']})
    out = ROOT / 'app/assets'
    out.mkdir(exist_ok=True)
    document = {'origin': 'training partition only; examples contain no IDs or outcomes',
                'frozen_manifest_sha256': hashlib.sha256((ROOT / 'models/frozen_manifest.json').read_bytes()).hexdigest(),
                'fields': fields, 'examples': examples}
    (out / 'input_schema.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(f'Prepared {len(fields)} fields and {len(examples)} training examples.')


if __name__ == '__main__':
    main()
