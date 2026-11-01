"""Input validation, frozen model loading, inference, and additive SHAP explanations."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load_artifacts():
    manifest_path = ROOT / 'models/frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Raw data is not needed to serve the app. Verify the model and its Python feature code.
    for relative, expected in manifest['files_sha256'].items():
        if relative.startswith(('models/', 'src/')):
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen artifact checksum mismatch: {relative}')
    schema = json.loads((ROOT / 'app/assets/input_schema.json').read_text())
    if schema['frozen_manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError('Input schema belongs to a different frozen manifest')
    # Frozen metadata records a Windows path; normalize separators for Linux hosting.
    model = joblib.load(ROOT / manifest['policy']['model_path'].replace('\\', '/'))
    return model, manifest['policy'], schema


def make_frame(values, policy, schema):
    if set(values) != set(policy['features']):
        raise ValueError('Provide exactly the frozen model input fields')
    cleaned = {}
    for column, spec in schema['fields'].items():
        value = values[column]
        if spec['kind'] == 'integer':
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f'{column} must be an integer')
            if not np.isfinite(value) or int(value) != value or not spec['min'] <= value <= spec['max']:
                raise ValueError(f'{column} must be between {spec["min"]} and {spec["max"]}')
            cleaned[column] = int(value)
        else:
            if value not in spec['options']:
                raise ValueError(f'Unsupported value for {column}')
            cleaned[column] = np.nan if value == '__MISSING__' else value
    return pd.DataFrame([cleaned], columns=policy['features'])


def predict(model, policy, schema, values):
    frame = make_frame(values, policy, schema)
    with threadpool_limits(limits=2):
        probability = float(model.predict_proba(frame)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Invalid model probability')
    return {'probability': probability, 'flagged': probability >= policy['threshold'],
            'threshold': policy['threshold']}


def explain(model, policy, schema, values):
    import shap
    frame = make_frame(values, policy, schema)
    engineered = model.named_steps['features'].transform(frame)
    preprocessing = model.named_steps['preprocessing']
    with threadpool_limits(limits=2):
        encoded = preprocessing.transform(engineered)
        explainer = shap.TreeExplainer(model.named_steps['model'], model_output='raw',
                                        feature_perturbation='tree_path_dependent')
        contributions = np.asarray(explainer.shap_values(encoded, check_additivity=True))[0]
    labels = []
    for name, transformer, columns in preprocessing.transformers_:
        if name == 'remainder':
            continue
        if name == 'numeric':
            labels.extend(columns)
        else:
            encoder = transformer.named_steps['encoder'] if name == 'categorical' else transformer
            for column, categories in zip(columns, encoder.categories_):
                labels.extend([column] * len(categories))
    grouped = pd.Series(contributions).groupby(labels, sort=False).sum()
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed = float(1 / (1 + np.exp(-(base + grouped.sum()))))
    result = predict(model, policy, schema, values)
    np.testing.assert_allclose(reconstructed, result['probability'], atol=1e-7)
    table = pd.DataFrame({'feature': grouped.index,
                          'value': [str(engineered.iloc[0][c]) for c in grouped.index],
                          'shap_log_odds': grouped.values})
    return table, base, result
