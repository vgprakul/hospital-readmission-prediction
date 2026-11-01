"""Explain the validation predictions, then freeze the policy before test access."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'reports/explainability'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'models/frozen_manifest.json'
    if manifest_path.exists():
        raise RuntimeError('A frozen manifest already exists; preserve the existing final evaluation policy.')
    policy = json.loads((ROOT / 'models/operating_policy.json').read_text())
    meta = json.loads((ROOT / 'reports/model_comparison/run_metadata.json').read_text())
    model_path = ROOT / policy['model_path']
    assert sha(model_path) == policy['model_sha256']
    assert policy['selected_variant'] == 'uncalibrated', 'These raw SHAP explanations require the selected uncalibrated tree.'
    assert sha(ROOT / 'data/raw/diabetic_data.csv') == meta['source_sha256']
    assignments_path = ROOT / 'data/processed/partition_assignments.csv'
    assert sha(assignments_path) == meta['assignments_sha256']
    assignments = pd.read_csv(assignments_path, usecols=['encounter_id', 'patient_nbr', 'partition'])
    ids = assignments.loc[assignments.partition.eq('validation'), ['encounter_id', 'patient_nbr']]
    pieces = []
    for chunk in pd.read_csv(ROOT / 'data/raw/diabetic_data.csv', keep_default_na=False,
                             na_values=['?'], low_memory=False, chunksize=5000):
        pieces.append(chunk.loc[chunk.encounter_id.isin(ids.encounter_id)].copy())
    val = pd.concat(pieces, ignore_index=True)
    assert len(val) == len(ids) and val.encounter_id.is_unique
    assert set(zip(val.encounter_id, val.patient_nbr)) == set(zip(ids.encounter_id, ids.patient_nbr))
    model = joblib.load(model_path)
    with threadpool_limits(limits=2):
        probabilities = model.predict_proba(val[policy['features']])[:, 1]
    # Pick six examples by score quantiles, without choosing cases by outcome or attractiveness.
    order = np.argsort(probabilities, kind='stable')
    local_indices = order[np.round(np.linspace(0, len(order) - 1, 6)).astype(int)]
    global_indices = np.sort(np.random.default_rng(42).choice(len(val), min(1000, len(val)), replace=False))
    indices = np.unique(np.r_[global_indices, local_indices])
    sample = val.iloc[indices]
    engineered = model.named_steps['features'].transform(sample[policy['features']])
    preprocessing = model.named_steps['preprocessing']
    with threadpool_limits(limits=2):
        encoded = preprocessing.transform(engineered)
        print('Computing exact tree SHAP on validation examples.', flush=True)
        explainer = shap.TreeExplainer(model.named_steps['model'], model_output='raw',
                                        feature_perturbation='tree_path_dependent')
        values = np.asarray(explainer.shap_values(encoded, check_additivity=True))
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    # Map one-hot columns back to their original engineered feature for readable additive explanations.
    names = []
    for name, transformer, columns in preprocessing.transformers_:
        if name == 'remainder':
            continue
        if name == 'numeric':
            names.extend(columns)
        else:
            encoder = transformer.named_steps['encoder'] if name == 'categorical' else transformer
            for column, categories in zip(columns, encoder.categories_):
                names.extend([column] * len(categories))
    assert values.shape == (len(sample), len(names))
    grouped = pd.DataFrame(values.T).groupby(names, sort=False).sum().T
    reconstructed = expected + grouped.sum(axis=1).to_numpy()
    with threadpool_limits(limits=2):
        np.testing.assert_allclose(reconstructed, model.named_steps['model'].decision_function(encoded), atol=1e-7)
    np.testing.assert_allclose(1 / (1 + np.exp(-reconstructed)), probabilities[indices], atol=1e-7)
    positions = {index: pos for pos, index in enumerate(indices)}
    global_positions = [positions[index] for index in global_indices]
    importance = pd.DataFrame({'feature': grouped.columns,
                               'mean_absolute_shap_log_odds': grouped.iloc[global_positions].abs().mean().values})
    importance = importance.sort_values('mean_absolute_shap_log_odds', ascending=False)
    importance.to_csv(out / 'global_shap_importance.csv', index=False)
    local_rows, local_summaries = [], []
    for case, index in enumerate(local_indices, 1):
        pos = positions[index]
        contributions = grouped.iloc[pos]
        local_summaries.append({'case': case, 'encounter_id': int(val.iloc[index].encounter_id),
                                'probability': float(probabilities[index]),
                                'flagged': bool(probabilities[index] >= policy['threshold']),
                                'base_log_odds': expected, 'sum_shap_log_odds': float(contributions.sum())})
        for feature, contribution in contributions.items():
            value = engineered.iloc[pos][feature]
            local_rows.append({'case': case, 'encounter_id': int(val.iloc[index].encounter_id),
                               'feature': feature, 'feature_value': str(value), 'shap_log_odds': float(contribution)})
    pd.DataFrame(local_rows).to_csv(out / 'individual_shap_contributions.csv', index=False)
    pd.DataFrame(local_summaries).to_csv(out / 'individual_predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 6))
    top = importance.head(15).iloc[::-1]
    axis.barh(top.feature, top.mean_absolute_shap_log_odds, color='#2876a3')
    axis.set(xlabel='Mean absolute grouped SHAP contribution (log-odds)',
             title='Global model explanations — 1,000 validation encounters')
    fig.tight_layout()
    fig.savefig(out / 'global_shap_importance.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for axis, case in zip(axes.flat, range(1, 7)):
        frame = pd.DataFrame(local_rows).query('case == @case').copy()
        frame = frame.loc[frame.shap_log_odds.abs().nlargest(8).index].sort_values('shap_log_odds')
        info = local_summaries[case - 1]
        axis.barh(frame.feature, frame.shap_log_odds,
                  color=['#c84c45' if v > 0 else '#2876a3' for v in frame.shap_log_odds])
        axis.axvline(0, color='gray', linewidth=0.6)
        axis.set(title=f"Case {case}: score {info['probability']:.1%}", xlabel='SHAP contribution (log-odds)')
    fig.tight_layout()
    fig.savefig(out / 'individual_explanations.png', dpi=150)
    plt.close(fig)
    report = f'''# Validation model explanations

Exact tree SHAP explains the selected uncalibrated gradient boosting model. Global importance is computed from a fixed-seed random sample of 1,000 validation encounters. Six individual examples are selected at evenly spaced score ranks, without using their outcomes. No test data was used.

One-hot contributions are summed into their source engineered fields before computing mean absolute importance. Contributions are additive in **log-odds**, not percentage points. The model's training tree-path counts define the background; these are model explanations, not causal effects or clinical recommendations. Correlated raw and derived variables can share attribution. A grouped field's value is an attribution aggregation, not a separate independently trained explanation model.

Top features:

```text
{importance.head(15).round(5).to_string(index=False)}
```

Individual predictions:

```text
{pd.DataFrame(local_summaries).round(5).to_string(index=False)}
```

For each case, base_log_odds + all contributions = model log-odds; applying the logistic function reproduces the full pipeline probability. Both equalities were verified numerically. Individual plots show the eight largest contributions only; full contributions are in the CSV.

## Operating-policy review before test access

Retain the existing illustrative objective of 70% validation recall. It flags 48.97% of validation encounters at 15.93% precision. Alternatives at 60% and 50% recall flag 38.69% and 29.00% respectively. No real intervention-capacity constraint has been provided, so the 70% policy remains an educational demonstration rather than an operational recommendation. No threshold or model changes were made based on explanations.

Freeze model, source code, calibration choice, feature list, threshold, source data, and partition assignments in models/frozen_manifest.json before evaluating the test partition. The test script verifies those hashes and uses this frozen policy without tuning.
'''
    (out / 'explainability_report.md').write_text(report, encoding='utf-8')
    freeze_paths = [model_path, ROOT / 'src/features.py', ROOT / 'src/__init__.py',
                    ROOT / 'requirements.txt', ROOT / 'models/operating_policy.json',
                    ROOT / 'data/raw/diabetic_data.csv', assignments_path,
                    ROOT / 'scripts/final_test_evaluation.py']
    manifest = {'status': 'frozen before final test evaluation', 'project_date': '2026-11-01',
                'policy': policy, 'test_evaluated_at_freeze': False,
                'files_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in freeze_paths},
                'explainability': {'method': 'Tree SHAP, training tree-path background, raw log-odds',
                                   'global_validation_sample': 1000, 'local_cases': 6, 'shap_version': shap.__version__}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('SHAP reconstruction checks passed. Frozen manifest saved before test evaluation.', flush=True)


if __name__ == '__main__':
    main()
