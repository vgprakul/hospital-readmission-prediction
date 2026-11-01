"""One final evaluation of a frozen policy; no training or threshold changes."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from scripts.calibrate_and_audit import score, reliability


def main():
    manifest_path = ROOT / 'models/frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    out = ROOT / 'reports/final_test'
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'test_metrics.json'
    if result_path.exists():
        print('Final evaluation already saved. Reusing results; no new test scoring.', flush=True)
        print(result_path.read_text(), flush=True)
        return
    # Exclusive marker prevents accidentally treating repeated evaluations as fresh holdouts.
    marker = out / 'evaluation_started.json'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump({'status': 'started under frozen policy',
                   'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}, handle, indent=2)
    policy = manifest['policy']
    assignments = pd.read_csv(ROOT / 'data/processed/partition_assignments.csv',
                              usecols=['encounter_id', 'patient_nbr', 'partition'])
    assert assignments.encounter_id.is_unique
    assert assignments.groupby('patient_nbr').partition.nunique().max() == 1
    ids = assignments.loc[assignments.partition.eq('test'), ['encounter_id', 'patient_nbr']]
    pieces = []
    for chunk in pd.read_csv(ROOT / 'data/raw/diabetic_data.csv', keep_default_na=False,
                             na_values=['?'], low_memory=False, chunksize=5000):
        pieces.append(chunk.loc[chunk.encounter_id.isin(ids.encounter_id)].copy())
    test = pd.concat(pieces, ignore_index=True)
    assert len(test) == len(ids) and test.encounter_id.is_unique
    assert set(zip(test.encounter_id, test.patient_nbr)) == set(zip(ids.encounter_id, ids.patient_nbr))
    model = joblib.load(ROOT / policy['model_path'])
    print('Scoring reserved test encounters with the frozen model and threshold.', flush=True)
    with threadpool_limits(limits=2):
        p = model.predict_proba(test[policy['features']])[:, 1]
    y = test.readmitted.eq('<30').astype(int).to_numpy()
    threshold = policy['threshold']
    result = score(y, p, threshold)
    result['patients'] = int(test.patient_nbr.nunique())
    result['f1'] = 2 * result['tp'] / (2 * result['tp'] + result['fp'] + result['fn'])
    result['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result['status'] = 'final held-out evaluation; no model or threshold changes'
    # Save predictions once and use them for diagnostics; no subsequent model scoring is needed.
    predictions = test[['encounter_id', 'patient_nbr', 'age', 'gender', 'race']].copy()
    predictions['target'] = y
    predictions['probability'] = p
    predictions['flagged'] = p >= threshold
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    bins = reliability(y, p)
    bins.to_csv(out / 'test_calibration_bins.csv', index=False)
    rows = []
    for attribute in ['age', 'gender', 'race']:
        labels = test[attribute].fillna('Missing').to_numpy()
        for label in sorted(set(labels)):
            mask = labels == label
            row = {'attribute': attribute, 'group': label,
                   'patients': int(test.loc[mask, 'patient_nbr'].nunique()), **score(y[mask], p[mask], threshold)}
            row['small_group'] = bool(row['encounters'] < 100 or row['positive_count'] < 30 or
                                      row['encounters'] - row['positive_count'] < 30)
            rows.append(row)
    subgroup = pd.DataFrame(rows)
    subgroup.to_csv(out / 'test_subgroup_metrics.csv', index=False)
    # Cluster bootstrap respects repeated encounters: sample patients, carrying all their encounters.
    codes, patients = pd.factorize(test.patient_nbr)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(300):
        weights_by_patient = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))
        weights = weights_by_patient[codes]
        tp = weights[(y == 1) & (p >= threshold)].sum()
        fp = weights[(y == 0) & (p >= threshold)].sum()
        positives = weights[y == 1].sum()
        draws.append({'average_precision': average_precision_score(y, p, sample_weight=weights),
                      'roc_auc': roc_auc_score(y, p, sample_weight=weights),
                      'recall': tp / positives, 'precision': tp / (tp + fp),
                      'brier_score': np.average((y - p) ** 2, weights=weights),
                      'flagged_rate': (tp + fp) / weights.sum()})
    draws = pd.DataFrame(draws)
    intervals = pd.DataFrame({'metric': draws.columns,
                              'estimate': [result[column] for column in draws],
                              'lower_95': draws.quantile(0.025).values,
                              'upper_95': draws.quantile(0.975).values})
    intervals.to_csv(out / 'patient_bootstrap_intervals.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f"AUC {result['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(title='Final test ROC', xlabel='False-positive rate', ylabel='Recall')
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, label=f"AP {result['average_precision']:.3f}")
    axes[1].axhline(y.mean(), linestyle='--', color='gray')
    axes[1].scatter([result['recall']], [result['precision']], color='#c84c45', label='Frozen threshold')
    axes[1].set(title='Final test precision–recall', xlabel='Recall', ylabel='Precision')
    axes[2].plot(bins.mean_probability, bins.observed_rate, marker='o', label='Test bins')
    axes[2].plot([0, 0.4], [0, 0.4], '--', color='gray')
    axes[2].set(title='Final test calibration (detail)', xlabel='Mean probability',
                ylabel='Observed fraction', xlim=(0, 0.4), ylim=(0, 0.4))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / 'test_evaluation.png', dpi=160)
    plt.close(fig)
    report = f'''# Final held-out test evaluation

The model, calibration choice, features, source code, data/partition checksums, and exact threshold were frozen in models/frozen_manifest.json before this test evaluation. No model fitting, calibration fitting, threshold adjustment, or model selection was performed on test outcomes. All outputs derive from one model-scoring pass.

## Results

- Encounters: {len(test):,}; unique patients: {result['patients']:,}; prevalence: {y.mean():.2%}
- Average precision: {result['average_precision']:.4f}; ROC-AUC: {result['roc_auc']:.4f}
- Brier score: {result['brier_score']:.5f}; log loss: {result['log_loss']:.5f}
- Frozen threshold: {threshold:.12f}
- Recall: {result['recall']:.2%}; precision: {result['precision']:.2%}; F1: {result['f1']:.4f}
- Flagged encounters: {result['flagged_count']:,} ({result['flagged_rate']:.2%})
- TN: {result['tn']:,}; FP: {result['fp']:,}; FN: {result['fn']:,}; TP: {result['tp']:,}

Approximate 95% patient-cluster bootstrap intervals (300 draws, seed 42):

```text
{intervals.round(5).to_string(index=False)}
```

The validation recall target of 70% was a development objective, not a guarantee on held-out data. These results do not trigger threshold changes. Bootstrap intervals describe sampling variation under patient resampling, not distribution shifts or model-selection uncertainty. AP is average precision, not trapezoidal PR-AUC.

## Calibration and demographic results

See test_calibration_bins.csv and test_evaluation.png. Reliability bins are approximately equal-count. Mean predicted probability: {p.mean():.2%}; observed readmission rate: {y.mean():.2%}. Brier score and log loss measure probabilistic accuracy and do not establish perfect calibration.

test_subgroup_metrics.csv reports age/gender/race counts, unique patients, prevalence, calibration gaps, Brier score, log loss, recall, precision, and false-positive rates at the single frozen threshold. Small groups (fewer than 100 encounters or 30 positive/negative outcomes) are flagged; undefined metrics remain blank. These descriptive subgroup estimates have no subgroup confidence intervals and do not prove fairness or clinical utility.

## Interpretation and limitations

This educational model uses historical 1999–2008 US diabetes encounter records and source readmission labels, which do not establish complete cross-hospital follow-up or distinguish all planned readmissions. Cohort exclusions and retained transfers are documented in reports/cohort_and_splits.md. Patient-disjoint splitting prevents patient overlap but does not establish temporal or hospital-level external validity. Flagging a large share of encounters is an important capacity limitation. Explanations identify model associations, not causal treatment effects. This is not clinically validated.

## Verification and next work

Verified frozen hashes, unique encounter assignments, patient-disjoint partitions, complete test membership, and finite probabilities. The evaluation marker and saved metrics prevent accidental fresh test evaluations on rerun. Original validation artifacts and the frozen policy are preserved; this test is now consumed and cannot serve as an untouched holdout for later tuning.

Next: build the Streamlit demonstration around the frozen raw-input pipeline and threshold, showing the actual model performance and educational limitations. New modeling experiments require a new evaluation design rather than optimizing against these test results.
'''
    (out / 'final_test_report.md').write_text(report, encoding='utf-8')
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    marker.write_text(json.dumps({'status': 'completed', 'manifest_sha256': result['manifest_sha256'],
                                  'test_scoring_passes': 1}, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
