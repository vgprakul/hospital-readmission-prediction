"""Training-only grouped calibration, validation threshold and subgroup audit."""
from pathlib import Path
import gc
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

RECALL_TARGET = 0.70  # Illustrative project objective; not a clinical requirement.


def score(y, p, threshold):
    y = np.asarray(y)
    p = np.asarray(p)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'encounters': len(y), 'positive_count': int(y.sum()),
            'readmission_rate': float(y.mean()), 'mean_probability': float(p.mean()),
            'calibration_gap': float(p.mean() - y.mean()),
            'average_precision': float(average_precision_score(y, p)) if y.sum() else np.nan,
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
            'brier_score': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'precision': float(tp / (tp + fp)) if tp + fp else np.nan,
            'recall': float(tp / (tp + fn)) if tp + fn else np.nan,
            'false_positive_rate': float(fp / (fp + tn)) if fp + tn else np.nan,
            'flagged_count': int(tp + fp), 'flagged_rate': float((tp + fp) / len(y)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def choose_threshold(y, p, target):
    """Highest score threshold achieving target recall (fewest flagged encounters)."""
    if not 0 < target <= 1 or np.asarray(y).sum() == 0:
        raise ValueError('Need a positive recall target and positive outcomes')
    _, recall, thresholds = precision_recall_curve(y, p)
    return float(thresholds[recall[:-1] >= target].max())


def reliability(y, p, **labels):
    frame = pd.DataFrame({'target': np.asarray(y), 'probability': np.asarray(p)})
    # Quantile bins have approximately equal counts; tied predictions remain together.
    frame['bin'] = pd.qcut(frame.probability, 10, duplicates='drop')
    if frame['bin'].isna().all():
        frame['bin'] = 'All'
    table = frame.groupby('bin', observed=True).agg(
        encounters=('target', 'size'), positive_count=('target', 'sum'),
        mean_probability=('probability', 'mean'), observed_rate=('target', 'mean'),
        min_probability=('probability', 'min'), max_probability=('probability', 'max')).reset_index()
    table['bin'] = table['bin'].astype(str)
    for key, value in labels.items():
        table[key] = value
    return table


def main():
    # Tie handling matters for a reproducible threshold and alert workload.
    yy = np.array([1, 0, 1, 0])
    pp = np.array([0.9, 0.8, 0.8, 0.1])
    tt = choose_threshold(yy, pp, 1.0)
    assert tt == 0.8 and (pp >= tt).sum() == 3
    assert np.isnan(score([0, 0], [0.1, 0.2], 0.5)['recall'])
    assert np.isnan(score([1, 0], [0.1, 0.2], 0.5)['precision'])
    out = ROOT / 'reports/calibration'
    out.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((ROOT / 'reports/model_comparison/run_metadata.json').read_text())
    source = ROOT / 'data/raw/diabetic_data.csv'
    assignments_path = ROOT / 'data/processed/partition_assignments.csv'
    for path, key in [(source, 'source_sha256'), (assignments_path, 'assignments_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == comparison[key]
    assignments = pd.read_csv(assignments_path, usecols=['encounter_id', 'patient_nbr', 'partition'])
    assert assignments.encounter_id.is_unique
    assert assignments.groupby('patient_nbr').partition.nunique().max() == 1

    def load_partition(partition):
        ids = assignments.loc[assignments.partition.eq(partition), ['encounter_id', 'patient_nbr']]
        pieces = []
        selected_ids = set(ids.encounter_id)
        for chunk in pd.read_csv(source, keep_default_na=False, na_values=['?'], low_memory=False, chunksize=5000):
            pieces.append(chunk.loc[chunk.encounter_id.isin(selected_ids)].copy())
        frame = pd.concat(pieces, ignore_index=True)
        assert len(frame) == len(ids) and frame.encounter_id.is_unique
        assert set(zip(frame.encounter_id, frame.patient_nbr)) == set(zip(ids.encounter_id, ids.patient_nbr))
        return frame

    train = load_partition('train')
    y = train.readmitted.eq('<30').astype(int)
    groups = train.patient_nbr.copy()
    X = train[comparison['raw_features']].copy()
    del train
    gc.collect()
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups))
    for a, b in splits:
        assert set(groups.iloc[a]).isdisjoint(set(groups.iloc[b]))
        assert y.iloc[a].nunique() == 2 and y.iloc[b].nunique() == 2
    base_path = ROOT / 'models/selected_engineered_model.joblib'
    base = joblib.load(base_path)
    # Explicit split indices enforce patient separation even though calibration does not accept groups.
    # ensemble=False uses pooled out-of-fold scores for sigmoid fitting, then refits on all training rows.
    calibrated = CalibratedClassifierCV(estimator=clone(base), method='sigmoid', cv=splits,
                                        ensemble=False, n_jobs=1)
    print('Fitting sigmoid calibration on three patient-grouped training folds, then refitting base model.', flush=True)
    with threadpool_limits(limits=2):
        calibrated.fit(X, y)
    training_encounters = len(y)
    training_patients = groups.nunique()
    del X, y, splits
    gc.collect()
    val = load_partition('validation')
    assert set(groups).isdisjoint(set(val.patient_nbr))
    y_val = val.readmitted.eq('<30').astype(int).to_numpy()
    print('Evaluating uncalibrated and sigmoid-calibrated candidates on validation only.', flush=True)
    with threadpool_limits(limits=2):
        probabilities = {'uncalibrated': base.predict_proba(val[comparison['raw_features']])[:, 1],
                         'sigmoid': calibrated.predict_proba(val[comparison['raw_features']])[:, 1]}
    quality = pd.DataFrame([{'variant': name, **score(y_val, p, 0.5)} for name, p in probabilities.items()])
    quality.to_csv(out / 'validation_probability_metrics.csv', index=False)
    q = quality.set_index('variant')
    # Conservative pre-specified comparison: retain calibration only if both proper scoring rules improve.
    selected = 'sigmoid' if (q.loc['sigmoid', 'brier_score'] < q.loc['uncalibrated', 'brier_score'] and
                             q.loc['sigmoid', 'log_loss'] < q.loc['uncalibrated', 'log_loss']) else 'uncalibrated'
    p = probabilities[selected]
    threshold = choose_threshold(y_val, p, RECALL_TARGET)
    operating = score(y_val, p, threshold)
    assert operating['recall'] >= RECALL_TARGET
    higher = np.unique(p[p > threshold])
    if len(higher):
        assert score(y_val, p, float(higher.min()))['recall'] < RECALL_TARGET
    threshold_rows = [score(y_val, p, choose_threshold(y_val, p, target)) | {'recall_objective': target}
                      for target in [0.50, 0.60, 0.70, 0.80, 0.90]]
    pd.DataFrame(threshold_rows).to_csv(out / 'recall_target_tradeoffs.csv', index=False)
    overall_curves = pd.concat([reliability(y_val, pp, variant=name) for name, pp in probabilities.items()], ignore_index=True)
    overall_curves.to_csv(out / 'calibration_bins.csv', index=False)
    audit = val[['encounter_id', 'patient_nbr', 'age', 'gender', 'race']].copy()
    audit['target'] = y_val
    audit['probability'] = p
    audit['flagged'] = p >= threshold
    audit.to_csv(out / 'validation_predictions.csv', index=False)
    subgroup_rows, subgroup_curves = [], []
    for attribute in ['age', 'gender', 'race']:
        labels = val[attribute].fillna('Missing').to_numpy()
        for label in sorted(set(labels)):
            mask = labels == label
            row = {'attribute': attribute, 'group': label,
                   'patients': int(val.loc[mask, 'patient_nbr'].nunique()), **score(y_val[mask], p[mask], threshold)}
            # Flag inadequate denominators instead of making strong comparisons for tiny groups.
            row['small_group'] = bool(row['encounters'] < 100 or row['positive_count'] < 30 or
                                      row['encounters'] - row['positive_count'] < 30)
            subgroup_rows.append(row)
            if not row['small_group']:
                subgroup_curves.append(reliability(y_val[mask], p[mask], attribute=attribute, group=label))
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(out / 'subgroup_metrics.csv', index=False)
    if subgroup_curves:
        pd.concat(subgroup_curves, ignore_index=True).to_csv(out / 'subgroup_calibration_bins.csv', index=False)
    gaps = []
    for attribute, frame in subgroup.loc[~subgroup.small_group].groupby('attribute'):
        gaps.append({'attribute': attribute, 'included_groups': len(frame),
                     'recall_min': frame.recall.min(), 'recall_max': frame.recall.max(),
                     'recall_gap': frame.recall.max() - frame.recall.min(),
                     'precision_min': frame.precision.min(), 'precision_max': frame.precision.max(),
                     'false_positive_rate_min': frame.false_positive_rate.min(),
                     'false_positive_rate_max': frame.false_positive_rate.max()})
    pd.DataFrame(gaps).to_csv(out / 'subgroup_gaps.csv', index=False)
    pipeline_path = ROOT / 'models/sigmoid_calibrated_model.joblib'
    joblib.dump(calibrated, pipeline_path)
    reloaded = joblib.load(pipeline_path)
    with threadpool_limits(limits=2):
        np.testing.assert_allclose(reloaded.predict_proba(val[comparison['raw_features']].head(20))[:, 1],
                                   probabilities['sigmoid'][:20])
    selected_path = pipeline_path if selected == 'sigmoid' else base_path
    policy = {'status': 'provisional educational operating policy', 'selected_variant': selected,
              'model_path': str(selected_path.relative_to(ROOT)),
              'model_sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest(),
              'threshold': threshold, 'decision_rule': 'probability >= threshold',
              'threshold_objective': 'highest validation threshold with recall >= 0.70; minimizes flags subject to recall target',
              'recall_target': RECALL_TARGET, 'objective_origin': 'illustrative project assumption; not a clinical requirement',
              'calibration_selection': 'sigmoid only if validation Brier and log loss both improve; otherwise uncalibrated',
              'validation_metrics': operating, 'test_evaluated': False,
              'features': comparison['raw_features']}
    (ROOT / 'models/operating_policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    metadata = {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                'method': 'sigmoid; ensemble=False; three-fold StratifiedGroupKFold; seed=42',
                'training_encounters': training_encounters, 'training_patients': int(training_patients),
                'validation_encounters': len(val), 'test_evaluated': False,
                'source_sha256': comparison['source_sha256'],
                'assignments_sha256': comparison['assignments_sha256'],
                'base_model_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest()}
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Standalone plot uses validation outcomes only; bins are descriptive, not confidence intervals.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, frame in overall_curves.groupby('variant'):
        axes[0].plot(frame.mean_probability, frame.observed_rate, marker='o', label=name)
        axes[1].plot(frame.mean_probability, frame.observed_rate, marker='o', label=name)
    for axis in axes:
        axis.plot([0, 1], [0, 1], '--', color='gray', label='ideal')
        axis.set(xlabel='Mean predicted probability', ylabel='Observed readmission fraction')
        axis.grid(alpha=0.2)
    axes[0].set(title='Validation calibration', xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title='Detail: common probability range', xlim=(0, 0.4), ylim=(0, 0.4))
    axes[0].legend()
    fig.suptitle('Approximately equal-count bins; no test data used')
    fig.tight_layout()
    fig.savefig(out / 'calibration_curve.png', dpi=160)
    plt.close(fig)
    report = f'''# Calibration, threshold, and demographic audit

## Probability calibration

The selected gradient boosting pipeline was cloned and calibrated with sigmoid scaling using pooled out-of-fold scores from three patient-grouped training folds (seed 42, ensemble=False). All preprocessing and base-model fits are inside those folds. The calibrated estimator then refits the base model on all {training_encounters:,} training encounters. Validation outcomes did not fit the calibrator. The previously saved full-training base pipeline provides the uncalibrated comparison. Both candidates are evaluated on {len(val):,} validation encounters.

```text
{quality[['variant', 'average_precision', 'roc_auc', 'brier_score', 'log_loss', 'mean_probability', 'readmission_rate']].round(5).to_string(index=False)}
```

Selection rule, fixed before inspecting results: use sigmoid only if both validation Brier score and log loss improve; otherwise retain uncalibrated. Selected: **{selected}**. Lower Brier/log loss is better, but both measure overall probabilistic accuracy and cannot alone establish perfect calibration. See calibration_curve.png and calibration_bins.csv for approximately equal-count reliability bins; tied probabilities stay together. Calibration selection used development data, so these are not unbiased final performance estimates. Monotone sigmoid calibration should preserve ranking; it changes the probability scale, not clinical discrimination.

## Provisional operating threshold

Because no intervention capacity or cost criterion was supplied, use the explicit educational assumption: recall at least 70%, choosing the highest validation score threshold that achieves it. This minimizes flagged encounters under that recall constraint. It does not optimize clinical utility or establish a medically validated cutoff. Equal-to-threshold scores are flagged; tied scores cannot be split.

- Selected variant: {selected}
- Threshold: {threshold:.10f} (use the full value in models/operating_policy.json)
- Recall: {operating['recall']:.2%}; precision: {operating['precision']:.2%}
- Flagged: {operating['flagged_count']:,} / {len(val):,} encounters ({operating['flagged_rate']:.2%})
- True positives: {operating['tp']:,}; false positives: {operating['fp']:,}
- False negatives: {operating['fn']:,}; true negatives: {operating['tn']:,}

```text
{pd.DataFrame(threshold_rows)[['recall_objective', 'threshold', 'precision', 'recall', 'flagged_rate', 'fp', 'fn']].round(5).to_string(index=False)}
```

The alert workload is an essential tradeoff. This provisional policy needs revision if the intended intervention cannot support that volume. The recall target is a development criterion, not a guarantee on future or test data. Do not change it after inspecting final test performance.

## Demographic performance

Audit uses the same selected model and global threshold for every age, gender, and race group. subgroup_metrics.csv reports encounter and unique-patient counts, positive counts, prevalence, precision, recall, false-positive rate, flagged rate, mean predicted probability, calibration gap (mean probability minus observed rate), Brier score, log loss, AP, and ROC-AUC. subgroup_calibration_bins.csv provides reliability bins for sufficiently large groups.

Descriptive recall ranges among groups with at least 100 encounters, 30 positives, and 30 negatives:

```text
{pd.DataFrame(gaps).round(5).to_string(index=False)}
```

Small groups are flagged, not silently removed; undefined metrics are blank rather than zero. These are encounter-weighted descriptive estimates, with repeated encounters per patient and no confidence intervals. Group gaps may reflect sampling variation, prevalence, coding, and case mix; they do not prove discrimination, fairness, or equal clinical benefit. Calibration-gap averages can hide within-group miscalibration; examine the bins as well. Do not interpret small-group differences as reliable or use them to create demographic-specific thresholds without further study.

## Artifacts and verification

models/sigmoid_calibrated_model.joblib stores the full calibration/preprocessing/model estimator. models/operating_policy.json points to the selected model, its checksum, required input features, exact threshold, and decision rule. Existing uncalibrated and comparison artifacts are preserved. Keep the src package importable when loading trusted joblib files.

Verified source/assignment checksums, unique assignments, patient-disjoint outer and calibration folds, complete partition loading, threshold tie handling and minimal flagging at the target, finite probabilities, safe undefined subgroup metrics, and calibrated model prediction equality after reload. Test outcomes have not been evaluated.

Next: review the provisional workload and freeze the intended operating policy before final test evaluation. Explainability can be added before opening the test set. This historical-data project is educational and is not clinically validated.
'''
    (out / 'calibration_and_fairness_report.md').write_text(report, encoding='utf-8')
    print(quality[['variant', 'brier_score', 'log_loss']].to_string(index=False), flush=True)
    print(json.dumps(policy, indent=2), flush=True)


if __name__ == '__main__':
    main()
