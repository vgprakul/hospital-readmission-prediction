"""Fixed patient-grouped CV experiments, then validation of the CV-selected model."""
from pathlib import Path
import hashlib
import gc
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from src.features import EncounterFeatures, diagnosis_group

CODE_COLUMNS = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']


def metrics(y, p, threshold=0.5):
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'average_precision': average_precision_score(y, p), 'roc_auc': roc_auc_score(y, p),
            'brier_score': brier_score_loss(y, p), 'log_loss': log_loss(y, p),
            'threshold': threshold, 'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0),
            'f1': f1_score(y, predicted, zero_division=0),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def main():
    out = ROOT / 'reports/model_comparison'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/raw/diabetic_data.csv'
    assignment_path = ROOT / 'data/processed/partition_assignments.csv'
    cohort = json.loads((ROOT / 'reports/cohort_split_metadata.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cohort['source_sha256']
    assignments = pd.read_csv(assignment_path, usecols=['encounter_id', 'patient_nbr', 'partition'])
    assert assignments.encounter_id.is_unique
    assert assignments.groupby('patient_nbr').partition.nunique().max() == 1
    train_ids = assignments.loc[assignments.partition.eq('train'), ['encounter_id', 'patient_nbr']]
    # Filter records before constructing train/validation frames; never score the test partition.
    def load_partition(ids):
        pieces = []
        selected_ids = set(ids.encounter_id)
        for chunk in pd.read_csv(source, keep_default_na=False, na_values=['?'],
                                 low_memory=False, chunksize=5000):
            pieces.append(chunk.loc[chunk.encounter_id.isin(selected_ids)].copy())
        return pd.concat(pieces, ignore_index=True).merge(
            ids, on=['encounter_id', 'patient_nbr'], validate='one_to_one')
    train = load_partition(train_ids)
    assert len(train) == len(train_ids)
    excluded = ['encounter_id', 'patient_nbr', 'readmitted']
    features = [c for c in train if c not in excluded]
    X = train[features]
    y = train.readmitted.eq('<30').astype(int)
    groups = train.patient_nbr
    engineered = EncounterFeatures().fit_transform(X.head(10))
    numeric = [c for c in engineered if pd.api.types.is_numeric_dtype(engineered[c]) and c not in CODE_COLUMNS]
    categorical = [c for c in engineered if c not in numeric + CODE_COLUMNS]
    # Meaningful boundary checks for the project diagnosis mapping.
    assert diagnosis_group(pd.Series(['250.13', '390', '459', '460', 'V45', None])).tolist() == [
        'Diabetes', 'Circulatory', 'Circulatory', 'Respiratory', 'Other', 'Missing']
    pd.DataFrame({'column': X.columns, 'training_missing_percent': X.isna().mean().values * 100,
                  'training_unique_values': X.nunique().values}).to_csv(out / 'training_feature_audit.csv', index=False)
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore', dtype=np.float32))]), categorical),
        ('codes', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CODE_COLUMNS),
    ], sparse_threshold=1.0)
    models = {
        'logistic_engineered': LogisticRegression(solver='liblinear', max_iter=2000, random_state=42),
        'random_forest_engineered': RandomForestClassifier(n_estimators=120, max_depth=14,
                                    min_samples_leaf=15, max_features='sqrt', n_jobs=2, random_state=42),
        'hist_gradient_boosting_engineered': HistGradientBoostingClassifier(max_iter=150, learning_rate=0.07,
                                    max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=5,
                                    early_stopping=False, random_state=42),
    }
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups))
    fold_assignments = train[['encounter_id', 'patient_nbr']].copy()
    fold_assignments['fold'] = -1
    for fold, (a, b) in enumerate(splits, 1):
        assert set(groups.iloc[a]).isdisjoint(set(groups.iloc[b]))
        fold_assignments.loc[fold_assignments.index[b], 'fold'] = fold
    assert fold_assignments.fold.gt(0).all()
    assert fold_assignments.groupby('patient_nbr').fold.nunique().max() == 1
    fold_assignments.to_csv(out / 'training_cv_assignments.csv', index=False)
    del train, fold_assignments, engineered
    gc.collect()
    rows = []
    for name, estimator in models.items():
        model_preprocessing = clone(preprocessing)
        if name == 'hist_gradient_boosting_engineered':
            model_preprocessing.set_params(categorical__encoder__sparse_output=False,
                                            codes__sparse_output=False, sparse_threshold=0.0)
        for fold, (a, b) in enumerate(splits, 1):
            print(f'Training {name}, fold {fold}/3', flush=True)
            model = Pipeline([('features', EncounterFeatures()), ('preprocessing', clone(model_preprocessing)),
                              ('model', clone(estimator))])
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[a], y.iloc[a])
                p = model.predict_proba(X.iloc[b])[:, 1]
            assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            row = {'model': name, 'fold': fold, **metrics(y.iloc[b], p),
                   'seconds': time.perf_counter() - start}
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'cv_fold_metrics.csv', index=False)
            print(f"AP={row['average_precision']:.4f}; ROC-AUC={row['roc_auc']:.4f}", flush=True)
            del model, p
            gc.collect()
    results = pd.DataFrame(rows)
    summary = results.groupby('model')[['average_precision', 'roc_auc', 'brier_score', 'log_loss']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.sort_values('average_precision_mean', ascending=False)
    summary.to_csv(out / 'cv_model_summary.csv')
    winner = summary.index[0]
    print('CV selected: ' + winner, flush=True)
    winning_preprocessing = clone(preprocessing)
    if winner == 'hist_gradient_boosting_engineered':
        winning_preprocessing.set_params(categorical__encoder__sparse_output=False,
                                         codes__sparse_output=False, sparse_threshold=0.0)
    selected_model = Pipeline([('features', EncounterFeatures()), ('preprocessing', winning_preprocessing),
                               ('model', clone(models[winner]))])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always', ConvergenceWarning)
        selected_model.fit(X, y)
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    val_ids = assignments.loc[assignments.partition.eq('validation'), ['encounter_id', 'patient_nbr']]
    val = load_partition(val_ids)
    assert len(val) == len(val_ids)
    assert set(val.patient_nbr).isdisjoint(set(groups))
    y_val = val.readmitted.eq('<30').astype(int)
    with threadpool_limits(limits=2):
        p = selected_model.predict_proba(val[features])[:, 1]
    validation = {'model': winner, **metrics(y_val, p)}
    pd.DataFrame([validation]).to_csv(out / 'selected_validation_metrics.csv', index=False)
    thresholds = pd.DataFrame([metrics(y_val, p, float(t)) for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]])
    thresholds.to_csv(out / 'validation_threshold_tradeoffs.csv', index=False)
    model_path = ROOT / 'models/selected_engineered_model.joblib'
    joblib.dump(selected_model, model_path)
    restored = joblib.load(model_path)
    with threadpool_limits(limits=2):
        np.testing.assert_allclose(restored.predict_proba(val[features].head(20))[:, 1], p[:20])
    val[['encounter_id', 'patient_nbr']].assign(target=y_val, probability=p).to_csv(out / 'validation_predictions.csv', index=False)
    metadata = {'selected_model': winner, 'selection_rule': 'highest mean training CV average precision',
                'folds': 3, 'seed': 42, 'test_evaluated': False, 'threshold_selected': False,
                'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                'joblib': joblib.__version__, 'source_sha256': cohort['source_sha256'],
                'assignments_sha256': hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
                'raw_features': features, 'numeric_after_engineering': numeric,
                'categorical_after_engineering': categorical, 'code_columns': CODE_COLUMNS,
                'models': {name: estimator.get_params() for name, estimator in models.items()}}
    (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report = f'''# Engineered features and grouped model comparison

## Training-only comparison

All candidates used identical three-fold StratifiedGroupKFold splits of the reserved training partition (69,651 encounters). Every patient's encounters stay in one fold. Feature transforms are deterministic; each fold fits its own imputation, scaling, encoding, and model on the fitting subset only. No validation or test outcome was used to rank models.

```text
{summary.round(4).to_string()}
```

Selected **{winner}** by highest mean fold average precision. Standard deviations describe fold variation, not confidence intervals. Average precision is the non-interpolated precision-recall summary, not trapezoidal PR-AUC. This is a fixed-parameter comparison, not an exhaustive tuning study; fold differences do not establish statistical superiority.

## Feature changes

Add total previous outpatient/emergency/inpatient visits, count of medication columns marked Steady/Up/Down, and count marked Up/Down. Replace each detailed diagnosis with project-defined broad numeric ICD-9 ranges: circulatory [390,460), respiratory [460,520), digestive [520,580), genitourinary [580,630), musculoskeletal [710,740), injury [800,1000), neoplasm [140,240), diabetes [250,251); all other codes (including V/E prefixes) map to Other, with Missing separate. These simple groups are not a comprehensive clinical ontology. Medication counts count active columns, not individual active ingredients.

Retain the remaining raw predictors, including sparse weight, for this fixed experiment. Admission/discharge codes remain categorical. Numerical fields receive median imputation/scaling; categorical missing values receive an explicit category; unseen categories are ignored. Original lab None categories are preserved. IDs and outcomes never enter predictors. See training_feature_audit.csv for training-only column statistics. This comparison does not isolate the effect of engineering: an ablation would compare raw and engineered features on identical CV folds.

## Selected model on validation

Refit the CV-selected pipeline on all training encounters, then score the reserved validation partition only.

```text
{pd.DataFrame([validation]).round(4).to_string(index=False)}
```

Validation threshold tradeoffs:

```text
{thresholds[['threshold', 'precision', 'recall', 'f1', 'fp', 'fn', 'tp']].round(4).to_string(index=False)}
```

No operating threshold has been selected. A recall target or intervention capacity/cost criterion is needed to choose one; these thresholds are exploratory project values, not clinical standards. Validation is now development data, so its threshold metrics are not an unbiased final performance estimate. Test data remains reserved. Probability calibration has not been validated.

## Artifacts and verification

Saved the full raw-feature-to-probability pipeline as models/selected_engineered_model.joblib. From the project root load it with joblib (the src package must be importable). Input columns are recorded in run_metadata.json. Only load trusted artifacts.

Verified source checksum, assignment uniqueness, patient-disjoint outer and CV partitions, complete train/validation joins, diagnosis grouping boundaries, convergence, finite probabilities, and prediction agreement after model reload. CSV files contain fold metrics, summaries, validation metrics, and threshold tradeoffs. Test outcomes were not evaluated.

Next: review the validation precision/recall tradeoff and define an operating objective, then assess calibration and demographic performance. Any further model tuning should use training-only grouped CV. Educational historical-data experiment; not clinically validated.
'''
    (out / 'comparison_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(validation), flush=True)
    print(thresholds[['threshold', 'precision', 'recall', 'fp']].round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
