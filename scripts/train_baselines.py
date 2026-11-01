"""Train discharge-risk baselines without evaluating the reserved test partition."""
from pathlib import Path
import hashlib
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CODE_COLUMNS = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']


def main():
    source = ROOT / 'data/raw/diabetic_data.csv'
    assignment_path = ROOT / 'data/processed/partition_assignments.csv'
    metadata = json.loads((ROOT / 'reports/cohort_split_metadata.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata['source_sha256']
    assignments = pd.read_csv(assignment_path)
    assert assignments.encounter_id.is_unique
    assert set(assignments.partition) == {'train', 'validation', 'test'}
    assert assignments.groupby('patient_nbr').partition.nunique().max() == 1
    # Select training/validation assignments; test outcomes never enter fitting or metrics.
    selected = assignments.loc[assignments.partition.isin(['train', 'validation']),
                               ['encounter_id', 'patient_nbr', 'partition']]
    chunks = []
    for chunk in pd.read_csv(source, keep_default_na=False, na_values=['?'],
                             low_memory=False, chunksize=10000):
        chunks.append(chunk.loc[chunk.encounter_id.isin(selected.encounter_id)])
    data = pd.concat(chunks, ignore_index=True).merge(
        selected, on=['encounter_id', 'patient_nbr'], validate='one_to_one')
    assert len(data) == len(selected)
    assert data.readmitted.isin(['<30', '>30', 'NO']).all()
    train = data.loc[data.partition.eq('train')]
    val = data.loc[data.partition.eq('validation')]
    features = [c for c in data if c not in ['encounter_id', 'patient_nbr', 'readmitted', 'partition']]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c]) and c not in CODE_COLUMNS]
    categorical = [c for c in features if c not in numeric + CODE_COLUMNS]
    X_train, X_val = train[features], val[features]
    y_train, y_val = train.readmitted.eq('<30').astype(int), val.readmitted.eq('<30').astype(int)
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical),
        ('codes', OneHotEncoder(handle_unknown='ignore'), CODE_COLUMNS),
    ], sparse_threshold=1.0)
    estimators = {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logistic_regression': LogisticRegression(solver='liblinear', max_iter=2000, random_state=42),
        'logistic_regression_balanced': LogisticRegression(solver='liblinear', max_iter=2000,
                                                           class_weight='balanced', random_state=42),
    }
    model_dir = ROOT / 'models'
    report_dir = ROOT / 'reports/baselines'
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = val[['encounter_id', 'patient_nbr']].copy()
    predictions['target'] = y_val
    for name, estimator in estimators.items():
        print('Training ' + name, flush=True)
        pipeline = Pipeline([('preprocessing', clone(preprocessing)), ('model', estimator)])
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise RuntimeError(f'{name} failed to converge; resolve before reporting results')
        probability = pipeline.predict_proba(X_val)[:, 1]
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        predicted = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, predicted, labels=[0, 1]).ravel()
        rows.append({'model': name, 'average_precision': average_precision_score(y_val, probability),
                     'roc_auc': roc_auc_score(y_val, probability),
                     'precision': precision_score(y_val, predicted, zero_division=0),
                     'recall': recall_score(y_val, predicted, zero_division=0),
                     'f1': f1_score(y_val, predicted, zero_division=0),
                     'brier_score': brier_score_loss(y_val, probability),
                     'log_loss': log_loss(y_val, probability), 'threshold': 0.5,
                     'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                     'fit_and_predict_seconds': time.perf_counter() - start})
        path = model_dir / f'{name}.joblib'
        joblib.dump(pipeline, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict_proba(X_val.iloc[:20])[:, 1], probability[:20])
        predictions[name + '_probability'] = probability
        print(json.dumps(rows[-1]), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(report_dir / 'validation_metrics.csv', index=False)
    predictions.to_csv(report_dir / 'validation_predictions.csv', index=False)
    run = {'python': platform.python_version(), 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
           'numpy': np.__version__, 'joblib': joblib.__version__,
           'source_sha256': metadata['source_sha256'],
           'assignments_sha256': hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
           'training_encounters': len(train), 'validation_encounters': len(val),
           'training_positive_rate': float(y_train.mean()), 'validation_positive_rate': float(y_val.mean()),
           'numeric_features': numeric, 'categorical_features': categorical, 'code_features': CODE_COLUMNS,
           'test_evaluated': False, 'threshold': 0.5}
    (report_dir / 'run_metadata.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    table = results.drop(columns=['fit_and_predict_seconds']).round(4).to_string(index=False)
    report = f'''# Validation baseline comparison

Trained on {len(train):,} encounters; evaluated on {len(val):,} validation encounters. Validation readmission prevalence: {y_val.mean():.2%}. Reserved test partition was not used for fitting, feature selection, or evaluation.

```text
{table}
```

## Method

All three full pipelines fit their preprocessing only on training data. Numerical features: median imputation and standard scaling. String categories: explicit `Missing` category and one-hot encoding. Admission/discharge integer codes: one-hot encoding, preserving the distinct source unknown/unavailable codes as explicit categories. Unseen validation categories are ignored by the encoder. Literal laboratory `None` categories remain intact. All 47 source predictors are retained for this initial baseline, including sparse weight and detailed diagnosis codes; any later feature simplification should be based on training data.

Identifiers, source outcome, partition, and target are excluded from predictors. Models: prior-probability dummy, ordinary logistic regression, and balanced-class logistic regression. Logistic regression uses liblinear, C=1, max_iter=2000, random_state=42. No resampling, tuning, threshold selection, or calibration was performed.

Average precision is the non-interpolated precision-recall summary from sklearn, not trapezoidal PR-AUC. Precision/recall/F1 and confusion matrices use a fixed 0.50 threshold; tn/fp/fn/tp are shown above. Undefined precision for zero predicted positives is reported as zero. The dummy predicts the training prevalence and ranks every case equally.

Balanced weights change the error tradeoff and can distort probability calibration; their raw output should not be presented as calibrated patient risk. Compare ranking metrics separately from threshold-dependent metrics. Brier score and log loss are included as initial probability-quality diagnostics, not a calibration validation.

## Verification and artifacts

Verified source checksum, unique assignment IDs, patient-disjoint partitions, complete training/validation joins, convergence, finite probabilities, and saved-model reload prediction agreement. Full preprocessing/model pipelines are saved under `models/*.joblib`. Only load trusted joblib artifacts. Runtime package versions and feature lists are in `run_metadata.json`.

Next: investigate training-only feature engineering and patient-grouped cross-validation, compare additional models, and use validation for model/threshold selection. Keep the test partition reserved until the methodology is fixed. These historical-data models are educational and not clinically validated.
'''
    (report_dir / 'baseline_report.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
