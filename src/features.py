"""Deterministic encounter-level features; no population statistics are learned."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MEDICATIONS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
               'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
               'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
               'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
               'glipizide-metformin', 'glimepiride-pioglitazone',
               'metformin-rosiglitazone', 'metformin-pioglitazone']


def diagnosis_group(series):
    """Project-defined broad ICD-9 groups; preserve unknown and missing separately."""
    value = pd.to_numeric(series, errors='coerce')
    group = pd.Series('Other', index=series.index, dtype=object)
    for low, high, label in [(390, 460, 'Circulatory'), (460, 520, 'Respiratory'),
                             (520, 580, 'Digestive'), (580, 630, 'Genitourinary'),
                             (710, 740, 'Musculoskeletal'), (800, 1000, 'Injury'),
                             (140, 240, 'Neoplasm'), (250, 251, 'Diabetes')]:
        group.loc[value.ge(low) & value.lt(high)] = label
    group.loc[series.isna()] = 'Missing'
    return group


class EncounterFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        result['total_previous_visits'] = result[
            ['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1, min_count=3)
        result['active_diabetes_medications'] = result[MEDICATIONS].isin(['Steady', 'Up', 'Down']).sum(axis=1)
        result['changed_diabetes_medications'] = result[MEDICATIONS].isin(['Up', 'Down']).sum(axis=1)
        for column in ['diag_1', 'diag_2', 'diag_3']:
            result[column + '_group'] = diagnosis_group(result[column])
        return result.drop(columns=['diag_1', 'diag_2', 'diag_3'])
