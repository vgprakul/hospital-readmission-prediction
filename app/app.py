"""Run from the project root: python -m streamlit run app/app.py."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from app.runtime import load_artifacts, predict, explain

st.set_page_config(page_title='Readmission Explorer', page_icon='🏥', layout='wide')
st.markdown('''<style>
.stApp {background: #f6f8fb;}
h1,h2,h3 {color:#16354a;}
[data-testid="stMetric"] {background:white;border:1px solid #dde5ed;border-radius:12px;padding:16px;}
[data-testid="stSidebar"] {background:#eaf0f5;}
</style>''', unsafe_allow_html=True)


@st.cache_resource
def artifacts():
    return load_artifacts()


try:
    model, policy, schema = artifacts()
except Exception as error:
    st.error(f'Unable to load frozen app artifacts: {error}')
    st.stop()

LABELS = {'time_in_hospital': 'Length of stay (days)', 'num_lab_procedures': 'Lab procedures',
          'num_procedures': 'Procedures', 'num_medications': 'Medications during encounter',
          'number_outpatient': 'Prior outpatient visits', 'number_emergency': 'Prior emergency visits',
          'number_inpatient': 'Prior inpatient visits', 'number_diagnoses': 'Number of diagnoses',
          'diag_1': 'Primary diagnosis (ICD-9)', 'diag_2': 'Secondary diagnosis (ICD-9)',
          'diag_3': 'Additional diagnosis (ICD-9)', 'A1Cresult': 'HbA1c result',
          'max_glu_serum': 'Glucose result', 'diabetesMed': 'Diabetes medication prescribed',
          'change': 'Diabetes medication changed', 'admission_type_id': 'Admission type',
          'admission_source_id': 'Admission source', 'discharge_disposition_id': 'Discharge destination',
          'payer_code': 'Payer', 'medical_specialty': 'Admitting specialty', 'weight': 'Weight band (lb)'}


def label(column):
    return LABELS.get(column, column.replace('_', ' ').capitalize())


def option_label(column, value):
    if value == '__MISSING__':
        return 'Unknown / missing'
    spec = schema['fields'][column]
    if 'labels' in spec:
        return spec['labels'].get(str(value), str(value))
    if column in ['A1Cresult', 'max_glu_serum'] and value == 'None':
        return 'Not measured / no result recorded'
    return str(value)


def set_example(index):
    for column, value in schema['examples'][index].items():
        st.session_state['field_' + column] = value


if 'field_age' not in st.session_state:
    set_example(0)

st.sidebar.title('Readmission Explorer')
st.sidebar.caption('Historical diabetes encounters · educational ML')
page = st.sidebar.radio('Page', ['Readmission predictor', 'Prediction explanation', 'Model performance'])
st.sidebar.divider()
st.sidebar.caption(f'Frozen score threshold: {policy["threshold"]:.5f}')
st.sidebar.caption('Model decisions are illustrative and are not clinical recommendations.')
st.title('Readmission Explorer')
st.caption('Explore a frozen model for recorded readmission within 30 days of discharge.')
st.warning('Educational demonstration using historical US diabetes data. Not validated for clinical use or medical decisions.')


def result_cards(result):
    left, right = st.columns(2)
    left.metric('Estimated readmission probability', f'{result["probability"]:.1%}')
    right.metric('Frozen policy flag', 'Above threshold' if result['flagged'] else 'Below threshold')
    st.caption(f"Flag rule: score ≥ {result['threshold']:.5f}. Below threshold does not mean no risk. "
               'On the final test set, this policy flagged 49.5% of encounters with 15.9% precision.')


if page == 'Readmission predictor':
    st.header('Enter an encounter at discharge')
    st.write('Load a historical training example, or edit every input below. No names or patient identifiers are needed.')
    example = st.selectbox('Training example', [0, 1, 2], format_func=lambda x: f'Example {x + 1}')
    st.button('Load example', on_click=set_example, args=(example,))
    st.caption('Examples come from the training partition; they are not test cases. Edits take effect when you submit.')
    with st.form('encounter'):
        values = {}

        def fields(columns):
            grid = st.columns(3)
            for i, column in enumerate(columns):
                spec = schema['fields'][column]
                with grid[i % 3]:
                    if spec['kind'] == 'integer':
                        values[column] = st.number_input(label(column), min_value=spec['min'], max_value=spec['max'],
                                                         step=1, key='field_' + column)
                    else:
                        values[column] = st.selectbox(label(column), spec['options'], key='field_' + column,
                                                      format_func=lambda v, c=column: option_label(c, v))
        st.subheader('Patient and encounter')
        fields(['age', 'gender', 'race', 'admission_type_id', 'admission_source_id', 'discharge_disposition_id'])
        fields(['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'])
        st.caption('Prior outpatient, emergency, and inpatient visits refer to the year before this encounter.')
        with st.expander('Diagnoses, laboratory results, and context', expanded=False):
            fields(['diag_1', 'diag_2', 'diag_3', 'A1Cresult', 'max_glu_serum',
                    'weight', 'payer_code', 'medical_specialty'])
        with st.expander('Diabetes medication details', expanded=False):
            fields([c for c in policy['features'] if c not in values])
            st.caption('No = not prescribed; Steady = unchanged; Up/Down = dose change. Defaults are from the loaded example.')
        submitted = st.form_submit_button('Predict readmission score', type='primary')
    if submitted:
        try:
            with st.spinner('Computing score…'):
                st.session_state['prediction'] = predict(model, policy, schema, values)
                st.session_state['submitted_values'] = values.copy()
                st.session_state.pop('explanation', None)
        except ValueError as error:
            st.error(str(error))
    if 'prediction' in st.session_state:
        st.divider()
        st.subheader('Last submitted prediction')
        result_cards(st.session_state['prediction'])
        st.info('Open Prediction explanation in the sidebar to inspect this submitted encounter.')
elif page == 'Prediction explanation':
    st.header('What influenced the submitted score?')
    if 'prediction' not in st.session_state:
        st.info('Submit an encounter on the Readmission predictor page first.')
    else:
        result_cards(st.session_state['prediction'])
        st.caption('This explanation uses the last submitted encounter, even if the form has since been edited.')
        if st.button('Explain submitted prediction', type='primary'):
            with st.spinner('Computing SHAP contributions…'):
                st.session_state['explanation'] = explain(model, policy, schema, st.session_state['submitted_values'])
        if 'explanation' in st.session_state:
            table, base, _ = st.session_state['explanation']
            st.write('Positive values raise the score relative to the model baseline; negative values lower it.')
            largest = table.loc[table.shap_log_odds.abs().nlargest(12).index].sort_values('shap_log_odds')
            plot = largest.assign(factor=largest.feature.map(label)).set_index('factor')[['shap_log_odds']]
            st.bar_chart(plot, horizontal=True, color='#237b91')
            st.caption('SHAP values are additive in log-odds, not percentage points. Correlated features may share attribution. '
                       'These are model associations, not causes or treatment advice.')
            st.dataframe(table.assign(feature=table.feature.map(label)), hide_index=True, width='stretch')
            st.caption(f'Baseline log-odds: {base:.4f}. Baseline + all contributions reconstructs the submitted score (verified).')
    with st.expander('Global explanations from validation data'):
        st.image(str(ROOT / 'reports/explainability/global_shap_importance.png'))
        st.caption('Fixed sample of 1,000 validation encounters. This is not a new test evaluation.')
else:
    st.header('Final held-out performance')
    metrics = json.loads((ROOT / 'reports/final_test/test_metrics.json').read_text())
    columns = st.columns(4)
    for column, title, key in zip(columns, ['ROC-AUC', 'Average precision', 'Recall', 'Precision'],
                                  ['roc_auc', 'average_precision', 'recall', 'precision']):
        column.metric(title, f'{metrics[key]:.3f}' if key in ['roc_auc', 'average_precision'] else f'{metrics[key]:.1%}')
    st.write(f"Evaluated once on **{metrics['encounters']:,} encounters** from **{metrics['patients']:,} patients**. "
             f"Readmission prevalence: **{metrics['readmission_rate']:.1%}**. Model and threshold were frozen before evaluation.")
    st.info(f"Workload tradeoff: {metrics['flagged_count']:,} encounters flagged ({metrics['flagged_rate']:.1%}); "
            f"{metrics['fp']:,} false positives and {metrics['fn']:,} missed readmissions. "
            'The threshold targeted 70% recall on validation data; this is an illustrative objective.')
    st.subheader('Confusion matrix')
    st.table(pd.DataFrame([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                          index=['Actual: no early readmission', 'Actual: early readmission'],
                          columns=['Below threshold', 'Above threshold']))
    st.image(str(ROOT / 'reports/final_test/test_evaluation.png'))
    st.caption('Average precision summarizes precision–recall ranking. Calibration bins compare estimated probabilities '
               'with observed fractions; neither this plot nor Brier score establishes clinical validity.')
    with st.expander('Patient-cluster bootstrap intervals'):
        st.dataframe(pd.read_csv(ROOT / 'reports/final_test/patient_bootstrap_intervals.csv'), hide_index=True)
        st.caption('300 patient-resampling draws; intervals do not capture temporal or hospital distribution shifts.')
    st.subheader('Demographic performance')
    subgroup = pd.read_csv(ROOT / 'reports/final_test/test_subgroup_metrics.csv')
    attribute = st.selectbox('Group by', ['age', 'gender', 'race'])
    shown = subgroup.loc[subgroup.attribute.eq(attribute), ['group', 'encounters', 'patients', 'positive_count',
                        'precision', 'recall', 'false_positive_rate', 'brier_score', 'calibration_gap', 'small_group']]
    st.dataframe(shown, hide_index=True, width='stretch')
    st.caption('Small group = fewer than 100 encounters or 30 positive/negative outcomes. Undefined metrics are blank. '
               'These descriptive differences do not prove fairness; subgroup confidence intervals are not provided.')
    st.subheader('Scope and limitations')
    st.write('Historical 1999–2008 records from US hospitals, restricted to encounters involving diabetes. '
             'The source labels do not establish complete follow-up across hospitals or distinguish every planned readmission. '
             'Transfers are retained; deaths, hospice, and unclear completed discharges were excluded. '
             'Patient-separated testing does not establish present-day, temporal, or hospital-level external validity. '
             'Model scores and threshold flags must not be used for medical decisions.')
    st.download_button('Download final evaluation report',
                       (ROOT / 'reports/final_test/final_test_report.md').read_text(encoding='utf-8'),
                       file_name='final_test_report.md', mime='text/markdown')
