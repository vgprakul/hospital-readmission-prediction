"""Exercise prediction, explanations, navigation, and the frozen artifact contract."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from streamlit.testing.v1 import AppTest
from threadpoolctl import threadpool_limits
from app.runtime import load_artifacts, make_frame, predict, explain


def main():
    frozen_path = ROOT / 'models/frozen_manifest.json'
    original_manifest_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    model, policy, schema = load_artifacts()
    assert len(schema['fields']) == len(policy['features']) == 47
    for example in schema['examples']:
        frame = make_frame(example, policy, schema)
        result = predict(model, policy, schema, example)
        with threadpool_limits(limits=2):
            expected = model.predict_proba(frame)[0, 1]
        np.testing.assert_allclose(result['probability'], expected, atol=1e-12)
        assert result['flagged'] == (expected >= policy['threshold'])
    missing = schema['examples'][0].copy()
    missing['race'] = '__MISSING__'
    missing['A1Cresult'] = 'None'
    frame = make_frame(missing, policy, schema)
    assert frame.race.isna().all() and frame.A1Cresult.iloc[0] == 'None'
    for bad_value in [-1, float('nan'), 1.5]:
        invalid = schema['examples'][0] | {'number_inpatient': bad_value}
        try:
            make_frame(invalid, policy, schema)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid numeric input accepted')
    at = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.selectbox) + len(at.number_input) >= 47
    next(b for b in at.button if b.label == 'Predict readmission score').click().run()
    assert not at.exception, at.exception
    actual = at.session_state['prediction']['probability']
    expected = predict(model, policy, schema, schema['examples'][0])['probability']
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    at.selectbox[0].set_value(1).run()
    next(b for b in at.button if b.label == 'Load example').click().run()
    next(b for b in at.button if b.label == 'Predict readmission score').click().run()
    assert not at.exception, at.exception
    np.testing.assert_allclose(at.session_state['prediction']['probability'],
                               predict(model, policy, schema, schema['examples'][1])['probability'])
    submitted = at.session_state['submitted_values'].copy()
    at.sidebar.radio[0].set_value('Prediction explanation').run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == 'Explain submitted prediction').click().run()
    assert not at.exception, at.exception
    assert at.session_state['submitted_values'] == submitted
    table, base, result = at.session_state['explanation']
    reconstructed = 1 / (1 + np.exp(-(base + table.shap_log_odds.sum())))
    np.testing.assert_allclose(reconstructed, result['probability'], atol=1e-7)
    at.sidebar.radio[0].set_value('Model performance').run()
    assert not at.exception, at.exception
    assert any(m.label == 'ROC-AUC' and m.value == '0.671' for m in at.metric)
    at.selectbox[0].set_value('race').run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest() == original_manifest_hash
    manifest = json.loads(frozen_path.read_text())
    for relative, expected_hash in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash, relative
    results = {'status': 'passed', 'checks': ['47 input fields', 'three examples match frozen pipeline',
               'missing values and lab None preserved', 'invalid numerical inputs rejected',
               'UI prediction matches pipeline', 'example loading and resubmission',
               'explanation uses submitted encounter', 'SHAP reconstructs score',
               'performance and subgroup pages render', 'all frozen file hashes unchanged']}
    (ROOT / 'reports/app_verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
