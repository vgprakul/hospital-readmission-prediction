"""Verify a runtime-only checkout: frozen inference, SHAP, and all UI pages."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from streamlit.testing.v1 import AppTest
from app.runtime import load_artifacts, predict


def main():
    model, policy, schema = load_artifacts()
    test_metrics = json.loads((ROOT / 'reports/final_test/test_metrics.json').read_text())
    manifest_hash = hashlib.sha256((ROOT / 'models/frozen_manifest.json').read_bytes()).hexdigest()
    assert test_metrics['manifest_sha256'] == manifest_hash
    at = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=240).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == 'Predict readmission score').click().run()
    assert not at.exception, at.exception
    np.testing.assert_allclose(at.session_state['prediction']['probability'],
                               predict(model, policy, schema, schema['examples'][0])['probability'])
    at.sidebar.radio[0].set_value('Prediction explanation').run()
    next(b for b in at.button if b.label == 'Explain submitted prediction').click().run()
    assert not at.exception, at.exception
    table, base, result = at.session_state['explanation']
    np.testing.assert_allclose(1 / (1 + np.exp(-(base + table.shap_log_odds.sum()))), result['probability'], atol=1e-7)
    at.sidebar.radio[0].set_value('Model performance').run()
    assert not at.exception, at.exception
    assert any(m.label == 'ROC-AUC' and m.value == '0.671' for m in at.metric)
    at.selectbox[0].set_value('race').run()
    assert not at.exception, at.exception
    print(json.dumps({'status': 'passed', 'python': sys.version.split()[0],
                      'runtime_only': not (ROOT / 'data/raw/diabetic_data.csv').exists(),
                      'checks': ['frozen artifacts', 'prediction consistency', 'SHAP reconstruction',
                                 'all three app pages', 'subgroup interaction']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
