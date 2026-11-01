"""Build a fresh runtime-only ZIP from an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def runtime_files():
    paths = [ROOT / p for p in ['README.md', '.streamlit/config.toml', '.gitattributes',
             'requirements.txt', 'requirements-app.txt', 'models/frozen_manifest.json',
             'models/operating_policy.json', 'models/selected_engineered_model.joblib',
             'src/__init__.py', 'src/features.py', 'deployment/README.md',
             'scripts/verify_deployment.py']]
    paths += [p for p in (ROOT / 'app').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    # Include only aggregate artifacts that the app actually renders/downloads.
    paths += [ROOT / p for p in ['reports/explainability/global_shap_importance.png',
             'reports/final_test/test_metrics.json', 'reports/final_test/test_evaluation.png',
             'reports/final_test/test_subgroup_metrics.csv', 'reports/final_test/patient_bootstrap_intervals.csv',
             'reports/final_test/final_test_report.md']]
    return sorted(set(paths))


def main():
    files = runtime_files()
    for path in files:
        assert path.is_file(), path
    inventory = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert not any('predictions.csv' in name or name.startswith('data/') for name in inventory)
    out = ROOT / 'dist'
    out.mkdir(exist_ok=True)
    archive = out / 'hospital-readmission-app.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(ROOT).as_posix())
        handle.writestr('deployment/runtime_inventory.json', json.dumps(inventory, indent=2))
    print(f'Created {archive}: {len(files)} files, {archive.stat().st_size:,} bytes.')


if __name__ == '__main__':
    main()
