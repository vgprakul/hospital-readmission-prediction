# Deployment runbook

The app serves a frozen educational model, not a clinical tool. It needs no credentials or raw dataset at runtime.

## Streamlit Community Cloud

1. Publish this repository to GitHub, including the small `models/selected_engineered_model.joblib` file. Raw datasets and encounter-level prediction files must remain excluded.
2. Sign in at https://share.streamlit.io/ and choose **Create app**.
3. Select the repository and branch; set the entrypoint to **app/app.py**.
4. In advanced settings, select **Python 3.13** to match the model's tested environment. No secrets are required.
5. Deploy and check a training example, the SHAP explanation page, and the performance/subgroup page.

`app/requirements.txt` is the pinned deployment dependency file beside the entrypoint; it takes precedence over the root training requirements. The original frozen training requirements and manifest remain unchanged. Model-path separators are normalized by the app for Linux. `.gitattributes` protects bytes covered by the frozen hashes from line-ending conversion.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies.

## Clean local verification

```text
python -m venv .venv-deploy
.venv-deploy/Scripts/python -m pip install -r app/requirements.txt
.venv-deploy/Scripts/python scripts/verify_deployment.py
```

On Linux use `.venv-deploy/bin/python`. Run `python scripts/package_deployment.py` to create a runtime-only ZIP containing explicit allowlisted files and a SHA-256 inventory. It excludes raw data, notebooks, training example IDs, encounter-level predictions, unused models, secrets, Git metadata, and virtual environments. It includes three ID-free training example input vectors and aggregate evaluation/explanation artifacts needed by the app.

## Publication status

Repository and hosting publication require an authenticated GitHub connection and Streamlit account. Local packaging or successful tests do not mean the app has been deployed. Record the actual repository and app URLs only after successful publication.
