# Telecom Churn MLOps

This repository implements Stage 1 of a reproducible telecom churn portfolio
project: data acquisition and validation, leakage-safe preprocessing, model
training and evaluation, MLflow tracking/registry, and offline tests.

FastAPI, Docker, GitLab CI, data drift, and operational monitoring are planned
for later stages and are intentionally not included yet.

## Setup

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run python -m telecom_churn.train
```

The training command downloads the dataset only when the raw file is absent,
records its SHA256 and inspection metadata, trains both models, writes reports
under `reports/`, and exports the selected model under `models/champion/`.
The human-readable comparison can also be regenerated without retraining:

```powershell
uv run python -m telecom_churn.reporting
```

## MLflow

The Stage 1 pipeline uses a local SQLite tracking store and filesystem
artifacts. After training, inspect the experiment with:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Open the displayed local URL, select `telecom-churn-stage-1`, compare the
Logistic Regression and Random Forest runs, and inspect the `evaluation`
artifacts and logged model signatures. The registered model is
`telecom-churn-model`; its active version is selected explicitly through the
`champion` alias.
Every repeated training run creates a new Registry version; the active version
is the one referenced by the `champion` alias. The local Registry database and
MLflow artifact store remain ignored, while the small reports and
`models/champion/` bundle are committed project outputs.

## Measured Stage 1 result

The latest real run selected Logistic Regression by validation F1 (`0.6237`)
over Random Forest (`0.6081`). On the held-out test set, the selected model
achieved Accuracy `0.7374`, Precision `0.5034`, Recall `0.7834`, F1 `0.6130`,
and ROC-AUC `0.8415`. The complete report is generated at
`reports/model_comparison.json`; these values must be regenerated rather than
treated as permanent guarantees.

This stage does not implement the later FastAPI, Docker, GitLab CI, drift, or
Prometheus/Grafana requirements.
