# Telecom Churn MLOps

This repository is a small, reproducible MLOps portfolio project for estimating
telecom customer churn risk. It covers data validation, leakage-safe training,
MLflow tracking and Registry, a FastAPI service, Docker packaging, GitLab CI
tests, data-drift reports, and Prometheus-format application metrics.

The model is decision support, not an automatic business decision.

## Business problem and data

Telecom churn causes lost revenue and increases customer-acquisition cost. The
API estimates whether a customer resembles the churn class so a human team can
investigate retention options.

The source is IBM's public Telco Customer Churn sample dataset. It contains
synthetic company data, not internal operator data or real customer records. See
[docs/DATASET.md](docs/DATASET.md) for the source, license note, checksum, and
limitations. The target is `Churn` (`Yes`/`No`); `customerID` is an identifier
and is excluded. Features include `tenure`, `MonthlyCharges`, `TotalCharges`,
service attributes, contract, payment method, and customer profile fields.

## Reproducible setup

### Prerequisites

- Python 3.13
- `uv`
- Docker Desktop (required only for container verification)
- Internet access is required only for the first dataset download; later runs
  can use the existing raw CSV

PowerShell on a new computer:

```powershell
git clone <repository-url>
Set-Location Telecom-Churn-MLOps
uv sync --frozen
uv run python -m telecom_churn.train
uv run pytest -q
uv run ruff check .
uv run python -m telecom_churn.drift
```

`uv sync` creates the project environment from `pyproject.toml` and `uv.lock`.
Training downloads the CSV only when `data/raw/Telco-Customer-Churn.csv` is
absent, validates it, trains both models, updates MLflow, writes model
comparison reports, and exports `models/champion/`.

## Measured model result

The committed report at [reports/model_comparison.json](reports/model_comparison.json)
records the latest real training run:

| Model | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (selected) | 0.6237 | 0.7374 | 0.5034 | 0.7834 | 0.6130 | 0.8415 |
| Random Forest | 0.6081 | not selected | not selected | not selected | not selected | not selected |

Selection used validation F1 rather than Accuracy alone. The held-out test
metrics were calculated only after selection. Re-running training may create a
new Registry version and new measured values.

## MLflow and Registry

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

The experiment is `telecom-churn-stage-1`. The recorded Registry model is
`telecom-churn-model`, version `6`, with the `champion` alias in the project
author's local training run. The `mlflow/` directory is local-only and is not
stored in Git, so the MLflow UI is empty before training on a new checkout.
Training then creates a new local tracking database and the first Registry
version starts at `1`; repeated training runs increment that local version.
The committed `models/champion/` bundle is separate from that database and is
the artifact used by the API. MLflow stores parameters, validation metrics,
model signatures, and evaluation artifacts.

## API

```powershell
uv run uvicorn telecom_churn.api:app --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `POST /predict`, `GET /model-info`, and
`GET /metrics`. Swagger is available at `http://127.0.0.1:8000/docs`.

Example request:

```json
{"SeniorCitizen":0,"tenure":1,"MonthlyCharges":29.85,"TotalCharges":29.85,
"gender":"Female","Partner":"Yes","Dependents":"No","PhoneService":"No",
"MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No",
"OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No",
"StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
"PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}
```

The response contains `prediction`, `churn_probability`, `model_name`, and
`model_version`. Categorical unknown values are accepted by the pipeline's
`OneHotEncoder(handle_unknown="ignore")`; malformed or extra fields are
rejected.

Real response from the local API:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8007603116595001,
  "model_name": "telecom-churn-model",
  "model_version": "6"
}
```

To send the request from a JSON file with `curl.exe`:

```powershell
Set-Content -Path api-request.json -Value @'
{"SeniorCitizen":0,"tenure":1,"MonthlyCharges":29.85,"TotalCharges":29.85,"gender":"Female","Partner":"Yes","Dependents":"No","PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}
'@ -NoNewline
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@api-request.json"
Remove-Item api-request.json
```

`/metrics` exposes total HTTP requests, request duration, HTTP errors,
prediction count, and inference duration. Labels are limited to route templates
and status codes; customer values are never labels.

## Docker and CI

```powershell
docker build -t telecom-churn-api .
docker run -d --name churn-api-final -p 8013:8000 telecom-churn-api
curl.exe http://127.0.0.1:8013/health
curl.exe http://127.0.0.1:8013/model-info
curl.exe -X POST http://127.0.0.1:8013/predict `
  -H "Content-Type: application/json" `
  --data-binary "@api-request.json"
docker stop churn-api-final
docker rm churn-api-final
```

The Docker image and GitLab CI configuration are present. GitLab CI has not yet
been run; push the repository to GitLab to execute the test pipeline. It does
not publish or deploy an image. Until that push, no GitLab pipeline status is
claimed.

## Data drift

```powershell
uv run python -m telecom_churn.drift
```

This creates [reports/drift/SYNTHETIC_drift_report.html](reports/drift/SYNTHETIC_drift_report.html)
and JSON summaries, plus a control report. Reference data is the feature-only
training split. Control current data is sampled from the same distribution;
synthetic current data deliberately changes `MonthlyCharges`, `tenure`,
`Contract`, and `PaymentMethod`. Numeric features use two-sample KS and
categorical features use chi-square with `p < 0.05`.

Data drift is not concept drift and does not, by itself, prove worse model
quality. New labeled data requires a separate evaluation.

## Verification status

| Area | Status |
|---|---|
| Data, preprocessing, two-model training | Complete |
| MLflow tracking and Registry | Complete |
| FastAPI prediction and validation | Complete |
| Prometheus `/metrics` MVP | Complete |
| Control and SYNTHETIC drift HTML/JSON | Complete |
| Docker verification in this environment | Verified locally (build, run, /health, /predict, /metrics) |
| GitLab pipeline result | Pending push to GitLab |
| Grafana dashboard, cloud deployment, auto-retraining | Future work |

## Tests and screenshots

Run `uv run pytest -q` for the local suite. The suite covers data validation,
preprocessing, models, API behavior, tracking, reporting, drift control versus
synthetic data, and `/metrics`.

The current real run is `24 passed, 1 warning`. The warning is:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```

It comes from the installed FastAPI/Starlette test-client compatibility layer;
the repository uses the pinned `httpx==0.28.1` dependency and does not silently
change the test client or weaken the tests. It is recorded as a known warning.

Screenshots are intentionally not fabricated. Capture these manually under
`docs/images/`:

- `mlflow-runs.png` — MLflow run comparison
- `model-registry.png` — Registry model and `champion` alias
- `swagger-docs.png` — FastAPI `/docs`
- `gitlab-pipeline.png` — GitLab test pipeline
- `synthetic-drift-report.png` — rendered SYNTHETIC HTML report

## Limitations and future work

The dataset is synthetic and may not represent a production operator. Bias and
fairness across groups were not comprehensively evaluated. Drift metrics do
not measure model quality. The MVP has no live labeled feedback loop, automatic
retraining, Airflow, Grafana dashboard, AWS deployment, or IAM/OIDC setup.
Potential next steps are threshold and calibration analysis, fairness
evaluation, alerting/dashboard integration, and a separately governed
retraining workflow.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and
[docs/CODE_WALKTHROUGH.az.md](docs/CODE_WALKTHROUGH.az.md).
