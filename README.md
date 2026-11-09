# Telecom Churn MLOps

This repository implements Stage 1 and Stage 2 of a reproducible telecom churn
portfolio project: data acquisition and validation, leakage-safe preprocessing,
model training and evaluation, MLflow tracking/registry, an offline FastAPI
prediction service, Docker packaging, and GitLab CI tests.

Data drift and operational monitoring are planned for Stage 3 and are
intentionally not included yet.

## Setup

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run python -m telecom_churn.train
```

## API

The API loads `models/champion/model` once during application startup. Runtime
prediction does not require an MLflow server. Start it locally with:

```powershell
uv run uvicorn telecom_churn.api:app --host 127.0.0.1 --port 8000
```

`GET /health`, `POST /predict`, and `GET /model-info` are available, and
Swagger UI is at `http://127.0.0.1:8000/docs`. The complete request schema is
the 19-feature schema listed in `src/telecom_churn/preprocessing.py`.
Categorical values must be strings. Previously unseen string categories are
accepted consistently with Stage 1's `OneHotEncoder(handle_unknown="ignore")`.
The probability is a model risk estimate, not a guaranteed churn outcome.

Example request and response from the local API:

```powershell
curl.exe http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"SeniorCitizen":0,"tenure":1,"MonthlyCharges":29.85,"TotalCharges":29.85,"gender":"Female","Partner":"Yes","Dependents":"No","PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}'
```

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8007603116595001,
  "model_name": "telecom-churn-model",
  "model_version": "4"
}
```

## Docker

Build and run the API image locally:

```powershell
docker build -t telecom-churn-mlops:stage2 .
docker run -d --name churn-api-check -p 8000:8000 telecom-churn-mlops:stage2
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model-info
```

For a prediction, write the JSON body to a file and send it without inline
PowerShell JSON quoting:

```powershell
$json = @'
{"SeniorCitizen":0,"tenure":1,"MonthlyCharges":29.85,"TotalCharges":29.85,"gender":"Female","Partner":"Yes","Dependents":"No","PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}
'@
Set-Content -Path api-smoke-request.json -Value $json -NoNewline
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@api-smoke-request.json"
Remove-Item -LiteralPath api-smoke-request.json
docker rm -f churn-api-check
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

This stage does not implement the later data drift or Prometheus/Grafana
requirements. GitLab CI currently runs lint and offline tests only; it does not
build, publish, or deploy a Docker image.
