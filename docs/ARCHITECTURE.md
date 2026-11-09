# Architecture

```mermaid
flowchart LR
    raw[IBM Telco CSV] --> validate[Validation and inspection]
    validate --> split[Stratified train/validation/test split]
    split --> pipeline[Leakage-safe sklearn pipelines]
    pipeline --> train[Logistic Regression and Random Forest]
    train --> mlflow[MLflow tracking and Registry]
    mlflow --> bundle[Exported champion bundle]
    bundle --> api[FastAPI prediction API]
    api --> metrics[Prometheus-format /metrics]
    split --> reference[Training feature reference]
    reference --> drift[Control and SYNTHETIC drift comparison]
    drift --> reports[HTML and JSON reports]
```

Training fits preprocessing only on the training partition. The selected
pipeline is registered in MLflow and exported locally so the API does not need
an MLflow server at inference time. The drift utility uses the same
deterministic split to reconstruct the feature-only training reference.

The MVP intentionally does not include Airflow, automatic retraining, cloud
deployment, or a Grafana dashboard.
