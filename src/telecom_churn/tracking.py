"""MLflow tracking, registry, and selected-model export helpers."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow.sklearn
from mlflow.models import infer_signature

import mlflow
from mlflow import MlflowClient

REGISTERED_MODEL_NAME = "telecom-churn-model"
ALIAS = "champion"


def configure_tracking(tracking_uri: str, artifact_root: Path) -> MlflowClient:
    artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "telecom-churn-stage-1"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = artifact_root.resolve().as_uri()
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return client


def log_training_run(
    model,
    model_name: str,
    train_features,
    validation_metrics: dict[str, object],
    params: dict[str, object],
    tags: dict[str, str],
    artifact_dir: Path,
) -> str:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {
                key: value
                for key, value in validation_metrics.items()
                if key in {"accuracy", "precision", "recall", "f1", "roc_auc"}
            }
        )
        mlflow.set_tags(tags)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        signature = infer_signature(
            train_features.head(5), model.predict(train_features.head(5))
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=train_features.head(1),
            metadata={"model_name": model_name},
        )
        return run.info.run_id


def register_selected_model(
    client: MlflowClient,
    run_id: str,
    metrics: dict[str, object],
    reason: str,
) -> tuple[str, str]:
    model_version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    version = str(model_version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "selection_reason", reason)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "validation_f1", str(metrics["f1"])
    )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, version)
    return REGISTERED_MODEL_NAME, version


def export_model_bundle(
    model,
    output_dir: Path,
    model_name: str,
    version: str,
    run_id: str,
    metrics: dict[str, object],
    feature_names: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model"
    if model_path.exists():
        shutil.rmtree(model_path)
    mlflow.sklearn.save_model(model, path=str(model_path))
    metadata = {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "metrics": metrics,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "features": feature_names,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output_dir
