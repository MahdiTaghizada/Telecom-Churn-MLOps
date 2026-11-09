from pathlib import Path

import mlflow

from telecom_churn.models import build_models
from telecom_churn.preprocessing import prepare_features, prepare_target
from telecom_churn.tracking import configure_tracking, log_training_run
from tests.test_models import training_frame


def test_mlflow_run_works_with_local_sqlite_and_artifacts(tmp_path: Path):
    database = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database.as_posix()}"
    artifact_root = tmp_path / "artifacts"
    configure_tracking(tracking_uri, artifact_root)
    frame = training_frame()
    features = prepare_features(frame)
    target = prepare_target(frame)
    model = build_models(seed=42)["logistic_regression"].fit(features, target)
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    run_id = log_training_run(
        model,
        "test_model",
        features,
        {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "confusion_matrix": [[12, 0], [0, 12]],
        },
        {"random_seed": 42},
        {"test": "true"},
        evaluation_dir,
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics["f1"] == 1.0
    assert run.data.tags["test"] == "true"
