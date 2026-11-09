from pathlib import Path

import mlflow
import pandas as pd

from telecom_churn.evaluation import evaluate_model
from telecom_churn.models import build_models
from telecom_churn.preprocessing import prepare_features, prepare_target
from tests.test_data import valid_frame


def training_frame() -> pd.DataFrame:
    base = valid_frame()
    copies = []
    for index in range(12):
        copy = base.copy()
        copy["customerID"] = [f"customer-{index}-a", f"customer-{index}-b"]
        copies.append(copy)
    return pd.concat(copies, ignore_index=True)


def test_both_models_train_predict_and_score():
    frame = training_frame()
    features = prepare_features(frame)
    target = prepare_target(frame)
    for model in build_models(seed=42).values():
        model.fit(features, target)
        probabilities = model.predict_proba(features)[:, 1]
        metrics = evaluate_model(model, features, target)
        assert len(model.predict(features)) == len(frame)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_model_artifact_can_be_saved_and_loaded(tmp_path: Path):
    frame = training_frame()
    features = prepare_features(frame)
    target = prepare_target(frame)
    model = build_models(seed=42)["logistic_regression"].fit(features, target)
    artifact_path = tmp_path / "model"
    mlflow.sklearn.save_model(model, str(artifact_path))
    loaded = mlflow.sklearn.load_model(str(artifact_path))
    assert loaded.predict(features).shape == target.shape
