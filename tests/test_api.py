from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from telecom_churn.api import create_app
from tests.test_data import valid_frame


def payload() -> dict[str, object]:
    return valid_frame().drop(columns=["customerID", "Churn"]).iloc[0].to_dict()


def expected_model_version() -> str:
    import json

    metadata = json.loads(
        (Path("models/champion") / "model_metadata.json").read_text(encoding="utf-8")
    )
    return str(metadata["version"])


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(Path("models/champion"))) as test_client:
        yield test_client


def test_health_and_model_info(client: TestClient):
    health = client.get("/health")
    info = client.get("/model-info")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True}
    assert info.status_code == 200
    assert info.json()["model_name"] == "telecom-churn-model"
    assert info.json()["model_version"] == expected_model_version()
    assert info.json()["metadata"]["features"]


def test_predict_returns_model_result(client: TestClient):
    response = client.post("/predict", json=payload())
    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] in {"Yes", "No"}
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_name"] == "telecom-churn-model"
    assert body["model_version"] == expected_model_version()


def test_metrics_are_exposed_and_prediction_increments_counters(client: TestClient):
    before = client.get("/metrics")
    assert before.status_code == 200
    assert "telecom_http_requests_total" in before.text
    client.post("/predict", json=payload())
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "telecom_prediction_requests_total" in metrics.text
    assert "telecom_inference_duration_seconds" in metrics.text
    assert 'route="/predict"' in metrics.text


@pytest.mark.parametrize(
    "update",
    [
        {"Contract": None},
        {"SeniorCitizen": "not-an-integer"},
        {"Contract": 123},
        {"unknown": "field"},
    ],
)
def test_invalid_requests_are_rejected(client: TestClient, update: dict[str, object]):
    request = payload()
    request.update(update)
    response = client.post("/predict", json=request)
    assert response.status_code == 422


def test_missing_feature_is_rejected(client: TestClient):
    request = payload()
    del request["Contract"]
    assert client.post("/predict", json=request).status_code == 422


def test_unseen_category_follows_stage_one_behavior(client: TestClient):
    request = payload()
    request["Contract"] = "Previously unseen contract"
    response = client.post("/predict", json=request)
    assert response.status_code == 200


def test_missing_model_reports_health_and_returns_503(tmp_path: Path):
    with TestClient(create_app(tmp_path / "missing")) as client:
        assert client.get("/health").json() == {
            "status": "degraded",
            "model_loaded": False,
        }
        response = client.post("/predict", json=payload())
        assert response.status_code == 503
        assert client.get("/model-info").status_code == 503
