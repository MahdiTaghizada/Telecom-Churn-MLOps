"""FastAPI application for serving the exported churn model bundle."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .preprocessing import FEATURES, prepare_features

LOGGER = logging.getLogger(__name__)
DEFAULT_BUNDLE_DIR = Path("models/champion")
HTTP_REQUESTS = Counter(
    "telecom_http_requests_total",
    "Total HTTP requests.",
    labelnames=("route", "status"),
)
HTTP_ERRORS = Counter(
    "telecom_http_errors_total",
    "Total HTTP responses with a 4xx or 5xx status.",
    labelnames=("route", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "telecom_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("route",),
)
PREDICTION_REQUESTS = Counter(
    "telecom_prediction_requests_total",
    "Total prediction requests.",
)
INFERENCE_DURATION = Histogram(
    "telecom_inference_duration_seconds",
    "Model inference duration in seconds.",
)


class CustomerFeatures(BaseModel):
    """Validated model features accepted by the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    SeniorCitizen: Annotated[StrictInt, Field(ge=0, le=1)]
    tenure: Annotated[StrictInt, Field(ge=0)]
    MonthlyCharges: Annotated[float, Field(ge=0)]
    TotalCharges: Annotated[float, Field(ge=0)]
    gender: StrictStr
    Partner: StrictStr
    Dependents: StrictStr
    PhoneService: StrictStr
    MultipleLines: StrictStr
    InternetService: StrictStr
    OnlineSecurity: StrictStr
    OnlineBackup: StrictStr
    DeviceProtection: StrictStr
    TechSupport: StrictStr
    StreamingTV: StrictStr
    StreamingMovies: StrictStr
    Contract: StrictStr
    PaperlessBilling: StrictStr
    PaymentMethod: StrictStr


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: Annotated[float, Field(ge=0, le=1)]
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    metadata: dict[str, Any]


class ModelBundle:
    """In-memory model and metadata loaded for the application lifetime."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        metadata_path = self.bundle_dir / "model_metadata.json"
        model_path = self.bundle_dir / "model"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = mlflow.sklearn.load_model(str(model_path))

    @property
    def loaded(self) -> bool:
        return self.model is not None


def create_app(bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> FastAPI:
    """Create an application whose model is loaded once during lifespan startup."""

    bundle = ModelBundle(bundle_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            bundle.load()
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            mlflow.exceptions.MlflowException,
        ):
            LOGGER.exception("Model bundle could not be loaded.")
            bundle.model = None
            bundle.metadata = {}
        yield

    app = FastAPI(
        title="Telecom Churn Prediction API",
        version="0.1.0",
        description="Risk scoring API using the exported Stage 1 churn pipeline.",
        lifespan=lifespan,
    )
    app.state.bundle = bundle

    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        import time

        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        status_code = str(response.status_code)
        HTTP_REQUESTS.labels(route=route_name, status=status_code).inc()
        if response.status_code >= 400:
            HTTP_ERRORS.labels(route=route_name, status=status_code).inc()
        HTTP_REQUEST_DURATION.labels(route=route_name).observe(time.perf_counter() - started)
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if bundle.loaded else "degraded",
            model_loaded=bundle.loaded,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: CustomerFeatures) -> PredictionResponse:
        import time

        if not bundle.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is unavailable because the model is not loaded.",
            )
        frame = pd.DataFrame([features.model_dump()], columns=FEATURES)
        prepared = prepare_features(frame)
        PREDICTION_REQUESTS.inc()
        inference_started = time.perf_counter()
        prediction = int(bundle.model.predict(prepared)[0])
        probability = float(bundle.model.predict_proba(prepared)[0, 1])
        INFERENCE_DURATION.observe(time.perf_counter() - inference_started)
        return PredictionResponse(
            prediction="Yes" if prediction == 1 else "No",
            churn_probability=probability,
            model_name=str(bundle.metadata["model_name"]),
            model_version=str(bundle.metadata["version"]),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        if not bundle.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model information is unavailable because the model is not loaded.",
            )
        return ModelInfoResponse(
            model_name=str(bundle.metadata["model_name"]),
            model_version=str(bundle.metadata["version"]),
            metadata=bundle.metadata,
        )

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        LOGGER.exception("Unhandled API error.", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()
