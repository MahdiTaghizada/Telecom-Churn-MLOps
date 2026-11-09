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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from .preprocessing import FEATURES, prepare_features

LOGGER = logging.getLogger(__name__)
DEFAULT_BUNDLE_DIR = Path("models/champion")


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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if bundle.loaded else "degraded",
            model_loaded=bundle.loaded,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: CustomerFeatures) -> PredictionResponse:
        if not bundle.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service is unavailable because the model is not loaded.",
            )
        frame = pd.DataFrame([features.model_dump()], columns=FEATURES)
        prepared = prepare_features(frame)
        prediction = int(bundle.model.predict(prepared)[0])
        probability = float(bundle.model.predict_proba(prepared)[0, 1])
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

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        LOGGER.exception("Unhandled API error.", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    return app


app = create_app()
