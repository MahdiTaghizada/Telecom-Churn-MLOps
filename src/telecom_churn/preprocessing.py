"""Leakage-safe feature preparation and sklearn pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    result = frame[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(
            result[column].replace(r"^\s*$", pd.NA, regex=True)
        ).astype("float64")
    return result


def prepare_target(frame: pd.DataFrame) -> pd.Series:
    if TARGET not in frame:
        raise ValueError(f"Missing target column: {TARGET}")
    return frame[TARGET].map({"No": 0, "Yes": 1}).astype("int64")


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
