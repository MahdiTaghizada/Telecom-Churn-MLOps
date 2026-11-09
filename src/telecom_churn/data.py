"""Dataset acquisition, inspection, and schema validation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"
EXPECTED_TARGETS = frozenset({"Yes", "No"})
REQUIRED_FEATURES = frozenset(
    {
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
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
        "MonthlyCharges",
        "TotalCharges",
    }
)
NUMERIC_COLUMNS = frozenset({"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    try:
        with urlopen(url, timeout=30) as response:
            destination.write_bytes(response.read())
    except (OSError, URLError) as exc:
        raise RuntimeError(f"Unable to download dataset from {url}: {exc}") from exc
    if destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded dataset is empty: {destination}")
    return destination


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read dataset CSV {path}: {exc}") from exc
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Dataset must not be empty.")
    required = REQUIRED_FEATURES | {TARGET, ID_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    targets = set(frame[TARGET].dropna().unique())
    if not targets or not targets.issubset(EXPECTED_TARGETS):
        raise ValueError(
            f"{TARGET} must contain only {sorted(EXPECTED_TARGETS)}; got {sorted(targets)}"
        )
    if frame[TARGET].isna().any():
        raise ValueError(f"{TARGET} must not contain missing values.")
    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} must not contain missing values.")
    for column in NUMERIC_COLUMNS:
        values = frame[column].astype("string").str.strip()
        normalized = values.replace("", pd.NA)
        invalid = pd.to_numeric(normalized, errors="coerce").isna() & normalized.notna()
        if invalid.any():
            raise ValueError(f"{column} contains non-numeric values.")


def inspect_dataset(frame: pd.DataFrame) -> dict[str, object]:
    validate_dataset(frame)
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "categorical_unique_values": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in frame.select_dtypes(include=["object", "string"]).columns
        },
        "target_balance": frame[TARGET].value_counts().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def write_metadata(
    path: Path, source_url: str, dataset_path: Path, inspection: dict[str, object]
) -> None:
    payload = {
        "source_url": source_url,
        "license_note": "IBM sample dataset; verify source terms before redistribution.",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "sha256": sha256_file(dataset_path),
        "inspection": inspection,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
