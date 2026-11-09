"""Runtime configuration for the Stage 1 pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_url: str = os.getenv(
        "TELECOM_CHURN_DATA_URL",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    )
    data_path: Path = Path(
        os.getenv("TELECOM_CHURN_DATA_PATH", "data/raw/Telco-Customer-Churn.csv")
    )
    metadata_path: Path = Path(
        os.getenv("TELECOM_CHURN_METADATA_PATH", "data/raw/dataset_metadata.json")
    )
    mlflow_tracking_uri: str = os.getenv(
        "TELECOM_CHURN_MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db"
    )
    mlflow_artifact_root: Path = Path(
        os.getenv("TELECOM_CHURN_MLFLOW_ARTIFACT_ROOT", "mlflow/artifacts")
    )
    random_seed: int = int(os.getenv("TELECOM_CHURN_RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("TELECOM_CHURN_TEST_SIZE", "0.2"))
    validation_size: float = float(os.getenv("TELECOM_CHURN_VALIDATION_SIZE", "0.2"))

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")
