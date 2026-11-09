import pandas as pd
import pytest

from telecom_churn.drift import compare_distributions, make_current_data
from tests.test_data import valid_frame


def feature_frame() -> pd.DataFrame:
    return valid_frame().drop(columns=["customerID", "Churn"])


def test_same_distribution_has_no_drift():
    reference = pd.concat([feature_frame()] * 50, ignore_index=True)
    summary = compare_distributions(reference, make_current_data(reference, "control", seed=42))
    assert summary["drift_detected"] is False
    assert summary["drifted_features"] == []


def test_synthetic_distribution_has_drift():
    reference = pd.concat([feature_frame()] * 50, ignore_index=True)
    summary = compare_distributions(reference, make_current_data(reference, "synthetic", seed=42))
    assert summary["drift_detected"] is True
    assert {"MonthlyCharges", "tenure", "Contract", "PaymentMethod"}.issubset(
        summary["drifted_features"]
    )


def test_missing_feature_is_rejected():
    reference = feature_frame()
    with pytest.raises(ValueError, match="missing required feature columns"):
        compare_distributions(reference.drop(columns=["Contract"]), reference)
