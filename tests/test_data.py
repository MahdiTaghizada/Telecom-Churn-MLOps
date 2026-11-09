import pandas as pd
import pytest

from telecom_churn.data import validate_dataset


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 2],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "1889.5"],
            "Churn": ["No", "Yes"],
        }
    )


def test_valid_dataset_is_accepted():
    validate_dataset(valid_frame())


def test_missing_column_is_rejected():
    frame = valid_frame().drop(columns=["Contract"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(frame)


def test_invalid_target_is_rejected():
    frame = valid_frame()
    frame.loc[0, "Churn"] = "maybe"
    with pytest.raises(ValueError, match="Churn must contain"):
        validate_dataset(frame)


def test_invalid_numeric_value_is_rejected():
    frame = valid_frame()
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(object)
    frame.loc[0, "MonthlyCharges"] = "not-a-number"
    with pytest.raises(ValueError, match="MonthlyCharges contains"):
        validate_dataset(frame)
