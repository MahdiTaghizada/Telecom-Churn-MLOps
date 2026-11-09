from telecom_churn.preprocessing import prepare_features, prepare_target
from tests.test_data import valid_frame


def test_target_and_feature_preparation():
    frame = valid_frame()
    features = prepare_features(frame)
    target = prepare_target(frame)
    assert "customerID" not in features
    assert features.shape == (2, 19)
    assert target.tolist() == [0, 1]


def test_numeric_blank_is_normalized_to_missing():
    frame = valid_frame()
    frame.loc[0, "TotalCharges"] = " "
    features = prepare_features(frame)
    assert features.loc[0, "TotalCharges"] != features.loc[0, "TotalCharges"]


def test_unseen_category_does_not_break_fitted_pipeline():
    from telecom_churn.models import build_models

    frame = valid_frame()
    features = prepare_features(frame)
    target = prepare_target(frame)
    model = build_models(seed=42)["logistic_regression"].fit(features, target)
    unseen = frame.drop(columns=["Churn"]).copy()
    unseen.loc[0, "Contract"] = "New contract category"
    assert model.predict(prepare_features(unseen)).shape == (2,)
