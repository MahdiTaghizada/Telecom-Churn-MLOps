"""Required baseline and alternative classification models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def build_searches(seed: int) -> dict[str, GridSearchCV]:
    models = build_models(seed)
    return {
        "logistic_regression": GridSearchCV(
            models["logistic_regression"],
            param_grid={"classifier__C": [0.5, 1.0]},
            scoring="f1",
            cv=3,
            n_jobs=1,
            refit=True,
        ),
        "random_forest": GridSearchCV(
            models["random_forest"],
            param_grid={"classifier__max_depth": [None, 12]},
            scoring="f1",
            cv=3,
            n_jobs=1,
            refit=True,
        ),
    }
