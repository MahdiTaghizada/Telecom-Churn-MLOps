"""Classification metrics and human/machine-readable reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, features, target) -> dict[str, object]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        "classification_report": classification_report(
            target, predictions, output_dict=True, zero_division=0
        ),
    }


def write_evaluation_artifacts(
    metrics: dict[str, object], output_dir: Path, model_name: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    matrix = metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{model_name} confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close(figure)
