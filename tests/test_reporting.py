import json
from pathlib import Path

from telecom_churn.reporting import generate_model_comparison_report


def test_report_is_generated_from_json_without_training(tmp_path: Path):
    source = {
        "selected_model": "logistic_regression",
        "models": {
            "logistic_regression": {
                "validation": {
                    "accuracy": 0.7,
                    "precision": 0.6,
                    "recall": 0.8,
                    "f1": 0.6857,
                    "roc_auc": 0.82,
                },
                "test": {
                    "accuracy": 0.69,
                    "precision": 0.59,
                    "recall": 0.79,
                    "f1": 0.6772,
                    "roc_auc": 0.81,
                },
            },
            "random_forest": {
                "validation": {
                    "accuracy": 0.8,
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1": 0.6462,
                    "roc_auc": 0.83,
                }
            },
        },
    }
    source_path = tmp_path / "model_comparison.json"
    output_path = tmp_path / "model_comparison.md"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    generate_model_comparison_report(source_path, output_path)

    report = output_path.read_text(encoding="utf-8")
    assert "Validation metrics" in report
    assert "only after model selection" in report
    assert "Precision/Recall business trade-off" in report
    assert "./random_forest/random_forest_confusion_matrix.png" in report
