"""Generate human-readable model comparison reports from measured JSON results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


def _metric_table(results: dict[str, Any], split: str) -> list[str]:
    lines = [
        "| Model | "
        + " | ".join(METRIC_LABELS[metric] for metric in METRICS)
        + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for model_name, result in results["models"].items():
        metrics = result[split]
        lines.append(
            f"| {model_name} | "
            + " | ".join(f"{metrics[metric]:.4f}" for metric in METRICS)
            + " |"
        )
    return lines


def generate_model_comparison_report(input_path: Path, output_path: Path) -> None:
    results = json.loads(input_path.read_text(encoding="utf-8"))
    selected_name = results["selected_model"]
    selected = results["models"][selected_name]
    test_metrics = selected["test"]
    validation_f1 = selected["validation"]["f1"]

    lines = [
        "# Model comparison",
        "",
        "## Validation metrics",
        "",
        *_metric_table(results, "validation"),
        "",
        "Both models were compared using the validation split. The test split was",
        "not used for model or hyperparameter selection.",
        "",
        "## Selected model and test metrics",
        "",
        f"Selected model: **{selected_name}**.",
        "",
        f"The model was selected because it had the highest validation F1 ({validation_f1:.4f}); "
        "Accuracy alone was not used as the selection criterion.",
        "",
        f"The following test metrics were calculated **once, only after model selection**, "
        f"for **{selected_name}**:",
        "",
        "| Model | " + " | ".join(METRIC_LABELS[metric] for metric in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
        f"| {selected_name} | "
        + " | ".join(f"{test_metrics[metric]:.4f}" for metric in METRICS)
        + " |",
        "",
        "## Precision/Recall business trade-off",
        "",
        "Recall measures how many customers who actually churned were identified. "
        "Higher recall can help a retention team avoid missing at-risk customers, "
        "but it also creates more false positives and potentially unnecessary outreach.",
        "Precision measures how many customers flagged as churn-risk actually churned. "
        "Higher precision can reduce wasted intervention effort, but a precision-first "
        "threshold may miss customers who would have benefited from retention support.",
        f"This run prioritizes the balance represented by validation F1; "
        f"{selected_name} achieved validation recall {selected['validation']['recall']:.4f} "
        f"and precision {selected['validation']['precision']:.4f}.",
        "",
        "## Confusion matrices",
        "",
    ]
    for model_name in results["models"]:
        lines.append(
            f"- [{model_name} confusion matrix]"
            f"(./{model_name}/{model_name}_confusion_matrix.png)"
        )
    lines.extend(
        [
            "",
            "The machine-readable source for these values is "
            "[model_comparison.json](./model_comparison.json).",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    generate_model_comparison_report(
        Path("reports/model_comparison.json"),
        Path("reports/model_comparison.md"),
    )


if __name__ == "__main__":
    main()
