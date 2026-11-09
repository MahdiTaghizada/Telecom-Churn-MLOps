"""Command-line entry point for Stage 1 training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Settings
from .data import download_dataset, inspect_dataset, load_dataset, write_metadata
from .evaluation import evaluate_model, write_evaluation_artifacts
from .models import build_searches
from .preprocessing import FEATURES, prepare_features, prepare_target
from .reporting import generate_model_comparison_report
from .splitting import stratified_split
from .tracking import (
    configure_tracking,
    export_model_bundle,
    log_training_run,
    register_selected_model,
)


def main() -> None:
    settings = Settings()
    settings.validate()
    dataset_path = download_dataset(settings.data_url, settings.data_path)
    frame = load_dataset(dataset_path)
    inspection = inspect_dataset(frame)
    write_metadata(settings.metadata_path, settings.data_url, dataset_path, inspection)
    features = prepare_features(frame)
    target = prepare_target(frame)
    (
        train_frame,
        validation_frame,
        test_frame,
        train_target,
        validation_target,
        test_target,
    ) = stratified_split(
        features,
        target,
        settings.test_size,
        settings.validation_size,
        settings.random_seed,
    )
    client = configure_tracking(settings.mlflow_tracking_uri, settings.mlflow_artifact_root)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    runs: dict[str, str] = {}
    for model_name, search in build_searches(settings.random_seed).items():
        search.fit(train_frame, train_target)
        metrics = evaluate_model(search.best_estimator_, validation_frame, validation_target)
        model_report_dir = report_dir / model_name
        write_evaluation_artifacts(metrics, model_report_dir, model_name)
        runs[model_name] = log_training_run(
            search.best_estimator_,
            model_name,
            train_frame,
            metrics,
            {
                "random_seed": settings.random_seed,
                "test_size": settings.test_size,
                "validation_size": settings.validation_size,
                "best_params": search.best_params_,
            },
            {
                "dataset_sha256": json.loads(
                    Path(settings.metadata_path).read_text(encoding="utf-8")
                )["sha256"]
            },
            model_report_dir,
        )
        results[model_name] = {"validation": metrics, "best_params": search.best_params_}
    selected_name = max(results, key=lambda name: results[name]["validation"]["f1"])
    selected_search = build_searches(settings.random_seed)[selected_name]
    selected_search.estimator.set_params(**results[selected_name]["best_params"])
    selected_search.estimator.fit(
        pd.concat([train_frame, validation_frame]),
        pd.concat([train_target, validation_target]),
    )
    selected_model = selected_search.estimator
    test_metrics = evaluate_model(selected_model, test_frame, test_target)
    results[selected_name]["test"] = test_metrics
    model_name, version = register_selected_model(
        client,
        runs[selected_name],
        results[selected_name]["validation"],
        (
            "Highest validation F1 among required models: "
            f"{results[selected_name]['validation']['f1']:.4f}."
        ),
    )
    export_model_bundle(
        selected_model,
        Path("models") / "champion",
        model_name,
        version,
        runs[selected_name],
        test_metrics,
        FEATURES,
    )
    (report_dir / "model_comparison.json").write_text(
        json.dumps(
            {
                "models": results,
                "selected_model": selected_name,
                "registry": {"name": model_name, "version": version, "run_id": runs[selected_name]},
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_model_comparison_report(
        report_dir / "model_comparison.json",
        report_dir / "model_comparison.md",
    )
    print(
        f"Validated {inspection['shape']['rows']} rows and "
        f"{inspection['shape']['columns']} columns."
    )
    print(f"Selected model: {selected_name}; registry version: {version}")


if __name__ == "__main__":
    main()
