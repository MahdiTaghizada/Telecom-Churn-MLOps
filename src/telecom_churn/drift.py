"""Statistical data-drift reports for the training feature distribution."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from .config import Settings
from .data import download_dataset, load_dataset
from .preprocessing import FEATURES, NUMERIC_FEATURES, prepare_features
from .splitting import stratified_split

DEFAULT_OUTPUT_DIR = Path("reports/drift")
P_VALUE_THRESHOLD = 0.05


def compare_distributions(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, Any]:
    """Compare numeric features with KS and categorical features with chi-square."""

    missing = sorted(
        (set(FEATURES) - set(reference.columns)) | (set(FEATURES) - set(current.columns))
    )
    if missing:
        raise ValueError(f"Drift comparison is missing required feature columns: {missing}")
    if reference.empty or current.empty:
        raise ValueError("Reference and current data must not be empty.")

    feature_results: dict[str, dict[str, Any]] = {}
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            reference_values = pd.to_numeric(reference[feature], errors="coerce").dropna()
            current_values = pd.to_numeric(current[feature], errors="coerce").dropna()
            statistic, p_value = ks_2samp(reference_values, current_values)
            test_name = "Kolmogorov-Smirnov"
        else:
            categories = sorted(
                set(reference[feature].astype(str)) | set(current[feature].astype(str))
            )
            table = np.array(
                [
                    reference[feature].astype(str).value_counts().reindex(categories, fill_value=0),
                    current[feature].astype(str).value_counts().reindex(categories, fill_value=0),
                ]
            )
            statistic, p_value, _, _ = chi2_contingency(table)
            test_name = "chi-square"
        feature_results[feature] = {
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < P_VALUE_THRESHOLD),
        }

    drifted = [feature for feature, result in feature_results.items() if result["drift_detected"]]
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "p_value_threshold": P_VALUE_THRESHOLD,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "feature_results": feature_results,
    }


def make_current_data(reference: pd.DataFrame, scenario: str, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic control or explicitly synthetic drift sample."""

    if scenario not in {"control", "synthetic"}:
        raise ValueError("scenario must be 'control' or 'synthetic'.")
    current = reference.sample(
        n=len(reference), replace=True, random_state=seed
    ).reset_index(drop=True)
    if scenario == "synthetic":
        rng = np.random.default_rng(seed)
        current["MonthlyCharges"] = (
            pd.to_numeric(current["MonthlyCharges"]) * 1.8 + rng.normal(0, 2, len(current))
        ).clip(lower=0)
        current["tenure"] = (pd.to_numeric(current["tenure"]) // 2).clip(lower=0)
        current["Contract"] = "Month-to-month"
        current["PaymentMethod"] = "Electronic check"
    return current


def render_html(summary: dict[str, Any], scenario: str, output_path: Path) -> None:
    """Write a self-contained human-readable drift report."""

    rows = "\n".join(
        f"<tr><td>{html.escape(feature)}</td><td>{html.escape(result['test'])}</td>"
        f"<td>{result['statistic']:.6f}</td><td>{result['p_value']:.6g}</td>"
        f"<td>{'Yes' if result['drift_detected'] else 'No'}</td></tr>"
        for feature, result in summary["feature_results"].items()
    )
    title = (
        f"SYNTHETIC data drift report ({scenario})"
        if scenario == "synthetic"
        else "Control data drift report"
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem;text-align:left}}
.drift{{font-weight:bold;color:#a00}}</style></head><body>
<h1>{title}</h1>
<p>Reference data is the feature-only training split. Current data is a sampled
control distribution or an explicitly synthetic scenario; it is not live customer data.</p>
<p>Overall drift: <span class="drift">
{'Detected' if summary['drift_detected'] else 'Not detected'}</span>.
Reference rows: {summary['reference_rows']}; current rows: {summary['current_rows']};
p-value threshold: {summary['p_value_threshold']}.</p>
<p>Numeric features use the two-sample Kolmogorov-Smirnov test. Categorical
features use a chi-square test of count distributions. Data drift does not prove
model quality degradation; labeled evaluation is required to assess that.</p>
<table><thead><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Drift</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def build_reference(settings: Settings) -> pd.DataFrame:
    dataset_path = download_dataset(settings.data_url, settings.data_path)
    frame = load_dataset(dataset_path)
    features = prepare_features(frame)
    target = frame["Churn"].map({"No": 0, "Yes": 1}).astype("int64")
    train_frame, _, _, _, _, _ = stratified_split(
        features, target, settings.test_size, settings.validation_size, settings.random_seed
    )
    return train_frame


def generate_reports(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, dict[str, Any]]:
    settings = Settings()
    reference = build_reference(settings)
    reports = {}
    for scenario in ("control", "synthetic"):
        current = make_current_data(reference, scenario, settings.random_seed)
        summary = compare_distributions(reference, current)
        summary.update(
            {
                "scenario": scenario,
                "synthetic": scenario == "synthetic",
                "reference_description": "Feature-only training split.",
                "current_description": (
                    "Sampled from the same reference distribution."
                    if scenario == "control"
                    else "Synthetic drift created for testing; not live data."
                ),
            }
        )
        stem = "control" if scenario == "control" else "SYNTHETIC_drift"
        (output_dir / f"{stem}_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{stem}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        render_html(summary, scenario, output_dir / f"{stem}_report.html")
        reports[scenario] = summary
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    reports = generate_reports(args.output_dir)
    for scenario, summary in reports.items():
        print(
            f"{scenario}: drift={summary['drift_detected']}; "
            f"features={summary['drifted_features']}"
        )


if __name__ == "__main__":
    main()
