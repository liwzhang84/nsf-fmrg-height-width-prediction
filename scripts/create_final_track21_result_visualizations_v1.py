#!/usr/bin/env python3
"""Create publication figures from immutable Track-21 final-test CSV outputs."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nsf_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/models/final_track21_height_width_metrics.csv"
PREDICTIONS = ROOT / "outputs/models/final_track21_height_width_predictions.csv"
FIGURE_DIR = ROOT / "outputs/figures/final_track21_height_width_v2"
SUMMARY = ROOT / "outputs/reports/final_track21_visualization_summary_v1.md"
VALIDATION = ROOT / "outputs/reports/final_track21_visualization_validation_v1.json"
ORIGINAL_FIGURE = FIGURE_DIR / "final_track21_predictions_and_errors.png"
LOCK = ROOT / "configs/final_height_width_model_lock_v1.yaml"
AMENDMENT = ROOT / "configs/final_test_sem_source_compatibility_amendment_v1.yaml"
EXPECTED_LOCK_HASH = "2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
EXPECTED_AMENDMENT_HASH = "3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a"

IMMUTABLE = [
    LOCK,
    AMENDMENT,
    METRICS,
    PREDICTIONS,
    ROOT / "outputs/reports/final_track21_height_width_test_review_v2.md",
    ROOT / "outputs/reports/final_track21_height_width_leakage_audit_v2.md",
    ROOT / "outputs/reports/generalized_sem_mapping_production_equivalence_v1.md",
    ORIGINAL_FIGURE,
]

BAR_MODELS = [
    "cohort_b_x_only",
    "cohort_b_fixed_thermal_shape_ridge",
    "cohort_b_sem_summary_ridge",
    "cohort_b_thermal_plus_sem_summary_ridge",
]
SCATTER_MODELS = [
    "cohort_b_x_only",
    "cohort_b_sem_summary_ridge",
    "cohort_b_thermal_plus_sem_summary_ridge",
]
LABELS = {
    "cohort_b_x_only": "x-only",
    "cohort_b_fixed_thermal_shape_ridge": "fixed Thermal",
    "cohort_b_sem_summary_ridge": "SEM summary",
    "cohort_b_thermal_plus_sem_summary_ridge": "Thermal + SEM",
}
EXPECTED_MAE = {
    "cohort_b_x_only": 0.111856,
    "cohort_b_fixed_thermal_shape_ridge": 0.108451,
    "cohort_b_sem_summary_ridge": 0.109837,
    "cohort_b_thermal_plus_sem_summary_ridge": 0.110343,
}
COLORS = {
    "cohort_b_x_only": "#4C78A8",
    "cohort_b_fixed_thermal_shape_ridge": "#E6A23C",
    "cohort_b_sem_summary_ridge": "#59A14F",
    "cohort_b_thermal_plus_sem_summary_ridge": "#B279A2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def metric_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    cohort = metrics.loc[metrics["cohort_id"].eq("cohort_B")].copy()
    if cohort["model_id"].duplicated().any():
        raise RuntimeError("Duplicate Cohort-B metric model IDs")
    indexed = cohort.set_index("model_id")
    missing = sorted(set(BAR_MODELS) - set(indexed.index))
    if missing:
        raise RuntimeError(f"Missing required Cohort-B metrics: {missing}")
    for model, expected in EXPECTED_MAE.items():
        observed = float(indexed.loc[model, "mae_mm"])
        if not np.isfinite(observed) or abs(observed - expected) > 1e-6:
            raise RuntimeError(
                f"Frozen MAE mismatch for {model}: observed={observed}, expected≈{expected}"
            )
    return indexed


def validate_predictions(predictions: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], int]:
    selected = predictions.loc[
        predictions["cohort_id"].eq("cohort_B")
        & predictions["model_id"].isin(SCATTER_MODELS)
    ].copy()
    if selected.duplicated(["sample_id", "model_id"]).any():
        raise RuntimeError("Duplicate sample_id/model_id pairs in plotted predictions")
    groups: dict[str, pd.DataFrame] = {}
    reference_ids: list[str] | None = None
    reference_target: pd.Series | None = None
    for model in SCATTER_MODELS:
        group = selected.loc[selected["model_id"].eq(model)].sort_values("sample_id").reset_index(drop=True)
        if len(group) != 233:
            raise RuntimeError(f"{model} has {len(group)} rows; expected 233")
        if not np.isfinite(group[["target_width_mm", "prediction_mm"]].to_numpy(float)).all():
            raise RuntimeError(f"{model} contains nonfinite plotted values")
        ids = group["sample_id"].astype(str).tolist()
        target = group.set_index("sample_id")["target_width_mm"].sort_index()
        if reference_ids is None:
            reference_ids = ids
            reference_target = target
        else:
            if ids != reference_ids:
                raise RuntimeError(f"{model} does not use the identical 233 sample IDs")
            if not np.array_equal(target.to_numpy(), reference_target.to_numpy()):
                raise RuntimeError(f"{model} target_width_mm differs across models")
        groups[model] = group
    return groups, len(reference_ids or [])


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"
    if png == ORIGINAL_FIGURE or pdf == ORIGINAL_FIGURE:
        raise RuntimeError("Refusing to overwrite the original final-test figure")
    fig.savefig(png, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def create_mae_figure(metrics: pd.DataFrame) -> tuple[Path, Path]:
    values = [float(metrics.loc[model, "mae_mm"]) for model in BAR_MODELS]
    fig, ax = plt.subplots(figsize=(8.2, 5.6), facecolor="white")
    bars = ax.bar(
        [LABELS[m] for m in BAR_MODELS],
        values,
        color=[COLORS[m] for m in BAR_MODELS],
        width=0.68,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_title("Track 21 Cohort-B Prediction Error", fontsize=15, pad=14)
    ax.set_ylabel("MAE (mm)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.018,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    fig.text(
        0.5,
        0.015,
        "The fixed Thermal result is descriptive. The predeclared primary comparison "
        "was SEM summary versus x-only.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.98))
    return save_figure(fig, "final_track21_cohort_b_mae_comparison")


def create_threshold_figure(
    actual: float, required: float, relative_percent: float
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor="white")
    labels = ["Actual SEM\nimprovement", "Project-defined\nthreshold"]
    values = [actual, required]
    bars = ax.bar(
        labels,
        values,
        color=["#59A14F", "#9C9C9C"],
        edgecolor="#333333",
        linewidth=0.7,
        width=0.58,
    )
    ax.axhline(required, color="#555555", linestyle="--", linewidth=1.1, alpha=0.85)
    ax.set_ylabel("MAE reduction (mm)", fontsize=12)
    ax.set_title(
        "SEM Improvement Relative to the Project-Defined Threshold",
        fontsize=14,
        pad=14,
    )
    ax.set_ylim(0, required * 1.36)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + required * 0.035,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.text(
        0.03,
        0.95,
        f"Observed: {relative_percent:.3f}% relative reduction\n"
        "Project criterion: max(0.003 mm, 3%)",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    fig.text(
        0.5,
        0.018,
        "This project-defined criterion was not an official challenge requirement.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.98))
    return save_figure(fig, "final_track21_sem_improvement_threshold")


def create_scatter_figure(
    groups: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    include_range_note: bool,
) -> tuple[Path, Path]:
    combined = np.concatenate(
        [
            group[["target_width_mm", "prediction_mm"]].to_numpy(float).ravel()
            for group in groups.values()
        ]
    )
    lower, upper = float(combined.min()), float(combined.max())
    margin = max((upper - lower) * 0.045, 0.01)
    limits = (lower - margin, upper + margin)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), facecolor="white")
    for ax, model in zip(axes, SCATTER_MODELS):
        group = groups[model]
        ax.scatter(
            group["target_width_mm"],
            group["prediction_mm"],
            s=22,
            alpha=0.55,
            color=COLORS[model],
            edgecolors="none",
        )
        ax.plot(limits, limits, linestyle="--", color="#444444", linewidth=1.1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(LABELS[model], fontsize=13, pad=9)
        ax.set_xlabel("Frozen target width (mm)", fontsize=10.5)
        ax.set_ylabel("Predicted width (mm)", fontsize=10.5)
        ax.grid(color="#E0E0E0", linewidth=0.6, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        row = metrics.loc[model]
        ax.text(
            0.04,
            0.96,
            f"MAE = {float(row['mae_mm']):.6f} mm\n"
            f"RMSE = {float(row['rmse_mm']):.6f} mm\n"
            f"Pearson r = {float(row['pearson_correlation']):.3f}\n"
            f"R² = {float(row['r_squared']):.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
        )
    fig.suptitle(
        "Track 21: Predicted Width versus Frozen Height-Derived Target",
        fontsize=15,
        y=0.995,
    )
    note = "Points on the dashed identity line indicate exact agreement."
    if include_range_note:
        note += " All models predict a substantially narrower range than the frozen target."
    fig.text(0.5, 0.015, note, ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.95))
    return save_figure(fig, "final_track21_predicted_vs_target_scatter")


def inspect_outputs(paths: list[Path]) -> tuple[dict[str, str], dict[str, list[int]]]:
    hashes: dict[str, str] = {}
    dimensions: dict[str, list[int]] = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing or empty figure output: {path}")
        hashes[relative(path)] = sha256(path)
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            if width < 1000 or height < 700:
                raise RuntimeError(f"Unreasonable PNG dimensions for {path}: {width}x{height}")
            dimensions[relative(path)] = [width, height]
    return hashes, dimensions


def main() -> None:
    missing = [str(path) for path in IMMUTABLE if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing immutable inputs: {missing}")
    before = {relative(path): sha256(path) for path in IMMUTABLE}
    if before[relative(LOCK)] != EXPECTED_LOCK_HASH:
        raise RuntimeError("Immutable model-lock hash does not match the frozen value")
    if before[relative(AMENDMENT)] != EXPECTED_AMENDMENT_HASH:
        raise RuntimeError("Compatibility-amendment hash does not match the frozen value")

    metrics_raw = pd.read_csv(METRICS)
    predictions = pd.read_csv(PREDICTIONS)
    metrics = metric_rows(metrics_raw)
    groups, sample_count = validate_predictions(predictions)

    x_mae = float(metrics.loc["cohort_b_x_only", "mae_mm"])
    sem_mae = float(metrics.loc["cohort_b_sem_summary_ridge", "mae_mm"])
    actual = x_mae - sem_mae
    required = max(0.003, 0.03 * x_mae)
    relative_percent = 100.0 * actual / x_mae
    if not (
        abs(actual - 0.002019) <= 1e-6
        and abs(required - 0.003356) <= 1e-6
        and abs(relative_percent - 1.805) <= 0.001
    ):
        raise RuntimeError(
            f"Frozen comparison mismatch: actual={actual}, required={required}, relative={relative_percent}"
        )

    target_std = float(groups[SCATTER_MODELS[0]]["target_width_mm"].std(ddof=0))
    prediction_std = {
        model: float(groups[model]["prediction_mm"].std(ddof=0))
        for model in SCATTER_MODELS
    }
    include_range_note = all(value <= 0.75 * target_std for value in prediction_std.values())

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        *create_mae_figure(metrics),
        *create_threshold_figure(actual, required, relative_percent),
        *create_scatter_figure(groups, metrics, include_range_note),
    ]
    output_hashes, dimensions = inspect_outputs(outputs)

    after = {relative(path): sha256(path) for path in IMMUTABLE}
    immutable_ok = before == after
    if not immutable_ok:
        raise RuntimeError("An immutable frozen result changed during visualization")

    created = datetime.now(timezone.utc).isoformat()
    metric_values = {
        model: {
            "mae_mm": float(metrics.loc[model, "mae_mm"]),
            "rmse_mm": float(metrics.loc[model, "rmse_mm"]),
            "pearson_correlation": (
                None
                if pd.isna(metrics.loc[model, "pearson_correlation"])
                else float(metrics.loc[model, "pearson_correlation"])
            ),
            "r_squared": float(metrics.loc[model, "r_squared"]),
        }
        for model in BAR_MODELS
    }
    summary = f"""# Final Track-21 visualization summary v1

## Frozen sources

- `{relative(METRICS)}` — SHA-256 `{sha256(METRICS)}`
- `{relative(PREDICTIONS)}` — SHA-256 `{sha256(PREDICTIONS)}`
- Creation timestamp (UTC): `{created}`

No model fitting, estimator refitting, feature generation, label regeneration, model
selection, or metric recalculation was used to change the frozen results. Immutable
source and final-report hashes were identical before and after figure creation.

## Plotted models and values

- MAE figure model IDs, in order: `{json.dumps(BAR_MODELS)}`
- Scatter model IDs, in order: `{json.dumps(SCATTER_MODELS)}`
- Scatter sample count per model: `{sample_count}`
- Exact MAE values (mm): `{json.dumps({m: metric_values[m]['mae_mm'] for m in BAR_MODELS}, sort_keys=True)}`
- Actual SEM improvement: `{actual:.12f} mm`
- Project-defined predeclared confirmation threshold: `{required:.12f} mm`
- Relative SEM improvement: `{relative_percent:.9f}%`
- Target standard deviation (`ddof=0`): `{target_std:.12f} mm`
- Prediction standard deviations (`ddof=0`): `{json.dumps(prediction_std, sort_keys=True)}`

## Figures

![Track-21 predictions and absolute errors](../figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png)

![Track-21 Cohort-B MAE comparison](../figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png)

![SEM improvement threshold](../figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png)

![Predicted versus frozen target](../figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png)

Vector PDF versions are stored beside each PNG.

Generated PNG and PDF paths: `{json.dumps([relative(path) for path in outputs])}`.
The preserved pre-existing prediction/error PNG is
`{relative(ORIGINAL_FIGURE)}`.

## Interpretation

SEM summary produced a small positive MAE reduction relative to x-only, but the reduction
did not meet the project-defined predeclared confirmation threshold. This criterion was
defined for conservative interpretation and was not an official challenge requirement.
Not meeting it does not mean that the challenge submission failed. Thermal + SEM did not
improve over SEM-only.
The fixed Thermal-only Track-21 result is descriptive and does not overturn the lack of
stable Thermal generalization across development Tracks.

VISUALIZATIONS GENERATED FROM FROZEN FINAL RESULTS
"""
    SUMMARY.write_text(summary, encoding="utf-8")

    validation = {
        "protocol_version": "final_track21_result_visualizations_v1",
        "creation_timestamp_utc": created,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "output_paths": [relative(path) for path in outputs],
        "output_hashes": output_hashes,
        "png_dimensions_px": dimensions,
        "plotted_row_counts": {model: len(groups[model]) for model in SCATTER_MODELS},
        "plotted_model_ids": {
            "mae_comparison": BAR_MODELS,
            "scatter": SCATTER_MODELS,
        },
        "metric_values": metric_values,
        "actual_sem_improvement_mm": actual,
        "project_defined_predeclared_confirmation_threshold_mm": required,
        "relative_sem_improvement_percent": relative_percent,
        "target_standard_deviation_mm_ddof0": target_std,
        "prediction_standard_deviation_mm_ddof0": prediction_std,
        "narrower_range_note_supported": include_range_note,
        "model_fitting_performed": False,
        "frozen_result_modified": False,
        "immutable_file_hash_status": "PASS" if immutable_ok else "FAIL",
        "overall_status": "FINAL TRACK21 VISUALIZATIONS READY",
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    print("Cohort-B MAE values:")
    for model in BAR_MODELS:
        print(f"  {model}={metric_values[model]['mae_mm']:.12f} mm")
    print(f"actual_sem_improvement={actual:.12f} mm")
    print(f"project_defined_predeclared_confirmation_threshold={required:.12f} mm")
    print(f"relative_sem_improvement={relative_percent:.9f}%")
    print(f"scatter_sample_count_per_model={sample_count}")
    print(f"target_standard_deviation={target_std:.12f} mm")
    for model in SCATTER_MODELS:
        print(f"prediction_standard_deviation[{model}]={prediction_std[model]:.12f} mm")
    print("generated files:")
    for path in [*outputs, SUMMARY, VALIDATION]:
        print(f"  {relative(path)}")
    print(f"immutable_hash_status={'PASS' if immutable_ok else 'FAIL'}")
    print("FINAL TRACK21 VISUALIZATIONS READY")


if __name__ == "__main__":
    main()
