"""Fail-closed schemas for Tableau-ready CSV extracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExtractContract:
    columns: tuple[str, ...]
    key: tuple[str, ...]
    non_negative: tuple[str, ...] = ()

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(frame.columns) != self.columns:
            raise ValueError("BI extract columns do not match the governed contract")
        if frame.empty:
            raise ValueError("BI extract cannot be empty")
        if frame.loc[:, self.key].isna().any().any():
            raise ValueError("BI extract keys cannot be null")
        if frame.duplicated(list(self.key)).any():
            raise ValueError("BI extract contains duplicate keys")
        for column in self.non_negative:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column} must contain finite non-negative values")
        return frame


MODEL_PERFORMANCE = ExtractContract(
    columns=(
        "dataset",
        "use_case",
        "model_name",
        "decision_status",
        "evaluation_start",
        "evaluation_end",
        "metric_name",
        "metric_value",
        "baseline_value",
        "relative_improvement",
        "code_revision",
        "data_version",
    ),
    key=("dataset", "model_name", "metric_name", "evaluation_start", "evaluation_end"),
    non_negative=("metric_value", "baseline_value"),
)

M5_HIERARCHY_ACCURACY = ExtractContract(
    columns=(
        "dataset",
        "model_name",
        "evaluation_start",
        "evaluation_end",
        "level",
        "level_name",
        "series_count",
        "weight",
        "level_wrmsse",
        "baseline_level_wrmsse",
        "relative_improvement",
        "code_revision",
    ),
    key=("dataset", "model_name", "evaluation_start", "evaluation_end", "level"),
    non_negative=("series_count", "weight", "level_wrmsse", "baseline_level_wrmsse"),
)

M5_UNCERTAINTY = ExtractContract(
    columns=(
        "dataset",
        "model_name",
        "evaluation_start",
        "evaluation_end",
        "dimension_type",
        "dimension_key",
        "dimension_label",
        "entity_count",
        "nominal_coverage",
        "coverage",
        "mean_scaled_width",
        "normalized_wis",
        "decision_status",
        "code_revision",
    ),
    key=("dataset", "model_name", "evaluation_start", "evaluation_end", "dimension_type", "dimension_key"),
    non_negative=("entity_count", "nominal_coverage", "coverage", "mean_scaled_width", "normalized_wis"),
)

MRTS_REGIMES = ExtractContract(
    columns=(
        "dataset",
        "regime",
        "evaluation_start",
        "evaluation_end",
        "fold_rows_per_model",
        "lightgbm_wape",
        "seasonal_naive_wape",
        "relative_wape_improvement",
        "mean_category_wape_lightgbm",
        "mean_category_wape_seasonal_naive",
        "relative_mean_category_wape_improvement",
        "categories",
        "improved_categories",
        "regime_gate_passed",
        "code_revision",
    ),
    key=("dataset", "regime"),
    non_negative=(
        "fold_rows_per_model",
        "lightgbm_wape",
        "seasonal_naive_wape",
        "mean_category_wape_lightgbm",
        "mean_category_wape_seasonal_naive",
        "categories",
        "improved_categories",
    ),
)

DATA_FRESHNESS = ExtractContract(
    columns=(
        "dataset",
        "use_case",
        "evaluation_cutoff",
        "as_of_date",
        "serving_mode",
        "freshness_status",
        "point_status",
        "uncertainty_status",
        "evidence",
    ),
    key=("dataset", "as_of_date"),
)

LIVE_FORECASTS = ExtractContract(
    columns=(
        "dataset",
        "series_id",
        "forecast_origin",
        "target_time",
        "horizon",
        "prediction",
        "model_name",
        "model_version",
        "model_status",
        "uncertainty_status",
        "freshness_status",
        "code_revision",
    ),
    key=("series_id", "target_time", "horizon"),
    non_negative=("prediction",),
)

CONTRACTS = {
    "model_performance.csv": MODEL_PERFORMANCE,
    "m5_hierarchy_accuracy.csv": M5_HIERARCHY_ACCURACY,
    "m5_uncertainty.csv": M5_UNCERTAINTY,
    "mrts_regime_accuracy.csv": MRTS_REGIMES,
    "data_freshness.csv": DATA_FRESHNESS,
    "live_forecasts.csv": LIVE_FORECASTS,
}
