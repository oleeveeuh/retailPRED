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

MRTS_RESEARCH_WINDOWS = ExtractContract(
    columns=(
        "dataset",
        "model_name",
        "decision_status",
        "evaluation_window",
        "actual_definition",
        "evidence_class",
        "origin_count",
        "evaluation_start",
        "evaluation_end",
        "metric_name",
        "metric_value",
        "baseline_value",
        "relative_improvement",
        "categories_beating_naive",
        "categories_total",
        "beats_naive_in_window",
        "evidence_reference",
        "code_revision",
        "data_version",
    ),
    key=("model_name", "evaluation_window", "actual_definition"),
    non_negative=(
        "origin_count",
        "metric_value",
        "baseline_value",
        "categories_beating_naive",
        "categories_total",
    ),
)

MRTS_FEATURE_LADDER = ExtractContract(
    columns=(
        "dataset",
        "rung",
        "feature_groups",
        "compare_to",
        "mean_category_delta_pp",
        "improved_origins",
        "origins_total",
        "improved_origin_share",
        "advanced",
        "verdict",
        "evidence_reference",
        "code_revision",
        "data_version",
    ),
    key=("rung",),
    non_negative=("improved_origins", "origins_total", "improved_origin_share"),
)

MRTS_HORIZON_ACCURACY = ExtractContract(
    columns=(
        "dataset",
        "model_name",
        "evaluation_window",
        "actual_definition",
        "horizon",
        "mean_category_wape",
        "decision_status",
        "evidence_reference",
        "code_revision",
        "data_version",
    ),
    key=("model_name", "evaluation_window", "actual_definition", "horizon"),
    non_negative=("horizon", "mean_category_wape"),
)

MRTS_CANDIDATE_COMPARISONS = ExtractContract(
    columns=(
        "dataset",
        "model_name",
        "reference_model",
        "evaluation_window",
        "actual_definition",
        "origin_count",
        "mean_category_delta_pp",
        "ci_low_pp",
        "ci_high_pp",
        "resolved",
        "verdict",
        "evidence_reference",
        "code_revision",
        "data_version",
    ),
    key=("model_name", "reference_model", "evaluation_window", "actual_definition"),
    non_negative=("origin_count",),
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
    "mrts_research_windows.csv": MRTS_RESEARCH_WINDOWS,
    "mrts_feature_ladder.csv": MRTS_FEATURE_LADDER,
    "mrts_horizon_accuracy.csv": MRTS_HORIZON_ACCURACY,
    "mrts_candidate_comparisons.csv": MRTS_CANDIDATE_COMPARISONS,
    "data_freshness.csv": DATA_FRESHNESS,
    "live_forecasts.csv": LIVE_FORECASTS,
}
