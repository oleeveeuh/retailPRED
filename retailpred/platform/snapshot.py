"""Versioned public snapshot contract for scheduled batch forecasts.

Contract 2.1: every point carries the vintage-safe seasonal-naive benchmark
frozen at forecast time, plus optional realized evaluation fields joined
from the governed realized-error table. Publication fails closed when gold
forecasts are absent, when the frozen portfolio decision has not authorized
the model, or when uncertainty bounds appear without authorization.
``is_live`` reflects the scheduled pipeline being enabled — it must stay
``false`` until scheduled ingestion, scoring, and publication have all run.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forecasting.portfolio import ModelPortfolioDecision, model_decision
from retailpred.platform.local_store import validate_table_schema
from retailpred.platform.specs import GOLD_FORECASTS

SNAPSHOT_CONTRACT_VERSION = "2.1"

POINT_EVALUATION_COLUMNS = (
    "evaluation_actual",
    "evaluation_actual_vintage_time",
    "model_absolute_error",
    "seasonal_naive_absolute_error",
    "model_beat_seasonal_naive",
    "evaluation_status",
)


class SnapshotPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    horizon: int = Field(ge=1)
    target_time: date
    prediction: float = Field(ge=0)
    lower: float | None = None
    upper: float | None = None
    seasonal_naive_prediction: float | None = None
    seasonal_naive_source_time: date | None = None
    seasonal_naive_source_vintage_time: datetime | None = None
    seasonal_naive_status: Literal["available", "unavailable"]
    seasonal_naive_unavailable_reason: str | None = None
    evaluation_actual: float | None = None
    evaluation_actual_vintage_time: datetime | None = None
    model_absolute_error: float | None = None
    seasonal_naive_absolute_error: float | None = None
    model_beat_seasonal_naive: bool | None = None
    evaluation_status: str | None = None


class SeriesForecast(BaseModel):
    series_id: str
    points: list[SnapshotPoint]


class SnapshotFreshness(BaseModel):
    status: Literal["current", "stale"]
    generated_at: datetime
    forecast_vintage_time: datetime
    data_cutoff: date
    next_scheduled_run: datetime | None = None
    max_age_hours: int = Field(ge=1)


class ForecastSnapshot(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    contract_version: Literal["2.1"] = SNAPSHOT_CONTRACT_VERSION
    snapshot_kind: Literal["scheduled_batch_forecasts"]
    dataset: Literal["mrts"]
    use_case: str
    forecast_origin: date
    model_name: str
    model_version: str
    model_status: str
    uncertainty_status: str
    is_live: bool
    pipeline_run_id: str | None = None
    freshness: SnapshotFreshness
    series: list[SeriesForecast]
    provenance: dict[str, str]

    @field_validator("series")
    @classmethod
    def _require_series(cls, value: list[SeriesForecast]) -> list[SeriesForecast]:
        if not value:
            raise ValueError("snapshot must contain at least one series")
        ids = [item.series_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot contains duplicate series")
        return value


def is_stale(generated_at: datetime, vintage_time: datetime, max_age_hours: int) -> bool:
    return (generated_at - vintage_time) > timedelta(hours=max_age_hours)


def authorized_decision(dataset: str) -> ModelPortfolioDecision:
    """Fail closed unless the frozen portfolio promotes the scoring model."""
    decision = model_decision(dataset)
    if decision.point_status != "promoted_point_champion":
        raise ValueError(
            "snapshot publication requires an authorized champion; frozen "
            f"decision for {dataset!r} is {decision.point_status!r} "
            f"({decision.serving_mode})"
        )
    return decision


def _join_realized(frame: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest realized evaluation to each forecast row."""
    key = [
        "dataset",
        "series_id",
        "forecast_origin",
        "target_time",
        "model_name",
        "model_version",
        "forecast_vintage_time",
    ]
    if realized.empty:
        for column in POINT_EVALUATION_COLUMNS:
            frame[column] = None
        return frame
    clean = realized.drop_duplicates(subset=list(key), keep="last")
    joined = frame.merge(
        clean[list(key) + list(POINT_EVALUATION_COLUMNS)],
        on=list(key),
        how="left",
        validate="one_to_one",
    )
    return joined


def build_snapshot(
    frame: pd.DataFrame,
    *,
    generated_at: datetime,
    is_live: bool = False,
    max_age_hours: int = 336,
    next_scheduled_run: datetime | None = None,
    pipeline_run_id: str | None = None,
    provenance: dict[str, str] | None = None,
    decision: ModelPortfolioDecision | None = None,
    realized: pd.DataFrame | None = None,
) -> ForecastSnapshot:
    """Validate gold forecast rows and assemble the publishable snapshot."""
    clean = validate_table_schema(frame, GOLD_FORECASTS)
    if clean.empty:
        raise ValueError("no gold forecasts exist; snapshot publishing is unavailable")
    datasets = clean["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError(f"snapshot rows mix datasets: {sorted(datasets)}")
    dataset = str(datasets[0])
    if dataset != "mrts":
        raise ValueError(f"no snapshot contract exists for dataset {dataset!r}")
    origins = clean["forecast_origin"].unique()
    if len(origins) != 1:
        raise ValueError(f"snapshot rows mix forecast origins: {sorted(origins)}")
    models = clean[["model_name", "model_version"]].drop_duplicates()
    if len(models) != 1:
        raise ValueError(f"snapshot rows mix model versions: {models.to_dict('records')}")
    vintage_times = clean["forecast_vintage_time"].unique()
    if len(vintage_times) != 1:
        raise ValueError("snapshot rows mix forecast vintages")

    resolved = decision if decision is not None else authorized_decision(dataset)

    bounds_present = clean[["lower", "upper"]].notna().any().any()
    if bounds_present and resolved.uncertainty_status != "promoted_calibrated_uncertainty":
        raise ValueError(
            "snapshot contains uncertainty bounds but the frozen portfolio "
            f"decision does not authorize them ({resolved.uncertainty_status})"
        )

    if realized is not None:
        clean = _join_realized(clean, realized)
    else:
        for column in POINT_EVALUATION_COLUMNS:
            clean[column] = None

    cutoff = pd.Timestamp(clean["data_cutoff"].iloc[0]).date()
    vintage_time = pd.Timestamp(vintage_times[0]).to_pydatetime()
    series_frames: list[SeriesForecast] = []
    for series_id, group in clean.groupby("series_id", sort=True):
        ordered = group.sort_values("target_time")
        points = [
            SnapshotPoint(
                horizon=int(row.horizon),
                target_time=pd.Timestamp(row.target_time).date(),
                prediction=float(row.prediction),
                lower=None if pd.isna(row.lower) else float(row.lower),
                upper=None if pd.isna(row.upper) else float(row.upper),
                seasonal_naive_prediction=None
                if pd.isna(row.seasonal_naive_prediction)
                else float(row.seasonal_naive_prediction),
                seasonal_naive_source_time=None
                if pd.isna(row.seasonal_naive_source_time)
                else pd.Timestamp(row.seasonal_naive_source_time).date(),
                seasonal_naive_source_vintage_time=None
                if pd.isna(row.seasonal_naive_source_vintage_time)
                else pd.Timestamp(row.seasonal_naive_source_vintage_time).to_pydatetime(),
                seasonal_naive_status=str(row.seasonal_naive_status),
                seasonal_naive_unavailable_reason=None
                if pd.isna(row.seasonal_naive_unavailable_reason)
                else str(row.seasonal_naive_unavailable_reason),
                evaluation_actual=None
                if pd.isna(row.evaluation_actual)
                else float(row.evaluation_actual),
                evaluation_actual_vintage_time=None
                if pd.isna(row.evaluation_actual_vintage_time)
                else pd.Timestamp(row.evaluation_actual_vintage_time).to_pydatetime(),
                model_absolute_error=None
                if pd.isna(row.model_absolute_error)
                else float(row.model_absolute_error),
                seasonal_naive_absolute_error=None
                if pd.isna(row.seasonal_naive_absolute_error)
                else float(row.seasonal_naive_absolute_error),
                model_beat_seasonal_naive=None
                if pd.isna(row.model_beat_seasonal_naive)
                else bool(row.model_beat_seasonal_naive),
                evaluation_status=None
                if pd.isna(row.evaluation_status)
                else str(row.evaluation_status),
            )
            for row in ordered.itertuples()
        ]
        series_frames.append(SeriesForecast(series_id=str(series_id), points=points))

    snapshot = ForecastSnapshot(
        snapshot_kind="scheduled_batch_forecasts",
        dataset=dataset,
        use_case=resolved.use_case,
        forecast_origin=pd.Timestamp(origins[0]).date(),
        model_name=str(models.iloc[0]["model_name"]),
        model_version=str(models.iloc[0]["model_version"]),
        model_status=resolved.point_status,
        uncertainty_status=resolved.uncertainty_status,
        is_live=is_live,
        pipeline_run_id=pipeline_run_id,
        freshness=SnapshotFreshness(
            status="stale" if is_stale(generated_at, vintage_time, max_age_hours) else "current",
            generated_at=generated_at,
            forecast_vintage_time=vintage_time,
            data_cutoff=cutoff,
            next_scheduled_run=next_scheduled_run,
            max_age_hours=max_age_hours,
        ),
        series=series_frames,
        provenance=provenance or {},
    )
    return snapshot


def write_snapshot(snapshot: ForecastSnapshot, path: str | Path) -> Path:
    """Write the snapshot atomically so readers never see a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
