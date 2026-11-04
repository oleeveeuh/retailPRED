"""Versioned public snapshot contract for scheduled batch forecasts.

A snapshot is the only shape the static frontend may consume for live-loop
output. It is distinct from the historical evaluation fixture: it is labeled
``is_live`` with ``snapshot_kind="scheduled_batch_forecasts"``, carries
freshness metadata, and refuses to embed uncertainty bounds unless the frozen
portfolio decision authorizes them.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from forecasting.portfolio import model_decision
from retailpred.platform.local_store import validate_table_schema
from retailpred.platform.specs import GOLD_FORECASTS

SNAPSHOT_CONTRACT_VERSION = "2.0"


class SnapshotPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    horizon: int = Field(ge=1)
    target_time: date
    prediction: float = Field(ge=0)
    lower: float | None = None
    upper: float | None = None


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

    contract_version: Literal["2.0"] = SNAPSHOT_CONTRACT_VERSION
    snapshot_kind: Literal["scheduled_batch_forecasts"]
    dataset: Literal["mrts"]
    use_case: str
    forecast_origin: date
    model_name: str
    model_version: str
    model_status: str
    uncertainty_status: str
    is_live: Literal[True]
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


def build_snapshot(
    frame: pd.DataFrame,
    *,
    generated_at: datetime,
    max_age_hours: int = 336,
    next_scheduled_run: datetime | None = None,
    provenance: dict[str, str] | None = None,
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

    decision = model_decision(dataset)
    bounds_present = clean[["lower", "upper"]].notna().any().any()
    if bounds_present and decision.uncertainty_status != "promoted_calibrated_uncertainty":
        raise ValueError(
            "snapshot contains uncertainty bounds but the frozen portfolio "
            f"decision does not authorize them ({decision.uncertainty_status})"
        )

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
            )
            for row in ordered.itertuples()
        ]
        series_frames.append(SeriesForecast(series_id=str(series_id), points=points))

    snapshot = ForecastSnapshot(
        snapshot_kind="scheduled_batch_forecasts",
        dataset=dataset,
        use_case=decision.use_case,
        forecast_origin=pd.Timestamp(origins[0]).date(),
        model_name=str(models.iloc[0]["model_name"]),
        model_version=str(models.iloc[0]["model_version"]),
        model_status=decision.point_status,
        uncertainty_status=decision.uncertainty_status,
        is_live=True,
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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination
