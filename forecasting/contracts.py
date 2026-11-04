"""Canonical records shared by ingestion, evaluation, and serving."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Documented encodings of the seasonal-adjustment marker across Census
# surfaces: the timeseries API historically returned yes/no, while MRTS
# workbooks and some API schema revisions use SA/NSA (or boolean forms).
SEASONAL_ADJUSTED_VALUES = frozenset({"yes", "y", "true", "sa"})
SEASONAL_UNADJUSTED_VALUES = frozenset({"no", "n", "false", "nsa"})

_PERIOD_FORMATS = ("%Y-%m", "%b-%y", "%b-%Y", "%Y-%m-%d")
_YEAR_ONLY_PATTERN = re.compile(r"^\d{4}$")


def parse_period(value: str | int) -> "pd.Timestamp":
    """Parse a documented Census period string to a month start, or NaT.

    Documented encodings: ISO months (``2010-01``), three-letter month with
    two- or four-digit year (``Jan-10``, ``Jan-2010``), and ISO dates. A bare
    year (``2025``) is deliberately NOT a period: legacy responses carry the
    year in ``time`` and the month in ``time_slot_id``.
    """
    text = str(value).strip()
    for fmt in _PERIOD_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)
    if _YEAR_ONLY_PATTERN.match(text):
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)
    return pd.NaT


def derive_month_periods(
    frame: "pd.DataFrame",
    *,
    time_column: str = "time",
    slot_column: str = "time_slot_id",
    year_hint: int | None = None,
) -> "pd.DataFrame":
    """Derive month numbers and observation years from documented columns.

    Current API responses carry the period in ``time`` with an ordinal
    ``time_slot_id``; legacy responses carried a bare year in ``time`` and the
    month 1-12 in ``time_slot_id``. Both are supported. Returns columns
    ``month`` (1-12 or NA) and ``observation_year`` (NA when undeterminable).
    """
    if time_column in frame.columns:
        # Cast through to_datetime so empty selections still get a
        # datetime64 dtype for the .dt accessor below.
        parsed = pd.to_datetime(
            pd.Series(list(frame[time_column].map(parse_period)), index=frame.index),
            errors="coerce",
        )
    else:
        parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    year_valid = parsed.notna()
    if year_hint is not None:
        year_valid &= parsed.dt.year.eq(year_hint)

    slots = pd.to_numeric(frame.get(slot_column), errors="coerce")
    slot_valid = slots.between(1, 12)

    months = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    months.loc[year_valid] = parsed.loc[year_valid].dt.month.astype("Int64")
    slot_months = slot_valid & months.isna()
    months.loc[slot_months] = slots.loc[slot_months].astype("Int64")

    years = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    years.loc[year_valid] = parsed.loc[year_valid].dt.year.astype("Int64")
    if year_hint is not None:
        years.loc[slot_months] = int(year_hint)

    return pd.DataFrame({"month": months, "observation_year": years})


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class SeriesMetadata(Contract):
    series_id: str
    source: str
    source_code: str
    name: str
    frequency: Literal["monthly"] = "monthly"
    unit: str
    seasonal_adjustment: Literal["seasonally_adjusted", "not_adjusted"]


class Observation(Contract):
    series_id: str
    observation_time: date
    value: float
    release_time: datetime | None = None
    vintage_time: datetime
    ingested_at: datetime
    source: str

    @field_validator("release_time", "vintage_time", "ingested_at")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def ingestion_cannot_precede_vintage(self) -> "Observation":
        if self.ingested_at < self.vintage_time:
            raise ValueError("ingested_at cannot precede vintage_time")
        if self.release_time is not None and self.ingested_at < self.release_time:
            raise ValueError("ingested_at cannot precede release_time")
        return self


class Release(Contract):
    release_id: str
    source: str
    release_time: datetime
    vintage_time: datetime
    source_url: str

    @field_validator("release_time", "vintage_time")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class Forecast(Contract):
    series_id: str
    forecast_origin: date
    target_time: date
    horizon: int = Field(gt=0)
    training_cutoff: date
    data_vintage: datetime
    model_name: str
    model_version: str
    prediction: float
    actual_source: str

    @field_validator("data_vintage")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("data_vintage must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "Forecast":
        if self.training_cutoff > self.forecast_origin:
            raise ValueError("training_cutoff cannot follow forecast_origin")
        if self.target_time <= self.forecast_origin:
            raise ValueError("target_time must follow forecast_origin")
        return self


class Actual(Contract):
    series_id: str
    target_time: date
    value: float
    source: str
    vintage_time: datetime

    @field_validator("vintage_time")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("vintage_time must be timezone-aware")
        return value
