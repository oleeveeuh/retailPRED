"""Canonical records shared by ingestion, evaluation, and serving."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    vintage_time: datetime
    ingested_at: datetime
    source: str

    @field_validator("vintage_time", "ingested_at")
    @classmethod
    def timestamp_must_have_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def ingestion_cannot_precede_vintage(self) -> "Observation":
        if self.ingested_at < self.vintage_time:
            raise ValueError("ingested_at cannot precede vintage_time")
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
