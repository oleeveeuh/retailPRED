"""Logical table specifications independent of a storage engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    name: str
    layer: str
    schema_version: int
    columns: tuple[str, ...]
    key: tuple[str, ...]
    nullable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.layer not in {"bronze", "silver", "gold"}:
            raise ValueError("layer must be bronze, silver, or gold")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not set(self.key).issubset(self.columns):
            raise ValueError("table key must be part of the declared columns")
        if not set(self.nullable).issubset(self.columns):
            raise ValueError("nullable fields must be part of the declared columns")


BRONZE_MRTS = TableSpec(
    name="mrts_extract_rows",
    layer="bronze",
    schema_version=1,
    columns=(
        "source_record_id",
        "extract_id",
        "source_url",
        "data_type_code",
        "time_slot_id",
        "seasonally_adj",
        "category_code",
        "cell_value",
        "error_data",
        "year",
        "release_time",
        "vintage_time",
        "ingested_at",
    ),
    key=("source_record_id",),
    nullable=("error_data", "release_time"),
)


SILVER_MONTHLY = TableSpec(
    name="monthly_observations",
    layer="silver",
    schema_version=1,
    columns=(
        "series_id",
        "observation_time",
        "value",
        "release_time",
        "vintage_time",
        "ingested_at",
        "source",
    ),
    key=("series_id", "observation_time", "vintage_time"),
    nullable=("release_time",),
)


GOLD_MONTHLY = TableSpec(
    name="monthly_series",
    layer="gold",
    schema_version=1,
    columns=(
        "series_id",
        "source_code",
        "series_name",
        "observation_time",
        "value",
        "unit",
        "seasonal_adjustment",
        "release_time",
        "vintage_time",
        "source",
    ),
    key=("series_id", "observation_time"),
    nullable=("release_time",),
)
