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


BRONZE_FRED = TableSpec(
    name="fred_extract_rows",
    layer="bronze",
    schema_version=1,
    columns=(
        "source_record_id",
        "extract_id",
        "series_id",
        "source_url",
        "observation_date",
        "raw_value",
        "realtime_start",
        "realtime_end",
        "vintage_time",
        "ingested_at",
    ),
    key=("source_record_id",),
)


SILVER_ECONOMIC = TableSpec(
    name="economic_observations",
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


RETAIL_RELEASES = TableSpec(
    name="retail_releases",
    layer="silver",
    schema_version=1,
    columns=(
        "release_id",
        "product",
        "observation_month",
        "release_time",
        "release_kind",
        "source_url",
        "ingested_at",
    ),
    key=("release_id",),
)


BRONZE_SOURCE_FILES = TableSpec(
    name="source_files",
    layer="bronze",
    schema_version=1,
    columns=(
        "source_file_id",
        "dataset",
        "filename",
        "sha256",
        "size_bytes",
        "source_url",
        "license_scope",
        "ingested_at",
    ),
    key=("source_file_id",),
)


SILVER_M5_CALENDAR = TableSpec(
    name="m5_calendar",
    layer="silver",
    schema_version=1,
    columns=(
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "source_file_id",
    ),
    key=("d",),
    nullable=("event_name_1", "event_type_1", "event_name_2", "event_type_2"),
)


SILVER_M5_SALES = TableSpec(
    name="m5_daily_sales",
    layer="silver",
    schema_version=1,
    columns=(
        "series_id",
        "date",
        "d",
        "units",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "source_file_id",
    ),
    key=("series_id", "date"),
)


SILVER_M5_PRICES = TableSpec(
    name="m5_weekly_prices",
    layer="silver",
    schema_version=1,
    columns=("store_id", "item_id", "wm_yr_wk", "sell_price", "source_file_id"),
    key=("store_id", "item_id", "wm_yr_wk"),
)


GOLD_M5_DEMAND = TableSpec(
    name="m5_demand",
    layer="gold",
    schema_version=1,
    columns=(
        "series_id",
        "date",
        "units",
        "sell_price",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sales_source_file_id",
        "calendar_source_file_id",
        "price_source_file_id",
    ),
    key=("series_id", "date"),
    nullable=(
        "sell_price",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "price_source_file_id",
    ),
)
