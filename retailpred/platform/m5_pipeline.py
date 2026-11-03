"""M5 file registration, normalization, hierarchy checks, and gold assembly."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retailpred.platform.local_store import LocalTableStore, validate_table_schema
from retailpred.platform.specs import (
    BRONZE_SOURCE_FILES,
    GOLD_M5_DEMAND,
    SILVER_M5_CALENDAR,
    SILVER_M5_PRICES,
    SILVER_M5_SALES,
)


M5_ID_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
M5_CALENDAR_COLUMNS = tuple(
    column for column in SILVER_M5_CALENDAR.columns if column != "source_file_id"
)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source file without reading the entire artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class M5SourceIds:
    calendar: str
    sales: str
    prices: str


def source_file_record(
    path: str | Path,
    *,
    dataset: str,
    source_url: str,
    license_scope: str,
    ingested_at: datetime,
) -> pd.DataFrame:
    source_path = Path(path)
    if ingested_at.tzinfo is None or ingested_at.utcoffset() is None:
        raise ValueError("ingested_at must be timezone-aware")
    digest = file_sha256(source_path)
    result = pd.DataFrame(
        [
            {
                "source_file_id": digest,
                "dataset": dataset,
                "filename": source_path.name,
                "sha256": digest,
                "size_bytes": source_path.stat().st_size,
                "source_url": source_url,
                "license_scope": license_scope,
                "ingested_at": ingested_at,
            }
        ],
        columns=BRONZE_SOURCE_FILES.columns,
    )
    return validate_table_schema(result, BRONZE_SOURCE_FILES)


def normalize_m5_calendar(frame: pd.DataFrame, *, source_file_id: str) -> pd.DataFrame:
    if missing := set(M5_CALENDAR_COLUMNS).difference(frame.columns):
        raise ValueError(f"M5 calendar is missing columns: {sorted(missing)}")
    result = frame.loc[:, M5_CALENDAR_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], format="mixed", errors="raise")
    if result["d"].duplicated().any() or result["date"].duplicated().any():
        raise ValueError("M5 calendar day and date must both be unique")
    if not result["d"].astype(str).str.fullmatch(r"d_\d+").all():
        raise ValueError("M5 calendar contains invalid day identifiers")
    for column in ("snap_CA", "snap_TX", "snap_WI"):
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1")
    result["source_file_id"] = source_file_id
    return validate_table_schema(result.loc[:, SILVER_M5_CALENDAR.columns], SILVER_M5_CALENDAR)


def normalize_m5_sales(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    source_file_id: str,
) -> pd.DataFrame:
    if missing := set(M5_ID_COLUMNS).difference(frame.columns):
        raise ValueError(f"M5 sales are missing ID columns: {sorted(missing)}")
    day_columns = sorted(
        (column for column in frame.columns if re.fullmatch(r"d_\d+", str(column))),
        key=lambda value: int(value.split("_")[1]),
    )
    if not day_columns:
        raise ValueError("M5 sales contain no daily value columns")
    calendar_days = calendar.loc[:, ["d", "date"]]
    missing_days = set(day_columns).difference(calendar_days["d"])
    if missing_days:
        raise ValueError(f"M5 calendar is missing sales days: {sorted(missing_days)[:5]}")
    long = frame.melt(
        id_vars=list(M5_ID_COLUMNS),
        value_vars=day_columns,
        var_name="d",
        value_name="units",
    )
    long["units"] = pd.to_numeric(long["units"], errors="raise")
    if (long["units"] < 0).any() or not np.isfinite(long["units"]).all():
        raise ValueError("M5 unit sales must be finite and non-negative")
    if not np.equal(long["units"], np.floor(long["units"])).all():
        raise ValueError("M5 unit sales must be whole numbers")
    long = long.merge(calendar_days, on="d", how="left", validate="many_to_one")
    long = long.rename(columns={"id": "series_id"})
    long["source_file_id"] = source_file_id
    result = long.loc[:, SILVER_M5_SALES.columns]
    return validate_table_schema(result, SILVER_M5_SALES)


def normalize_m5_prices(frame: pd.DataFrame, *, source_file_id: str) -> pd.DataFrame:
    required = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"M5 prices are missing columns: {sorted(missing)}")
    result = frame.loc[:, sorted(required)].copy()
    result["sell_price"] = pd.to_numeric(result["sell_price"], errors="raise")
    if (result["sell_price"] <= 0).any() or not np.isfinite(result["sell_price"]).all():
        raise ValueError("M5 sell prices must be finite and positive")
    result["source_file_id"] = source_file_id
    return validate_table_schema(result.loc[:, SILVER_M5_PRICES.columns], SILVER_M5_PRICES)


def build_m5_gold(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    calendar_for_join = calendar.rename(
        columns={"source_file_id": "calendar_source_file_id"}
    )
    prices_for_join = prices.rename(columns={"source_file_id": "price_source_file_id"})
    result = sales.rename(columns={"source_file_id": "sales_source_file_id"}).merge(
        calendar_for_join.drop(columns="date"),
        on="d",
        how="left",
        validate="many_to_one",
    )
    if result["wm_yr_wk"].isna().any():
        raise ValueError("M5 calendar relationship failed")
    result = result.merge(
        prices_for_join,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    hierarchy_columns = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for column in hierarchy_columns:
        if result.groupby("series_id")[column].nunique().gt(1).any():
            raise ValueError(f"M5 hierarchy is inconsistent for {column}")
    return validate_table_schema(result.loc[:, GOLD_M5_DEMAND.columns], GOLD_M5_DEMAND)


def ingest_m5_frames(
    store: LocalTableStore,
    *,
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    source_files: pd.DataFrame,
    source_ids: M5SourceIds,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registered = store.merge(BRONZE_SOURCE_FILES, source_files)
    registered_ids = set(registered["source_file_id"])
    required_ids = {source_ids.calendar, source_ids.sales, source_ids.prices}
    if missing := required_ids.difference(registered_ids):
        raise ValueError(f"M5 source files are not registered: {sorted(missing)}")
    calendar_batch = normalize_m5_calendar(calendar, source_file_id=source_ids.calendar)
    calendar_table = store.merge(SILVER_M5_CALENDAR, calendar_batch)
    sales_batch = normalize_m5_sales(
        sales, calendar_table, source_file_id=source_ids.sales
    )
    sales_table = store.merge(SILVER_M5_SALES, sales_batch)
    prices_batch = normalize_m5_prices(prices, source_file_id=source_ids.prices)
    prices_table = store.merge(SILVER_M5_PRICES, prices_batch)
    gold = build_m5_gold(sales_table, calendar_table, prices_table)
    gold_table = store.replace_snapshot(GOLD_M5_DEMAND, gold)
    return calendar_table, sales_table, prices_table, gold_table
