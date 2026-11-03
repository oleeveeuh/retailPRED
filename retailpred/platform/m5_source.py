"""Memory-bounded validation for the licensed official M5 source files."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from forecasting.reconciliation.m5 import M5Hierarchy
from retailpred.platform.m5_pipeline import (
    M5_CALENDAR_COLUMNS,
    M5_ID_COLUMNS,
    file_sha256,
    normalize_m5_calendar,
)


M5_SOURCE_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
M5_LICENSE_SCOPE = "Kaggle competition rules; local use only"
OFFICIAL_BOTTOM_SERIES = 30_490
OFFICIAL_SALES_DAYS = 1_941
OFFICIAL_CALENDAR_DAYS = 1_969
OFFICIAL_PRICE_ROWS = 6_841_121


@dataclass(frozen=True)
class M5SourceValidation:
    validated_at: str
    source_directory: str
    file_sha256: dict[str, str]
    file_size_bytes: dict[str, int]
    bottom_series: int
    hierarchy_nodes: int
    hierarchy_level_counts: dict[str, int]
    sales_days: int
    sales_start_day: str
    sales_end_day: str
    total_units: int
    calendar_days: int
    calendar_start_date: str
    calendar_end_date: str
    price_rows: int
    price_min: float
    price_max: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"M5 source file is empty: {path.name}") from error


def _validate_sales(
    path: Path, calendar: pd.DataFrame, *, chunk_size: int
) -> tuple[pd.DataFrame, list[str], int]:
    header = _csv_header(path)
    if missing := set(M5_ID_COLUMNS).difference(header):
        raise ValueError(f"M5 sales are missing ID columns: {sorted(missing)}")
    day_columns = [column for column in header if column.startswith("d_")]
    expected_days = [f"d_{day}" for day in range(1, OFFICIAL_SALES_DAYS + 1)]
    if day_columns != expected_days:
        raise ValueError("M5 evaluation sales must contain consecutive d_1 through d_1941")
    if not set(day_columns).issubset(set(calendar["d"])):
        raise ValueError("M5 calendar does not cover every evaluation sales day")

    metadata = pd.read_csv(path, usecols=list(M5_ID_COLUMNS), dtype="string")
    if len(metadata) != OFFICIAL_BOTTOM_SERIES:
        raise ValueError(
            f"official M5 evaluation sales require {OFFICIAL_BOTTOM_SERIES:,} rows"
        )
    if metadata.isna().any().any() or metadata["id"].duplicated().any():
        raise ValueError("M5 bottom-series identifiers must be complete and unique")
    if not metadata["id"].str.endswith("_evaluation").all():
        raise ValueError("sales_train_evaluation.csv must use evaluation series IDs")

    total_units = 0
    for chunk in pd.read_csv(path, usecols=day_columns, chunksize=chunk_size):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("M5 unit sales must be finite and non-negative")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("M5 unit sales must be whole numbers")
        total_units += int(values.sum())
    return metadata, day_columns, total_units


def _validate_prices(
    path: Path,
    *,
    bottom: pd.DataFrame,
    calendar: pd.DataFrame,
    chunk_size: int,
) -> tuple[int, float, float]:
    required = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    if missing := set(required).difference(_csv_header(path)):
        raise ValueError(f"M5 prices are missing columns: {sorted(missing)}")
    valid_pairs = set(zip(bottom["store_id"], bottom["item_id"], strict=True))
    valid_weeks = set(calendar["wm_yr_wk"].astype(int))
    rows = 0
    minimum = np.inf
    maximum = -np.inf
    previous_key: tuple[str, str, int] | None = None
    for chunk in pd.read_csv(
        path,
        usecols=required,
        dtype={"store_id": "string", "item_id": "string"},
        chunksize=chunk_size,
    ):
        prices = pd.to_numeric(chunk["sell_price"], errors="coerce").to_numpy()
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("M5 sell prices must be finite and positive")
        weeks = pd.to_numeric(chunk["wm_yr_wk"], errors="coerce")
        if weeks.isna().any() or not set(weeks.astype(int)).issubset(valid_weeks):
            raise ValueError("M5 prices contain a week absent from the calendar")
        pairs = set(zip(chunk["store_id"], chunk["item_id"], strict=True))
        if not pairs.issubset(valid_pairs):
            raise ValueError("M5 prices contain an unknown store/item relationship")
        keys = list(zip(chunk["store_id"], chunk["item_id"], weeks.astype(int), strict=True))
        if len(keys) != len(set(keys)) or (previous_key is not None and keys[0] == previous_key):
            raise ValueError("M5 prices contain duplicate store/item/week keys")
        previous_key = keys[-1]
        rows += len(chunk)
        minimum = min(minimum, float(prices.min()))
        maximum = max(maximum, float(prices.max()))
    if rows != OFFICIAL_PRICE_ROWS:
        raise ValueError(f"official M5 prices require {OFFICIAL_PRICE_ROWS:,} rows")
    return rows, minimum, maximum


def validate_official_m5_sources(
    source_directory: str | Path, *, chunk_size: int = 512
) -> M5SourceValidation:
    """Validate official dimensions and relationships with bounded peak memory."""
    started = perf_counter()
    source = Path(source_directory).resolve()
    paths = {
        "calendar.csv": source / "calendar.csv",
        "sales_train_evaluation.csv": source / "sales_train_evaluation.csv",
        "sell_prices.csv": source / "sell_prices.csv",
    }
    if missing := [name for name, path in paths.items() if not path.is_file()]:
        raise FileNotFoundError(f"missing required M5 source files: {missing}")

    calendar_raw = pd.read_csv(paths["calendar.csv"])
    if set(calendar_raw.columns) != set(M5_CALENDAR_COLUMNS):
        raise ValueError("M5 calendar columns do not match the official contract")
    calendar = normalize_m5_calendar(calendar_raw, source_file_id="preflight")
    if len(calendar) != OFFICIAL_CALENDAR_DAYS:
        raise ValueError(f"official M5 calendar requires {OFFICIAL_CALENDAR_DAYS:,} rows")

    metadata, day_columns, total_units = _validate_sales(
        paths["sales_train_evaluation.csv"], calendar, chunk_size=chunk_size
    )
    bottom = metadata.rename(columns={"id": "series_id"})
    hierarchy = M5Hierarchy.from_bottom(bottom)
    hierarchy.validate_official_shape()
    price_rows, price_min, price_max = _validate_prices(
        paths["sell_prices.csv"],
        bottom=bottom,
        calendar=calendar,
        chunk_size=max(chunk_size * 1_000, 100_000),
    )

    hashes = {name: file_sha256(path) for name, path in paths.items()}
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    level_counts = hierarchy.nodes.groupby("level").size().to_dict()
    return M5SourceValidation(
        validated_at=datetime.now(timezone.utc).isoformat(),
        source_directory=str(source),
        file_sha256=hashes,
        file_size_bytes=sizes,
        bottom_series=len(bottom),
        hierarchy_nodes=len(hierarchy.nodes),
        hierarchy_level_counts={str(key): int(value) for key, value in level_counts.items()},
        sales_days=len(day_columns),
        sales_start_day=day_columns[0],
        sales_end_day=day_columns[-1],
        total_units=total_units,
        calendar_days=len(calendar),
        calendar_start_date=calendar["date"].min().date().isoformat(),
        calendar_end_date=calendar["date"].max().date().isoformat(),
        price_rows=price_rows,
        price_min=price_min,
        price_max=price_max,
        elapsed_seconds=round(perf_counter() - started, 3),
    )
