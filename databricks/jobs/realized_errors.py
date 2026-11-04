"""Append realized-error rows to the declared Delta gold table.

Reads the published gold forecasts and versioned silver observations from
Unity Catalog, scores every forecast target that now has a revised actual,
and merges the immutable realized-error rows. Safe to run on any schedule:
exact replays are no-ops and revised actuals append new rows.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd  # noqa: E402

from retailpred.live_loop.realized import compute_realized_errors  # noqa: E402
from retailpred.platform.spark_delta import merge_delta_immutable  # noqa: E402
from retailpred.platform.specs import (  # noqa: E402
    GOLD_FORECASTS,
    GOLD_REALIZED_ERRORS,
    SILVER_MONTHLY,
)


def to_spark_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize timezone-aware columns to naive UTC for spark.createDataFrame."""
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64tz_dtype(result[column]):
            result[column] = result[column].dt.tz_convert("UTC").dt.tz_localize(None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    forecasts = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_FORECASTS.name}`"
    ).toPandas()
    if forecasts.empty:
        print("No published forecasts exist yet; realized monitoring is skipped.")
        return
    silver = spark.table(
        f"`{args.catalog}`.`silver`.`{SILVER_MONTHLY.name}`"
    ).toPandas()

    errors = compute_realized_errors(
        forecasts, silver, now=datetime.now(timezone.utc)
    )
    if errors.empty:
        print("No forecast targets have revised actuals yet; nothing to append.")
        return

    merge_delta_immutable(
        spark, to_spark_ready(errors), GOLD_REALIZED_ERRORS, catalog=args.catalog
    )
    total = spark.table(
        f"`{args.catalog}`.`gold`.`{GOLD_REALIZED_ERRORS.name}`"
    ).count()
    print(f"Appended realized-error rows; table now holds {total:,} rows.")


if __name__ == "__main__":
    main()
