"""Scheduled live-loop ingestion against the declared Delta tables.

Pulls the latest Census release using the API key stored in a Databricks
secret scope, merges the versioned silver batch immutably, and scores only
if the frozen portfolio decision authorizes it. The bundle declares this job
paused; unpause it after the secret scope is configured.
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

from forecasting.data.mrts import MRTSClient  # noqa: E402
from forecasting.registry import MRTS_SERIES  # noqa: E402
from retailpred.live_loop.ingest import scoring_decision  # noqa: E402
from retailpred.platform.spark_delta import merge_delta_immutable  # noqa: E402
from retailpred.platform.specs import SILVER_MONTHLY  # noqa: E402


def resolve_census_key(secret_scope: str) -> str:
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope=secret_scope, key="census-api-key")


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
    parser.add_argument("--secret-scope", default="retailpred")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument(
        "--end-year", type=int, default=datetime.now(timezone.utc).year
    )
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    client = MRTSClient(resolve_census_key(args.secret_scope))
    ingested_at = datetime.now(timezone.utc)
    frames = [
        client.fetch_range(item, args.start_year, args.end_year)
        for item in MRTS_SERIES
    ]
    batch = pd.concat(frames, ignore_index=True)
    batch["release_time"] = pd.NaT
    batch = batch.loc[:, SILVER_MONTHLY.columns]

    silver_df = spark.createDataFrame(to_spark_ready(batch))
    merge_delta_immutable(spark, silver_df, SILVER_MONTHLY, catalog=args.catalog)
    rows = spark.table(
        f"`{args.catalog}`.`silver`.`monthly_observations`"
    ).count()
    cutoff = batch["observation_time"].max().date().isoformat()

    decision = scoring_decision("mrts")
    print(f"Silver rows in Delta after merge: {rows:,}")
    print(f"Data cutoff: {cutoff}")
    print(
        f"Scoring: {decision['status']} (model={decision['model']}, "
        f"model_status={decision['model_status']}, "
        f"required_action={decision['required_action']})"
    )
    if decision["status"] != "authorized":
        print("No forecasts were generated; gold forecast tables remain untouched.")


if __name__ == "__main__":
    main()
