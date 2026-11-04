"""FRED/ALFRED bronze-to-silver vintage transforms."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from forecasting.data.monthly_dataset import validate_monthly_observations
from retailpred.platform.context import ExtractContext
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import BRONZE_FRED, SILVER_ECONOMIC


def bronze_from_fred_payload(
    payload: dict[str, Any], series_id: str, context: ExtractContext
) -> pd.DataFrame:
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("FRED payload has no observation rows")
    records: list[dict[str, object]] = []
    for row in observations:
        try:
            realtime_start = date.fromisoformat(str(row["realtime_start"]))
            realtime_end = date.fromisoformat(str(row["realtime_end"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("FRED observation has an invalid real-time period") from exc
        requested_vintage = context.vintage_time.date()
        if not realtime_start <= requested_vintage <= realtime_end:
            raise ValueError(
                "FRED observation real-time period does not contain the requested vintage"
            )
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        record_id = hashlib.sha256(
            f"{context.extract_id}|{series_id}|{canonical}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_record_id": record_id,
                "extract_id": context.extract_id,
                "series_id": series_id,
                "source_url": context.source_url,
                "observation_date": row.get("date"),
                "raw_value": row.get("value"),
                "realtime_start": realtime_start.isoformat(),
                "realtime_end": realtime_end.isoformat(),
                "vintage_time": context.vintage_time,
                "ingested_at": context.ingested_at,
            }
        )
    return pd.DataFrame.from_records(records, columns=BRONZE_FRED.columns)


def silver_from_fred_bronze(bronze: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(bronze["raw_value"].replace(".", np.nan), errors="coerce")
    selected = bronze.loc[values.notna()].copy()
    selected["value"] = values.loc[values.notna()].astype(float)
    if selected.empty:
        raise ValueError("FRED extract contains no numeric observations")
    observations = pd.DataFrame(
        {
            "series_id": selected["series_id"],
            "observation_time": pd.to_datetime(selected["observation_date"]),
            "value": selected["value"],
            "release_time": pd.NaT,
            "vintage_time": selected["vintage_time"],
            "ingested_at": selected["ingested_at"],
            "source": "fred_alfred",
        }
    )
    clean = validate_monthly_observations(observations)
    return clean.loc[:, SILVER_ECONOMIC.columns]


def ingest_fred_extract(
    store: LocalTableStore,
    payload: dict[str, Any],
    series_id: str,
    context: ExtractContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bronze_batch = bronze_from_fred_payload(payload, series_id, context)
    bronze = store.merge(BRONZE_FRED, bronze_batch)
    silver_batch = silver_from_fred_bronze(bronze_batch)
    silver = store.merge(SILVER_ECONOMIC, silver_batch)
    return bronze, silver
