"""MRTS bronze-to-silver transforms shared by local and Spark implementations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from forecasting.data.mrts import SEASONAL_ADJUSTED_VALUES
from forecasting.registry import MRTS_BY_CODE
from retailpred.platform.context import ExtractContext
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.quality import validate_silver_monthly
from retailpred.platform.specs import BRONZE_MRTS, SILVER_MONTHLY


def bronze_from_api_payload(
    payload: list[list[Any]], context: ExtractContext
) -> pd.DataFrame:
    if len(payload) < 2:
        raise ValueError("MRTS payload has no data rows")
    header, *rows = payload
    records: list[dict[str, object]] = []
    for row in rows:
        raw = dict(zip(header, row, strict=True))
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        record_id = hashlib.sha256(
            f"{context.extract_id}|{canonical}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_record_id": record_id,
                "extract_id": context.extract_id,
                "source_url": context.source_url,
                "data_type_code": raw.get("data_type_code"),
                "time_slot_id": raw.get("time_slot_id"),
                "seasonally_adj": raw.get("seasonally_adj"),
                "category_code": raw.get("category_code"),
                "cell_value": raw.get("cell_value"),
                "error_data": raw.get("error_data") or None,
                "year": raw.get("time"),
                "release_time": context.release_time,
                "vintage_time": context.vintage_time,
                "ingested_at": context.ingested_at,
            }
        )
    return pd.DataFrame.from_records(records, columns=BRONZE_MRTS.columns)


def silver_from_bronze(bronze: pd.DataFrame) -> pd.DataFrame:
    selected = bronze.loc[
        bronze["data_type_code"].eq("SM")
        & bronze["seasonally_adj"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(SEASONAL_ADJUSTED_VALUES)
        & bronze["category_code"].astype(str).isin(MRTS_BY_CODE)
    ].copy()
    selected["month"] = pd.to_numeric(selected["time_slot_id"], errors="coerce")
    selected["year_number"] = pd.to_numeric(selected["year"], errors="coerce")
    selected["value"] = pd.to_numeric(selected["cell_value"], errors="coerce")
    selected = selected.loc[
        selected["month"].between(1, 12)
        & selected["year_number"].notna()
        & selected["value"].notna()
    ]
    if selected.empty:
        raise ValueError("MRTS extract contains no registered seasonally adjusted monthly sales rows")
    observations = pd.DataFrame(
        {
            "series_id": selected["category_code"].map(
                lambda code: MRTS_BY_CODE[str(code)].series_id
            ),
            "observation_time": pd.to_datetime(
                {
                    "year": selected["year_number"].astype(int),
                    "month": selected["month"].astype(int),
                    "day": 1,
                }
            ),
            "value": selected["value"].astype(float),
            "release_time": selected["release_time"],
            "vintage_time": selected["vintage_time"],
            "ingested_at": selected["ingested_at"],
            "source": "census_mrts",
        }
    )
    clean, _ = validate_silver_monthly(observations)
    return clean.loc[:, SILVER_MONTHLY.columns]


def ingest_extract(
    store: LocalTableStore,
    payload: list[list[Any]],
    context: ExtractContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bronze_batch = bronze_from_api_payload(payload, context)
    bronze = store.merge(BRONZE_MRTS, bronze_batch)
    silver_batch = silver_from_bronze(bronze_batch)
    silver = store.merge(SILVER_MONTHLY, silver_batch)
    return bronze, silver
