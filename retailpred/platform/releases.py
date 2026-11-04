"""Census retail release catalog normalization."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd

from retailpred.platform.local_store import validate_table_schema
from retailpred.platform.specs import RETAIL_RELEASES


ALLOWED_PRODUCTS = {"marts", "mrts"}
ALLOWED_KINDS = {"advance", "revised", "annual_revision"}


def normalize_retail_releases(
    frame: pd.DataFrame,
    *,
    source_url: str,
    ingested_at: datetime,
) -> pd.DataFrame:
    required = {"product", "observation_month", "release_time", "release_kind"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"release schedule is missing columns: {sorted(missing)}")
    if ingested_at.tzinfo is None or ingested_at.utcoffset() is None:
        raise ValueError("ingested_at must be timezone-aware")
    result = frame.loc[:, sorted(required)].copy()
    result["product"] = result["product"].astype(str).str.lower()
    result["release_kind"] = result["release_kind"].astype(str).str.lower()
    unknown_products = sorted(set(result["product"]) - ALLOWED_PRODUCTS)
    unknown_kinds = sorted(set(result["release_kind"]) - ALLOWED_KINDS)
    if unknown_products or unknown_kinds:
        raise ValueError(
            f"unsupported release values: products={unknown_products}, kinds={unknown_kinds}"
        )
    result["observation_month"] = pd.to_datetime(
        result["observation_month"], format="mixed", errors="raise"
    )
    normalized_months = result["observation_month"].dt.to_period("M").dt.to_timestamp()
    if not result["observation_month"].equals(normalized_months):
        raise ValueError("observation_month must be the first day of the month")
    result["release_time"] = pd.to_datetime(
        result["release_time"], format="mixed", utc=True, errors="raise"
    )
    if (result["release_time"] > pd.Timestamp(ingested_at)).any():
        raise ValueError("release catalog cannot mark a future scheduled date as an actual release")
    result["release_id"] = result.apply(
        lambda row: hashlib.sha256(
            (
                f"{row['product']}|{row['observation_month'].date()}|"
                f"{row['release_time'].isoformat()}|{row['release_kind']}"
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    result["source_url"] = source_url
    result["ingested_at"] = ingested_at
    return validate_table_schema(result.loc[:, RETAIL_RELEASES.columns], RETAIL_RELEASES)


def attach_release_time(
    observations: pd.DataFrame,
    releases: pd.DataFrame,
    *,
    product: str,
    release_kind: str,
) -> pd.DataFrame:
    catalog = releases.loc[
        releases["product"].eq(product) & releases["release_kind"].eq(release_kind),
        ["observation_month", "release_time"],
    ]
    if catalog["observation_month"].duplicated().any():
        raise ValueError("release catalog has multiple matching releases for one month")
    result = observations.drop(columns=["release_time"], errors="ignore").merge(
        catalog,
        left_on="observation_time",
        right_on="observation_month",
        how="left",
        validate="many_to_one",
    )
    if result["release_time"].isna().any():
        raise ValueError("release catalog does not cover every observation month")
    return result.drop(columns="observation_month")
