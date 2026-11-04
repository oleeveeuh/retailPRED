"""Stable gold contracts for model training and business intelligence."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from forecasting.data.monthly_dataset import observations_as_of
from forecasting.registry import MRTS_SERIES
from retailpred.platform.local_store import validate_table_schema
from retailpred.platform.quality import assert_monthly_continuity, validate_silver_monthly
from retailpred.platform.specs import GOLD_MONTHLY


def build_gold_monthly(silver: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    clean, _ = validate_silver_monthly(silver)
    current = observations_as_of(clean, as_of)
    metadata = pd.DataFrame(
        [
            {
                "series_id": item.series_id,
                "source_code": item.source_code,
                "series_name": item.name,
                "unit": item.unit,
                "seasonal_adjustment": item.seasonal_adjustment,
            }
            for item in MRTS_SERIES
        ]
    )
    gold = current.merge(metadata, on="series_id", how="left", validate="many_to_one")
    gold = gold.rename(columns={"name": "series_name"})
    result = validate_table_schema(gold.loc[:, GOLD_MONTHLY.columns], GOLD_MONTHLY)
    assert_monthly_continuity(result)
    return result
