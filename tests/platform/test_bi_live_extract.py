from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from bi.contracts import CONTRACTS
from forecasting.portfolio import model_decision
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS

ORIGIN = pd.Timestamp("2025-12-01")
VINTAGE = pd.Timestamp("2026-01-14T15:30:00Z")


def _populate_store(tmp_path: Path) -> LocalTableStore:
    store = LocalTableStore(tmp_path)
    decision = model_decision("mrts")
    records = []
    for series in ("retail_total", "general_merchandise"):
        for horizon in (1, 2):
            records.append(
                {
                    "dataset": "mrts",
                    "series_id": series,
                    "forecast_origin": ORIGIN,
                    "target_time": ORIGIN + pd.offsets.MonthBegin(horizon),
                    "horizon": horizon,
                    "model_name": decision.point_model,
                    "model_version": "2.0.0",
                    "forecast_vintage_time": VINTAGE,
                    "prediction": 500.0 - horizon,
                    "lower": None,
                    "upper": None,
                    "data_cutoff": ORIGIN,
                    "origin_as_of": VINTAGE,
                    "uncertainty_status": decision.uncertainty_status,
                    "seasonal_naive_prediction": 495.0 - horizon,
                    "seasonal_naive_source_time": ORIGIN
                    + pd.offsets.MonthBegin(horizon)
                    - pd.DateOffset(months=12),
                    "seasonal_naive_source_vintage_time": VINTAGE,
                    "seasonal_naive_status": "available",
                    "seasonal_naive_unavailable_reason": None,
                    "generated_at": VINTAGE,
                    "code_revision": "b" * 40,
                }
            )
    store.replace_snapshot(
        GOLD_FORECASTS, pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)
    )
    return store


def test_live_forecasts_extract_validates_against_contract(tmp_path) -> None:
    from scripts.build_bi_extracts import _live_forecasts

    _populate_store(tmp_path)
    frame = _live_forecasts(tmp_path)
    validated = CONTRACTS["live_forecasts.csv"].validate(frame)
    assert len(validated) == 4
    assert validated["model_status"].unique().tolist() == ["development_challenger"]
    assert (validated["freshness_status"] == "scheduled_batch_snapshot").all()


def test_live_forecasts_extract_refuses_empty_store(tmp_path) -> None:
    from scripts.build_bi_extracts import _live_forecasts

    LocalTableStore(tmp_path).read(GOLD_FORECASTS)
    with pytest.raises(SystemExit, match="live_forecasts.csv is unavailable"):
        _live_forecasts(tmp_path)
