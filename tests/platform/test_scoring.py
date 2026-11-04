from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.live_loop.ingest import run_live_ingest
from retailpred.live_loop.scoring import (
    ScoringNotAuthorized,
    score_from_panel,
    score_store,
    scoring_decision,
)
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS, SILVER_MONTHLY

from test_live_ingest import FakeMRTSClient

NOW = datetime(2026, 1, 14, 16, 0, tzinfo=timezone.utc)
AUTHORIZED = {
    "status": "authorized",
    "model": "global_lightgbm",
    "model_status": "promoted_point_champion",
    "required_action": "scheduled_batch_on_databricks",
    "uncertainty_status": "research_only_not_calibrated",
}


def _silver_panel() -> pd.DataFrame:
    months = pd.date_range("2023-01-01", periods=36, freq="MS")
    records = []
    for series, base in (("retail_total", 100.0), ("general_merchandise", 60.0)):
        for index, month in enumerate(months):
            seasonal = 1.0 + 0.04 * ((index % 12) - 6) / 6
            records.append(
                {
                    "series_id": series,
                    "observation_time": month,
                    "value": base * seasonal + index,
                    "release_time": pd.NaT,
                    "vintage_time": month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=15),
                    "ingested_at": month + pd.offsets.MonthEnd(0) + pd.Timedelta(days=16),
                    "source": "census_mrts",
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.fixture
def authorized(monkeypatch):
    monkeypatch.setattr(
        "retailpred.live_loop.scoring.scoring_decision", lambda dataset: AUTHORIZED
    )


def test_scoring_decision_refuses_while_challenger() -> None:
    decision = scoring_decision("mrts")
    assert decision["model"] == "global_lightgbm"
    assert decision["model_status"] == "development_challenger"
    assert decision["status"] == "refused"
    assert decision["required_action"] == "unavailable_until_point_in_time_validation"


def test_score_from_panel_raises_without_authorization() -> None:
    with pytest.raises(ScoringNotAuthorized) as excinfo:
        score_from_panel(_silver_panel(), now=NOW, revision="0" * 40)
    assert "unavailable_until_point_in_time_validation" in str(excinfo.value)


def test_score_store_refusal_leaves_gold_untouched(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver_panel())
    outcome = score_store(store, now=NOW, revision="0" * 40)
    assert outcome.forecasts is None
    assert outcome.decision["status"] == "refused"
    assert store.read(GOLD_FORECASTS).empty


def test_authorized_scoring_writes_versioned_benchmarked_forecasts(
    tmp_path, authorized
) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver_panel())
    outcome = score_store(store, now=NOW, revision="0" * 40)
    forecasts = outcome.forecasts
    assert forecasts is not None
    assert len(forecasts) == 24  # 2 series x 12 horizons
    assert (forecasts["seasonal_naive_status"] == "available").all()
    january = forecasts.loc[forecasts["target_time"] == pd.Timestamp("2026-01-01")].iloc[0]
    assert pd.Timestamp(january["seasonal_naive_source_time"]) == pd.Timestamp("2025-01-01")
    assert forecasts["origin_as_of"].nunique() == 1
    stored = (
        store.read(GOLD_FORECASTS)
        .sort_values(list(GOLD_FORECASTS.key))
        .reset_index(drop=True)
    )
    ordered = forecasts.sort_values(list(GOLD_FORECASTS.key)).reset_index(drop=True)
    assert len(stored) == len(forecasts)
    assert stored["prediction"].tolist() == ordered["prediction"].tolist()
    assert stored["seasonal_naive_status"].tolist() == ordered["seasonal_naive_status"].tolist()
    assert stored["target_time"].tolist() == ordered["target_time"].tolist()


def test_rerunning_the_same_scoring_run_does_not_duplicate(tmp_path, authorized) -> None:
    store = LocalTableStore(tmp_path)
    store.merge(SILVER_MONTHLY, _silver_panel())
    first = score_store(store, now=NOW, revision="0" * 40)
    second = score_store(store, now=NOW, revision="0" * 40)
    assert first.forecasts is not None and second.forecasts is not None
    assert first.forecasts.equals(second.forecasts)
    assert len(store.read(GOLD_FORECASTS)) == 24


def test_ingest_then_score_refuses_on_the_frozen_decision(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    run_live_ingest(
        store,
        FakeMRTSClient(),
        start_year=2024,
        end_year=2024,
    )
    outcome = score_store(store, now=NOW, revision="0" * 40)
    assert outcome.forecasts is None
    assert outcome.decision["required_action"] == "unavailable_until_point_in_time_validation"
    assert store.read(GOLD_FORECASTS).empty
