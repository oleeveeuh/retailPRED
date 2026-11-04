from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from forecasting.registry import MRTS_SERIES
from retailpred.live_loop.ingest import (
    ScoringNotAuthorized,
    merge_latest_observations,
    run_live_ingest,
    scoring_decision,
)
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS, SILVER_MONTHLY

MONTHS = pd.date_range("2024-01-01", periods=6, freq="MS")


class FakeMRTSClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.vintage_time = pd.Timestamp("2025-11-15T14:35:00Z")
        self._offsets: dict[str, float] = {}

    def fetch_range(self, series, start_year: int, end_year: int) -> pd.DataFrame:
        self.calls.append((series.series_id, start_year, end_year))
        offset = self._offsets.setdefault(series.series_id, float(len(self._offsets)))
        return pd.DataFrame(
            {
                "series_id": series.series_id,
                "observation_time": MONTHS,
                "value": [100.0 + index + offset for index in range(len(MONTHS))],
                "vintage_time": self.vintage_time,
                "ingested_at": self.vintage_time,
                "source": "census_mrts",
            }
        )


def test_merge_latest_observations_is_idempotent(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    client = FakeMRTSClient()
    first = merge_latest_observations(store, client, start_year=2024, end_year=2024)
    second = merge_latest_observations(store, client, start_year=2024, end_year=2024)
    assert len(first) == len(second) == len(MRTS_SERIES) * len(MONTHS)
    assert len(client.calls) == 2 * len(MRTS_SERIES)
    silver = store.read(SILVER_MONTHLY)
    assert silver["release_time"].isna().all()


def test_run_live_ingest_refuses_scoring_while_challenger(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    outcome = run_live_ingest(
        store,
        FakeMRTSClient(),
        start_year=2024,
        end_year=2024,
        now=datetime(2025, 11, 15, tzinfo=timezone.utc),
        revision="0" * 40,
    )
    assert outcome.scoring["status"] == "refused"
    assert outcome.scoring["required_action"] == "unavailable_until_point_in_time_validation"
    assert outcome.gold_rows == 0
    assert outcome.silver_rows == len(MRTS_SERIES) * len(MONTHS)
    assert outcome.data_cutoff == "2024-06-01"
    assert store.read(GOLD_FORECASTS).empty


def test_scoring_decision_carries_frozen_state() -> None:
    decision = scoring_decision("mrts")
    assert decision["model"] == "global_lightgbm"
    assert decision["model_status"] == "development_challenger"
    assert decision["status"] == "refused"


def test_score_gold_panel_raises_without_authorization(tmp_path) -> None:
    from retailpred.live_loop.ingest import score_gold_panel

    store = LocalTableStore(tmp_path)
    with pytest.raises(ScoringNotAuthorized) as excinfo:
        score_gold_panel(
            store,
            now=datetime(2025, 11, 15, tzinfo=timezone.utc),
            revision="0" * 40,
        )
    assert "unavailable_until_point_in_time_validation" in str(excinfo.value)


def test_bundle_declares_paused_scheduled_ingest() -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = yaml.safe_load((root / "databricks.yml").read_text())
    assert bundle["variables"]["secret_scope"]["default"] == "retailpred"
    resources = yaml.safe_load(
        (root / "databricks/resources/ingest_latest.job.yml").read_text()
    )
    job = resources["resources"]["jobs"]["platform_ingest_latest"]
    assert job["schedule"]["pause_status"] == "PAUSED"
    assert job["schedule"]["timezone_id"] == "America/New_York"
    task = job["tasks"][0]
    assert task["spark_python_task"]["python_file"] == "../jobs/ingest_latest.py"
    assert "--secret-scope" in task["spark_python_task"]["parameters"]
    assert len(job["tasks"]) == 2
    second = job["tasks"][1]
    assert second["task_key"] == "score_realized_errors"
    assert second["spark_python_task"]["python_file"] == "../jobs/realized_errors.py"
    assert second["depends_on"] == [{"task_key": "pull_latest_census_release"}]
