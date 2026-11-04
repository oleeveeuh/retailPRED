from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from forecasting.registry import MRTS_SERIES
from retailpred.live_loop.ingest import (
    merge_latest_observations,
    run_live_ingest,
)
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import BRONZE_MRTS, SILVER_MONTHLY

MONTHS = pd.date_range("2024-01-01", periods=6, freq="MS")


RAW_PAYLOAD = [
    [
        "data_type_code",
        "time_slot_id",
        "seasonally_adj",
        "category_code",
        "cell_value",
        "error_data",
        "time",
    ],
    ["SM", "1", "yes", "443", "7004", "", "2024"],
    ["SM", "1", "no", "443", "6549", "", "2024"],
]


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

    def cached_year_payloads(self):
        return {2024: (RAW_PAYLOAD, self.vintage_time.to_pydatetime())}


def test_merge_latest_observations_is_idempotent(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    client = FakeMRTSClient()
    first = merge_latest_observations(store, client, start_year=2024, end_year=2024)
    second = merge_latest_observations(store, client, start_year=2024, end_year=2024)
    assert len(first) == len(second) == len(MRTS_SERIES) * len(MONTHS)
    assert len(client.calls) == 2 * len(MRTS_SERIES)
    silver = store.read(SILVER_MONTHLY)
    assert silver["release_time"].isna().all()


def test_run_live_ingest_reports_ingestion_only(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    outcome = run_live_ingest(
        store,
        FakeMRTSClient(),
        start_year=2024,
        end_year=2024,
        now=datetime(2025, 11, 15, 15, 0, tzinfo=timezone.utc),
    )
    assert outcome.silver_rows == len(MRTS_SERIES) * len(MONTHS)
    assert outcome.bronze_rows == 2
    assert outcome.data_cutoff == "2024-06-01"


def test_run_live_ingest_captures_raw_bronze_provenance(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    run_live_ingest(
        store,
        FakeMRTSClient(),
        start_year=2024,
        end_year=2024,
        now=datetime(2025, 11, 15, 15, 0, tzinfo=timezone.utc),
    )
    bronze = store.read(BRONZE_MRTS)
    assert len(bronze) == 2
    assert bronze["extract_id"].str.startswith("census-live-20251115T143500").all()
    assert bronze["release_time"].isna().all()

    # Replaying the identical run is idempotent: same retrieval time, same
    # content hash, no new bronze rows.
    run_live_ingest(
        store,
        FakeMRTSClient(),
        start_year=2024,
        end_year=2024,
        now=datetime(2025, 11, 15, 15, 0, tzinfo=timezone.utc),
    )
    assert len(store.read(BRONZE_MRTS)) == 2


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
    assert "existing_cluster_id" not in task
    assert task["environment_key"] == "serverless_default"
    assert len(job["tasks"]) == 4
    task_keys = [task["task_key"] for task in job["tasks"]]
    assert task_keys == [
        "pull_latest_census_release",
        "score_promoted_model",
        "evaluate_realized_errors",
        "publish_snapshot_github",
    ]
    score_task = job["tasks"][1]
    assert score_task["spark_python_task"]["python_file"] == "../jobs/score_latest.py"
    realized_task = job["tasks"][2]
    assert realized_task["spark_python_task"]["python_file"] == "../jobs/realized_errors.py"
    assert realized_task["depends_on"] == [
        {"task_key": "pull_latest_census_release"},
        {"task_key": "score_promoted_model"},
    ]
    publish_task = job["tasks"][3]
    assert publish_task["spark_python_task"]["python_file"] == (
        "../jobs/publish_github_snapshot.py"
    )
    assert publish_task["depends_on"] == [{"task_key": "evaluate_realized_errors"}]
