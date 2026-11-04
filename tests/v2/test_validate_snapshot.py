import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_snapshot import validate

VALID = {
    "contract_version": "2.1",
    "snapshot_kind": "scheduled_batch_forecasts",
    "dataset": "mrts",
    "use_case": "macro_planning",
    "forecast_origin": "2025-12-01",
    "model_name": "global_lightgbm",
    "model_version": "2.0.0",
    "model_status": "promoted_point_champion",
    "uncertainty_status": "research_only_not_calibrated",
    "is_live": False,
    "pipeline_run_id": None,
    "freshness": {
        "status": "current",
        "generated_at": "2026-01-14T16:00:00+00:00",
        "forecast_vintage_time": "2026-01-14T15:30:00+00:00",
        "data_cutoff": "2025-12-01",
        "next_scheduled_run": None,
        "max_age_hours": 336,
    },
    "series": [
        {
            "series_id": "retail_total",
            "points": [
                {
                    "horizon": 1,
                    "target_time": "2026-01-01",
                    "prediction": 1000.0,
                    "lower": None,
                    "upper": None,
                    "seasonal_naive_prediction": 990.0,
                    "seasonal_naive_source_time": "2025-01-01",
                    "seasonal_naive_source_vintage_time": "2025-02-14T14:00:00+00:00",
                    "seasonal_naive_status": "available",
                    "seasonal_naive_unavailable_reason": None,
                    "evaluation_actual": None,
                    "evaluation_actual_vintage_time": None,
                    "model_absolute_error": None,
                    "seasonal_naive_absolute_error": None,
                    "model_beat_seasonal_naive": None,
                    "evaluation_status": None,
                }
            ],
        }
    ],
    "provenance": {"code_revision": "a" * 40},
}


def test_valid_contract_passes() -> None:
    assert validate(VALID) == []


def test_missing_or_wrong_version_fails() -> None:
    payload = {**VALID, "contract_version": "2.0"}
    assert any("contract_version" in error for error in validate(payload))


def test_unavailable_benchmark_with_prediction_fails() -> None:
    payload = json.loads(json.dumps(VALID))
    payload["series"][0]["points"][0]["seasonal_naive_status"] = "unavailable"
    assert any("unavailable" in error for error in validate(payload))


def test_available_benchmark_without_prediction_fails() -> None:
    payload = json.loads(json.dumps(VALID))
    payload["series"][0]["points"][0]["seasonal_naive_prediction"] = None
    assert any("without a seasonal_naive_prediction" in error for error in validate(payload))


def test_negative_prediction_fails() -> None:
    payload = json.loads(json.dumps(VALID))
    payload["series"][0]["points"][0]["prediction"] = -1.0
    assert any("non-negative number" in error for error in validate(payload))


def test_empty_series_fails() -> None:
    payload = {**VALID, "series": []}
    assert any("series must be a non-empty list" in error for error in validate(payload))


def test_cli_rejects_unreadable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    result = subprocess.run(
        [sys.executable, "scripts/validate_snapshot.py", str(broken)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "unreadable" in result.stdout


def test_cli_accepts_valid_snapshot(tmp_path: Path) -> None:
    good = tmp_path / "v2-latest.json"
    good.write_text(json.dumps(VALID))
    result = subprocess.run(
        [sys.executable, "scripts/validate_snapshot.py", str(good)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
