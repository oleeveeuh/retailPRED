import json
import subprocess
import sys
from pathlib import Path

from scripts.validate_snapshot import validate, validate_runs_log, validate_status_record

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


STATUS_REFUSAL = {
    "schema_version": "2.2-status",
    "pipeline_run_id": "452582046469351",
    "job_id": "326179734702237",
    "pipeline_started_at": "2026-11-04T22:29:51+00:00",
    "finished_at": "2026-11-04T22:33:12+00:00",
    "ingest": {"rows_in_delta": 2200, "data_cutoff": "2026-08-01", "series_count": 11},
    "gates": {
        "ingest": "success",
        "score": {
            "status": "refused",
            "model": "global_lightgbm",
            "model_status": "development_challenger",
            "required_action": "unavailable_until_point_in_time_validation",
        },
        "realized": {"status": "skipped", "realized_rows": 0},
        "publish": "declined",
    },
    "is_live": False,
    "generated_at": "2026-11-04T22:33:12+00:00",
}


def test_valid_status_record_passes() -> None:
    assert validate_status_record(STATUS_REFUSAL, "status.json") == []


def test_status_record_requires_every_gate_field() -> None:
    for key in ("ingest", "score", "realized", "publish"):
        broken = json.loads(json.dumps(STATUS_REFUSAL))
        del broken["gates"][key]
        assert any(f"gates.{key}" in error for error in validate_status_record(broken, "s"))
    broken = json.loads(json.dumps(STATUS_REFUSAL))
    del broken["ingest"]
    assert any("ingest must be an object" in error for error in validate_status_record(broken, "s"))


def test_status_record_makes_no_forecast_claims() -> None:
    for key in ("prediction", "forecasts", "wape", "model_absolute_error"):
        smuggled = json.loads(json.dumps(STATUS_REFUSAL))
        smuggled[key] = 1000.0
        errors = validate_status_record(smuggled, "s")
        assert errors and "forbidden forecast-claim keys" in errors[-1]
        assert key in errors[-1]
    nested = json.loads(json.dumps(STATUS_REFUSAL))
    nested["ingest"]["series"] = ["retail_total"]
    errors = validate_status_record(nested, "s")
    assert any("forbidden forecast-claim keys" in error for error in errors)


def test_status_record_is_live_requires_completed_publication() -> None:
    premature = {**STATUS_REFUSAL, "is_live": True}
    errors = validate_status_record(premature, "s")
    assert any("is_live=true requires" in error for error in errors)
    completed = json.loads(json.dumps(STATUS_REFUSAL))
    completed["is_live"] = True
    completed["gates"]["publish"] = "published"
    assert validate_status_record(completed, "s") == []


def test_runs_log_validates_every_line(tmp_path: Path) -> None:
    log = tmp_path / "runs.jsonl"
    log.write_text(
        json.dumps(STATUS_REFUSAL) + "\n" + json.dumps({**STATUS_REFUSAL, "is_live": "no"}) + "\n",
        encoding="utf-8",
    )
    errors = validate_runs_log(log.read_text(encoding="utf-8"), str(log))
    assert any("line 2" in error and "is_live" in error for error in errors)
    assert validate_runs_log(json.dumps(STATUS_REFUSAL) + "\n", "log.jsonl") == []
    assert any("empty" in error for error in validate_runs_log("\n", "log.jsonl"))


def test_cli_passes_with_status_but_no_snapshot(tmp_path: Path) -> None:
    """A deploy with a status file but no snapshot is the honest fail-closed state."""
    status = tmp_path / "v2-ingest-status.json"
    status.write_text(json.dumps(STATUS_REFUSAL))
    result = subprocess.run(
        [sys.executable, "scripts/validate_snapshot.py", str(status)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    both = subprocess.run(
        [
            sys.executable,
            "scripts/validate_snapshot.py",
            str(status),
            str(tmp_path / "v2-latest.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert both.returncode == 1
    assert "unreadable" in both.stdout
