"""Contract 2.2-status: per-run ingest-status record assembly and guards.

The status record is a status report, not a forecast claim: these tests
pin the refusal projection (the verified hosted-run state), the
fail-closed assembly rules, the macro guards, and the dependency-free
``--dry-run`` path of the publish task.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from retailpred.live_loop.status import (
    FORBIDDEN_CLAIM_KEYS,
    IngestObservation,
    RealizedGate,
    ScoreGate,
    assemble_status,
    clean_macro,
    find_forbidden_claim_keys,
    run_log_line,
    started_at_from_epoch_ms,
    write_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = REPO_ROOT / "databricks" / "jobs" / "publish_github_snapshot.py"

FINISHED = datetime(2026, 11, 10, 22, 30, tzinfo=timezone.utc)
REFUSAL_SCORE = ScoreGate(
    status="refused",
    model="global_lightgbm",
    model_status="development_challenger",
    required_action="unavailable_until_point_in_time_validation",
)


def _refusal_record(**overrides):
    kwargs = dict(
        finished_at=FINISHED,
        is_live=False,
        ingest_gate="success",
        ingest=IngestObservation(rows_in_delta=2200, data_cutoff="2026-08-01", series_count=11),
        score=REFUSAL_SCORE,
        realized=RealizedGate(status="skipped", realized_rows=0),
        publish="declined",
        pipeline_run_id="452582046469351",
        job_id="326179734702237",
        pipeline_started_at=datetime(2026, 11, 4, 14, 30, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return assemble_status(**kwargs)


def test_refusal_projection_matches_the_verified_hosted_run_shape() -> None:
    record = _refusal_record()
    payload = json.loads(record.model_dump_json())
    assert payload["schema_version"] == "2.2-status"
    assert payload["pipeline_run_id"] == "452582046469351"
    assert payload["job_id"] == "326179734702237"
    assert payload["ingest"] == {
        "rows_in_delta": 2200,
        "data_cutoff": "2026-08-01",
        "series_count": 11,
    }
    assert payload["gates"]["ingest"] == "success"
    assert payload["gates"]["score"] == {
        "status": "refused",
        "model": "global_lightgbm",
        "model_status": "development_challenger",
        "required_action": "unavailable_until_point_in_time_validation",
    }
    assert payload["gates"]["realized"] == {"status": "skipped", "realized_rows": 0}
    assert payload["gates"]["publish"] == "declined"
    assert payload["is_live"] is False


def test_failed_ingest_gate_requires_null_observations_not_stale_reads() -> None:
    with pytest.raises(ValueError, match="null observations"):
        _refusal_record(
            ingest_gate="failed",
            ingest=IngestObservation(rows_in_delta=2200, data_cutoff="2026-08-01", series_count=11),
        )
    failed = _refusal_record(ingest_gate="failed", ingest=IngestObservation())
    assert failed.gates.ingest == "failed"
    assert failed.ingest.rows_in_delta is None


def test_successful_ingest_gate_requires_the_complete_triple() -> None:
    with pytest.raises(ValueError, match="complete observation triple"):
        _refusal_record(ingest=IngestObservation(rows_in_delta=2200))


def test_unknown_publish_verdict_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown publish verdict"):
        _refusal_record(publish="maybe")


def test_claim_keys_are_rejected_anywhere_in_the_record() -> None:
    for key in sorted(FORBIDDEN_CLAIM_KEYS):
        assert key in find_forbidden_claim_keys({key: 1})
    assert find_forbidden_claim_keys({"gates": [{"prediction": 5}]}) == ["prediction"]
    assert find_forbidden_claim_keys({"rows_in_delta": 2200}) == []
    smuggled = _refusal_record()
    payload = smuggled.model_dump()
    payload["gates"]["publish"] = "declined"
    # Any attempt to extend the model with a claim key fails on extra=forbid.
    with pytest.raises(Exception):
        type(smuggled).model_validate({**payload, "prediction": 1000.0})


def test_macro_guards_record_null_never_template_text() -> None:
    assert clean_macro(None) is None
    assert clean_macro("") is None
    assert clean_macro("  ") is None
    assert clean_macro("{{job.run_id}}") is None
    assert clean_macro("run {{x}}") is None
    assert clean_macro("452582046469351") == "452582046469351"
    assert started_at_from_epoch_ms("{{start_time}}") is None
    assert started_at_from_epoch_ms("not-a-number") is None
    expected = datetime(2026, 11, 4, 14, 30, tzinfo=timezone.utc)
    epoch_ms = str(int(expected.timestamp() * 1000))
    assert started_at_from_epoch_ms(epoch_ms) == expected
    unresolved = _refusal_record(pipeline_run_id="{{job.run_id}}", job_id="{{job.id}}")
    assert unresolved.pipeline_run_id is None
    assert unresolved.job_id is None


def test_write_status_is_atomic_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "v2-ingest-status.json"
    write_status(_refusal_record(), destination)
    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["schema_version"] == "2.2-status"
    assert not list(destination.parent.glob("*.partial"))


def test_run_log_line_is_one_compact_json_line() -> None:
    line = run_log_line(_refusal_record())
    assert "\n" not in line
    assert json.loads(line)["pipeline_run_id"] == "452582046469351"


def test_publish_script_dry_run_projects_the_refusal_without_writing(tmp_path: Path) -> None:
    fixture = tmp_path / "observations.json"
    fixture.write_text(
        json.dumps(
            {
                "rows_in_delta": 2200,
                "data_cutoff": "2026-08-01",
                "series_count": 11,
                "forecasts_empty": True,
                "realized_rows": 0,
                "publish_verdict": "declined",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PUBLISH_SCRIPT),
            "--catalog",
            "workspace",
            "--dry-run",
            "--dry-run-input",
            str(fixture),
            "--pipeline-started-at-ms",
            "{{start_time}}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "nothing was committed to GitHub" in completed.stdout
    marker = completed.stdout.index("{")
    payload = json.loads(completed.stdout[marker:])
    assert payload["schema_version"] == "2.2-status"
    assert payload["gates"]["publish"] == "declined"
    assert payload["gates"]["score"]["status"] == "refused"
    assert payload["gates"]["score"]["model_status"] == "development_challenger"
    assert payload["is_live"] is False
    assert payload["pipeline_started_at"] is None  # unresolved macro → null
    assert find_forbidden_claim_keys(payload) == []


def test_publish_script_dry_run_rejects_an_unknown_fixture_key(tmp_path: Path) -> None:
    fixture = tmp_path / "observations.json"
    fixture.write_text(
        json.dumps(
            {
                "rows_in_delta": 2200,
                "data_cutoff": "2026-08-01",
                "series_count": 11,
                "forecasts_empty": True,
                "realized_rows": 0,
                "prediction": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PUBLISH_SCRIPT),
            "--catalog",
            "workspace",
            "--dry-run",
            "--dry-run-input",
            str(fixture),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
    assert "unknown keys" in completed.stderr
