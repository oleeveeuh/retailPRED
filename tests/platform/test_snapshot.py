from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from retailpred.platform.snapshot import (
    build_snapshot,
    is_stale,
    write_snapshot,
)
from retailpred.platform.specs import GOLD_FORECASTS

ORIGIN = pd.Timestamp("2025-12-01")
VINTAGE = datetime(2026, 1, 14, 15, 30, tzinfo=timezone.utc)
SERIES = ("general_merchandise", "retail_total")


def _forecast_frame(**overrides) -> pd.DataFrame:
    records = []
    for series in SERIES:
        for horizon in (1, 2, 3):
            records.append(
                {
                    "dataset": "mrts",
                    "series_id": series,
                    "forecast_origin": ORIGIN,
                    "target_time": ORIGIN + pd.offsets.MonthBegin(horizon),
                    "horizon": horizon,
                    "model_name": "global_lightgbm",
                    "model_version": "2.0.0",
                    "forecast_vintage_time": pd.Timestamp(VINTAGE),
                    "prediction": 1000.0 - horizon,
                    "lower": None,
                    "upper": None,
                    "data_cutoff": ORIGIN,
                    "uncertainty_status": "research_only_not_calibrated",
                    "generated_at": pd.Timestamp(VINTAGE),
                    "code_revision": "a" * 40,
                }
            )
    frame = pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)
    for key, value in overrides.items():
        frame[key] = value
    return frame


def _generated_at() -> datetime:
    return datetime(2026, 1, 14, 16, 0, tzinfo=timezone.utc)


def test_build_snapshot_produces_publishable_contract() -> None:
    snapshot = build_snapshot(
        _forecast_frame(),
        generated_at=_generated_at(),
        next_scheduled_run=datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc),
        provenance={"code_revision": "a" * 40},
    )
    assert snapshot.contract_version == "2.0"
    assert snapshot.snapshot_kind == "scheduled_batch_forecasts"
    assert snapshot.dataset == "mrts"
    assert snapshot.is_live is True
    assert snapshot.freshness.status == "current"
    assert snapshot.freshness.data_cutoff == ORIGIN.date()
    assert [item.series_id for item in snapshot.series] == sorted(SERIES)
    first = snapshot.series[0].points[0]
    assert first.horizon == 1 and first.prediction >= 0
    assert first.lower is None and first.upper is None
    assert snapshot.provenance["code_revision"] == "a" * 40


def test_snapshot_refuses_mixed_models_origins_and_datasets() -> None:
    with pytest.raises(ValueError, match="mix model versions"):
        build_snapshot(
            _forecast_frame(model_version=["2.0.0", "9.9.9"] * 3),
            generated_at=_generated_at(),
        )
    with pytest.raises(ValueError, match="mix forecast origins"):
        build_snapshot(
            _forecast_frame(
                forecast_origin=[ORIGIN] * 3 + [ORIGIN + pd.offsets.MonthBegin(1)] * 3
            ),
            generated_at=_generated_at(),
        )
    with pytest.raises(ValueError, match="mix datasets"):
        build_snapshot(_forecast_frame(dataset=["mrts", "m5"] * 3), generated_at=_generated_at())


def test_snapshot_refuses_unauthorized_uncertainty_bounds() -> None:
    frame = _forecast_frame()
    frame["lower"] = frame["prediction"] - 5.0
    with pytest.raises(ValueError, match="does not authorize"):
        build_snapshot(frame, generated_at=_generated_at())


def test_stale_freshness_after_max_age() -> None:
    assert not is_stale(_generated_at(), VINTAGE, max_age_hours=24)
    assert is_stale(_generated_at() + timedelta(hours=25), VINTAGE, max_age_hours=24)
    late = build_snapshot(
        _forecast_frame(),
        generated_at=VINTAGE + timedelta(hours=400),
        max_age_hours=336,
    )
    assert late.freshness.status == "stale"


def test_empty_forecasts_are_refused() -> None:
    with pytest.raises(ValueError, match="snapshot publishing is unavailable"):
        build_snapshot(
            pd.DataFrame(columns=GOLD_FORECASTS.columns),
            generated_at=_generated_at(),
        )


def test_write_snapshot_round_trip(tmp_path) -> None:
    snapshot = build_snapshot(_forecast_frame(), generated_at=_generated_at())
    destination = write_snapshot(snapshot, tmp_path / "snapshots" / "v2-latest.json")
    payload = destination.read_text(encoding="utf-8")
    assert '"contract_version": "2.0"' in payload
    assert '"snapshot_kind": "scheduled_batch_forecasts"' in payload
    assert '"is_live": true' in payload
