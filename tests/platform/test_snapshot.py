from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from forecasting.portfolio import ModelPortfolioDecision
from retailpred.platform.snapshot import (
    build_snapshot,
    is_stale,
    write_snapshot,
)
from retailpred.platform.specs import GOLD_FORECASTS

ORIGIN = pd.Timestamp("2025-12-01")
VINTAGE = datetime(2026, 1, 14, 15, 30, tzinfo=timezone.utc)
SERIES = ("general_merchandise", "retail_total")

AUTHORIZED_DECISION = ModelPortfolioDecision(
    use_case="macro_planning",
    dataset="mrts",
    point_model="global_lightgbm",
    point_status="promoted_point_champion",
    serving_mode="scheduled_batch_on_databricks",
    uncertainty_status="research_only_not_calibrated",
    evaluation_cutoff="2025-12-01",
    primary_metric="wape",
    measured_value=0.04,
    evidence=("docs/v2/M3_LIGHTGBM.md",),
)


def _forecast_frame(**overrides) -> pd.DataFrame:
    records = []
    for series in SERIES:
        for horizon in (1, 2, 3):
            naive = 990.0 + horizon if series == "retail_total" else None
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
                    "origin_as_of": pd.Timestamp(VINTAGE),
                    "uncertainty_status": "research_only_not_calibrated",
                    "seasonal_naive_prediction": naive,
                    "seasonal_naive_source_time": ORIGIN
                    + pd.offsets.MonthBegin(horizon)
                    - pd.DateOffset(months=12),
                    "seasonal_naive_source_vintage_time": pd.Timestamp(VINTAGE),
                    "seasonal_naive_status": "available" if naive is not None else "unavailable",
                    "seasonal_naive_unavailable_reason": None
                    if naive is not None
                    else "prior_year_observation_missing",
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


def test_build_snapshot_carries_benchmark_and_live_flag() -> None:
    snapshot = build_snapshot(
        _forecast_frame(),
        generated_at=_generated_at(),
        decision=AUTHORIZED_DECISION,
        pipeline_run_id="run-123",
        next_scheduled_run=datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc),
        provenance={"code_revision": "a" * 40},
    )
    assert snapshot.contract_version == "2.1"
    assert snapshot.snapshot_kind == "scheduled_batch_forecasts"
    assert snapshot.is_live is False
    assert snapshot.pipeline_run_id == "run-123"
    assert snapshot.freshness.data_cutoff == ORIGIN.date()
    assert [item.series_id for item in snapshot.series] == sorted(SERIES)
    naive_point = snapshot.series[1].points[0]
    assert naive_point.seasonal_naive_status == "available"
    assert naive_point.seasonal_naive_prediction == pytest.approx(991.0)
    assert naive_point.seasonal_naive_source_time == date(2025, 1, 1)
    assert naive_point.seasonal_naive_source_vintage_time is not None
    missing_point = snapshot.series[0].points[0]
    assert missing_point.seasonal_naive_status == "unavailable"
    assert missing_point.seasonal_naive_prediction is None
    assert missing_point.seasonal_naive_unavailable_reason == "prior_year_observation_missing"


def test_snapshot_refuses_the_unauthorized_frozen_challenger() -> None:
    with pytest.raises(ValueError, match="authorized champion"):
        build_snapshot(_forecast_frame(), generated_at=_generated_at())


def test_snapshot_joins_realized_evaluation_when_present() -> None:
    key = ["dataset", "series_id", "forecast_origin", "target_time", "model_name", "model_version", "forecast_vintage_time"]
    realized = pd.DataFrame(
        [
            {
                **{column: value for column, value in zip(key, _forecast_frame().iloc[0][key])},
                "evaluation_actual": 1002.0,
                "evaluation_actual_vintage_time": datetime(2026, 2, 14, tzinfo=timezone.utc),
                "model_absolute_error": 3.0,
                "seasonal_naive_absolute_error": 11.0,
                "model_beat_seasonal_naive": True,
                "evaluation_status": "first_release",
            }
        ]
    )
    snapshot = build_snapshot(
        _forecast_frame(),
        generated_at=_generated_at(),
        decision=AUTHORIZED_DECISION,
        realized=realized,
    )
    evaluated = snapshot.series[0].points[0]  # iloc[0] is general_merchandise
    assert evaluated.evaluation_actual == pytest.approx(1002.0)
    assert evaluated.model_absolute_error == pytest.approx(3.0)
    assert evaluated.model_beat_seasonal_naive is True
    assert evaluated.evaluation_status == "first_release"
    unevaluated = snapshot.series[0].points[1]
    assert unevaluated.evaluation_actual is None
    assert unevaluated.evaluation_status is None


def test_snapshot_refuses_mixed_models_origins_and_datasets() -> None:
    with pytest.raises(ValueError, match="mix model versions"):
        build_snapshot(
            _forecast_frame(model_version=["2.0.0", "9.9.9"] * 3),
            generated_at=_generated_at(),
            decision=AUTHORIZED_DECISION,
        )
    with pytest.raises(ValueError, match="mix forecast origins"):
        build_snapshot(
            _forecast_frame(
                forecast_origin=[ORIGIN] * 3 + [ORIGIN + pd.offsets.MonthBegin(1)] * 3
            ),
            generated_at=_generated_at(),
            decision=AUTHORIZED_DECISION,
        )
    with pytest.raises(ValueError, match="mix datasets"):
        build_snapshot(
            _forecast_frame(dataset=["mrts", "m5"] * 3),
            generated_at=_generated_at(),
            decision=AUTHORIZED_DECISION,
        )


def test_snapshot_refuses_unauthorized_uncertainty_bounds() -> None:
    frame = _forecast_frame()
    frame["lower"] = frame["prediction"] - 5.0
    with pytest.raises(ValueError, match="does not authorize"):
        build_snapshot(frame, generated_at=_generated_at(), decision=AUTHORIZED_DECISION)


def test_stale_freshness_after_max_age() -> None:
    assert not is_stale(_generated_at(), VINTAGE, max_age_hours=24)
    assert is_stale(_generated_at() + timedelta(hours=25), VINTAGE, max_age_hours=24)
    late = build_snapshot(
        _forecast_frame(),
        generated_at=VINTAGE + timedelta(hours=400),
        max_age_hours=336,
        decision=AUTHORIZED_DECISION,
    )
    assert late.freshness.status == "stale"


def test_empty_forecasts_are_refused() -> None:
    with pytest.raises(ValueError, match="snapshot publishing is unavailable"):
        build_snapshot(
            pd.DataFrame(columns=GOLD_FORECASTS.columns),
            generated_at=_generated_at(),
            decision=AUTHORIZED_DECISION,
        )


def test_write_snapshot_is_atomic_and_round_trips(tmp_path) -> None:
    snapshot = build_snapshot(
        _forecast_frame(), generated_at=_generated_at(), decision=AUTHORIZED_DECISION
    )
    destination = write_snapshot(snapshot, tmp_path / "snapshots" / "v2-latest.json")
    payload = destination.read_text(encoding="utf-8")
    assert '"contract_version": "2.1"' in payload
    assert '"snapshot_kind": "scheduled_batch_forecasts"' in payload
    assert '"is_live": false' in payload
    assert not (tmp_path / "snapshots" / "v2-latest.json.partial").exists()
