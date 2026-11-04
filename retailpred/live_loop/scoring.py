"""Governed live scoring: promotion gate, point-in-time gold, benchmark.

Scoring runs only when the frozen portfolio decision authorizes it. Features
are assembled with point-in-time selection at the scoring moment, and every
forecast row carries the vintage-safe seasonal-naive benchmark frozen at the
same origin. Persisting model weights is deliberately avoided: the promoted
model is refit from governed data on each run, so no pickle artifact ever
leaves the training boundary.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from forecasting.benchmarking.seasonal_naive import attach_seasonal_naive_benchmark
from forecasting.data.monthly_dataset import observations_as_of
from forecasting.models.global_lightgbm import GlobalLightGBM
from forecasting.portfolio import model_decision
from retailpred.platform.gold import build_gold_monthly
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import GOLD_FORECASTS, GOLD_MONTHLY, SILVER_MONTHLY

MAX_FORECAST_HORIZON = 12


class ScoringNotAuthorized(RuntimeError):
    """Raised when live scoring is attempted without a portfolio authorization."""

    def __init__(self, decision: dict[str, str]) -> None:
        super().__init__(
            "live scoring is not authorized: "
            f"model_status={decision['model_status']}, "
            f"required_action={decision['required_action']}"
        )
        self.decision = decision


@dataclass(frozen=True)
class ScoringOutcome:
    forecasts: pd.DataFrame | None
    decision: dict[str, str]
    data_cutoff: str


def code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def scoring_decision(dataset: str = "mrts") -> dict[str, str]:
    """Whether the frozen portfolio authorizes live scoring, with its reason."""
    decision = model_decision(dataset)
    authorized = decision.point_status == "promoted_point_champion"
    return {
        "status": "authorized" if authorized else "refused",
        "model": decision.point_model,
        "model_status": decision.point_status,
        "required_action": decision.serving_mode,
        "uncertainty_status": decision.uncertainty_status,
    }


def score_from_panel(
    panel: pd.DataFrame,
    *,
    now: datetime,
    revision: str,
    dataset: str = "mrts",
) -> pd.DataFrame:
    """Score the promoted champion on the point-in-time gold panel.

    Refuses with :class:`ScoringNotAuthorized` unless the frozen portfolio
    decision authorizes scoring. Forecasts are written versioned by the
    caller; nothing is overwritten.
    """
    decision = scoring_decision(dataset)
    if decision["status"] != "authorized":
        raise ScoringNotAuthorized(decision)

    gold = build_gold_monthly(panel, as_of=now)
    model = GlobalLightGBM().fit(gold)
    data_cutoff = pd.Timestamp(gold["observation_time"].max())
    predictions = model.predict(MAX_FORECAST_HORIZON)
    benchmarked = attach_seasonal_naive_benchmark(
        predictions.assign(dataset=dataset), panel, origin_as_of=now
    )
    records = [
        {
            "dataset": dataset,
            "series_id": str(row.series_id),
            "forecast_origin": data_cutoff,
            "target_time": pd.Timestamp(row.target_time),
            "horizon": int(row.horizon),
            "model_name": decision["model"],
            "model_version": model.version,
            "forecast_vintage_time": pd.Timestamp(now),
            "prediction": float(row.prediction),
            "lower": None,
            "upper": None,
            "data_cutoff": data_cutoff,
            "origin_as_of": pd.Timestamp(now),
            "uncertainty_status": decision["uncertainty_status"],
            "seasonal_naive_prediction": row.seasonal_naive_prediction,
            "seasonal_naive_source_time": None
            if pd.isna(row.seasonal_naive_source_time)
            else pd.Timestamp(row.seasonal_naive_source_time),
            "seasonal_naive_source_vintage_time": None
            if pd.isna(row.seasonal_naive_source_vintage_time)
            else pd.Timestamp(row.seasonal_naive_source_vintage_time),
            "seasonal_naive_status": str(row.seasonal_naive_status),
            "seasonal_naive_unavailable_reason": None
            if pd.isna(row.seasonal_naive_unavailable_reason)
            else str(row.seasonal_naive_unavailable_reason),
            "generated_at": pd.Timestamp(now),
            "code_revision": revision,
        }
        for row in benchmarked.itertuples()
    ]
    frame = pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)
    if (frame["prediction"] < 0).any():
        raise ValueError("live scoring produced a negative prediction")
    return frame


def score_store(
    store: LocalTableStore,
    *,
    now: datetime,
    revision: str,
    dataset: str = "mrts",
) -> ScoringOutcome:
    """Assemble point-in-time gold, score, and snapshot the forecast table."""
    decision = scoring_decision(dataset)
    if decision["status"] != "authorized":
        return ScoringOutcome(forecasts=None, decision=decision, data_cutoff="")
    panel = observations_as_of(store.read(SILVER_MONTHLY), as_of=now)
    forecasts = score_from_panel(panel, now=now, revision=revision, dataset=dataset)
    store.replace_snapshot(GOLD_MONTHLY, build_gold_monthly(panel, as_of=now))
    store.replace_snapshot(GOLD_FORECASTS, forecasts)
    return ScoringOutcome(
        forecasts=forecasts,
        decision=decision,
        data_cutoff=pd.Timestamp(forecasts["data_cutoff"].iloc[0]).date().isoformat(),
    )
