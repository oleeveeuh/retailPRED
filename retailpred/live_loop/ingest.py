"""Live-loop ingestion and fail-closed scoring for the MRTS use case.

Pulling the latest Census release into the versioned silver table is always
permitted. Scoring runs only when the frozen portfolio decision authorizes it;
otherwise the refusal, with its required action, is recorded as an explicit
outcome instead of producing ungoverned forecasts.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from forecasting.data.monthly_dataset import observations_as_of
from forecasting.models.global_lightgbm import GlobalLightGBM
from forecasting.portfolio import model_decision
from forecasting.registry import MRTS_SERIES
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
class IngestOutcome:
    silver_rows: int
    gold_rows: int
    scoring: dict[str, str]
    data_cutoff: str


def code_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def merge_latest_observations(
    store: LocalTableStore, client: Any, *, start_year: int, end_year: int
) -> pd.DataFrame:
    """Pull the latest Census release and merge it into versioned silver."""
    frames = [client.fetch_range(item, start_year, end_year) for item in MRTS_SERIES]
    batch = pd.concat(frames, ignore_index=True)
    batch["release_time"] = pd.NaT
    batch = batch.loc[:, SILVER_MONTHLY.columns]
    return store.merge(SILVER_MONTHLY, batch)


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


def score_gold_panel(
    store: LocalTableStore,
    *,
    now: datetime,
    revision: str,
    dataset: str = "mrts",
) -> pd.DataFrame:
    """Assemble point-in-time gold, fit the promoted champion, and write forecasts.

    Refuses unless the frozen portfolio decision authorizes scoring; the
    refusal is an explicit :class:`ScoringNotAuthorized` error carrying the
    portfolio's required action.
    """
    decision = scoring_decision(dataset)
    if decision["status"] != "authorized":
        raise ScoringNotAuthorized(decision)

    panel = observations_as_of(store.read(SILVER_MONTHLY), as_of=now)
    gold = build_gold_monthly(panel, as_of=now)
    store.replace_snapshot(GOLD_MONTHLY, gold)
    model = GlobalLightGBM().fit(gold)
    data_cutoff = pd.Timestamp(gold["observation_time"].max())
    predictions = model.predict(MAX_FORECAST_HORIZON)
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
            "uncertainty_status": decision["uncertainty_status"],
            "generated_at": pd.Timestamp(now),
            "code_revision": revision,
        }
        for row in predictions.itertuples()
    ]
    frame = pd.DataFrame.from_records(records, columns=GOLD_FORECASTS.columns)
    if (frame["prediction"] < 0).any():
        raise ValueError("live scoring produced a negative prediction")
    return store.replace_snapshot(GOLD_FORECASTS, frame)


def run_live_ingest(
    store: LocalTableStore,
    client: Any,
    *,
    start_year: int,
    end_year: int,
    now: datetime,
    revision: str,
) -> IngestOutcome:
    """Pull the latest release, then score only if the portfolio authorizes it."""
    silver = merge_latest_observations(
        store, client, start_year=start_year, end_year=end_year
    )
    decision = scoring_decision("mrts")
    gold_rows = 0
    if decision["status"] == "authorized":
        gold_rows = len(score_gold_panel(store, now=now, revision=revision))
    return IngestOutcome(
        silver_rows=len(silver),
        gold_rows=gold_rows,
        scoring=decision,
        data_cutoff=pd.Timestamp(silver["observation_time"].max()).date().isoformat(),
    )
