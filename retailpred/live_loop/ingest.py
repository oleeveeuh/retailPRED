"""Live-loop ingestion for the MRTS use case.

Pulls the latest Census release into versioned silver with raw bronze
provenance. Scoring lives in :mod:`retailpred.live_loop.scoring` and is
gated by the frozen portfolio decision; ingestion never produces forecasts.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

import pandas as pd

from forecasting.registry import MRTS_SERIES
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import BRONZE_MRTS, SILVER_MONTHLY


@dataclass(frozen=True)
class IngestOutcome:
    silver_rows: int
    bronze_rows: int
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


def capture_bronze_provenance(store: LocalTableStore, client: Any) -> pd.DataFrame | None:
    """Merge every raw payload fetched this run into the bronze table.

    The extract id derives from the retrieval time, so a replay of the same
    run is idempotent while a later re-fetch legitimately records a new
    vintage. Ingestion time equals each payload's retrieval time: ingestion
    can never precede the vintage it captures.
    """
    from forecasting.data.mrts import MRTS_API_URL
    from retailpred.platform.context import ExtractContext
    from retailpred.platform.mrts_pipeline import bronze_from_api_payload

    frames = []
    for year, (payload, requested_at) in sorted(client.cached_year_payloads().items()):
        context = ExtractContext(
            extract_id=f"census-live-{requested_at.strftime('%Y%m%dT%H%M%S')}-year-{year}",
            source_url=MRTS_API_URL,
            vintage_time=requested_at,
            ingested_at=requested_at,
            release_time=None,
        )
        frames.append(bronze_from_api_payload(payload, context))
    if not frames:
        return None
    return store.merge(BRONZE_MRTS, pd.concat(frames, ignore_index=True))


def run_live_ingest(
    store: LocalTableStore,
    client: Any,
    *,
    start_year: int,
    end_year: int,
) -> IngestOutcome:
    """Pull the latest release into bronze and versioned silver."""
    silver = merge_latest_observations(
        store, client, start_year=start_year, end_year=end_year
    )
    bronze = capture_bronze_provenance(store, client)
    return IngestOutcome(
        silver_rows=len(silver),
        bronze_rows=0 if bronze is None else len(bronze),
        data_cutoff=pd.Timestamp(silver["observation_time"].max()).date().isoformat(),
    )
