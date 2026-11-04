#!/usr/bin/env python3
"""Download real-time MRTS vintages from ALFRED into the canonical CSV boundary.

Every monthly vintage date yields the full series history exactly as published
on that date, producing the versioned panel that point-in-time evaluation and
live serving require. Missing credentials fail loudly; no synthetic data is
ever substituted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from forecasting.data.fred import FREDClient, MissingFREDAPIKeyError
from forecasting.data.monthly_dataset import write_monthly_csv
from forecasting.registry import MRTS_SERIES

FRED_RELEASE_SERIES_URL = "https://api.stlouisfed.org/fred/release/series"
_VALID_FRED_ID = re.compile(r"^[A-Z0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=Path("config/fred_mrts_mapping.json"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--vintage-start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/v2/alfred_mrts_vintages.csv"))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Fetch only mapped series; point-in-time champion promotion still requires all 11.",
    )
    parser.add_argument(
        "--release-id",
        type=int,
        help="List FRED series in a release to complete the mapping, then exit.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        key: value.strip()
        for key, value in raw.items()
        if not key.startswith("_") and isinstance(value, str)
    }
    known = {item.series_id for item in MRTS_SERIES}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise SystemExit(f"mapping contains unknown MRTS series ids: {unknown}")
    missing = known - set(mapping)
    if missing:
        raise SystemExit(f"mapping is missing registered series: {sorted(missing)}")
    invalid = [
        (key, value)
        for key, value in mapping.items()
        if value and not _VALID_FRED_ID.fullmatch(value)
    ]
    if invalid:
        raise SystemExit(f"mapping contains malformed FRED series ids: {invalid}")
    return mapping


def discover_release(client: FREDClient, release_id: int) -> None:
    offset = 0
    while True:
        response = client.session.get(
            FRED_RELEASE_SERIES_URL,
            params={
                "release_id": release_id,
                "api_key": client.api_key,
                "file_type": "json",
                "limit": 1000,
                "offset": offset,
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise SystemExit(f"release discovery failed with HTTP {response.status_code}")
        payload = response.json()
        series = payload.get("seriess", [])
        for item in series:
            print(f"{item['id']}\t{item['title']}")
        if len(series) < 1000:
            return
        offset += 1000


def main() -> None:
    args = parse_args()
    api_key = os.getenv("FRED_API_KEY")
    try:
        client = FREDClient(api_key)
    except MissingFREDAPIKeyError as error:
        raise SystemExit(str(error)) from error

    if args.release_id is not None:
        discover_release(client, args.release_id)
        return

    mapping = load_mapping(args.mapping)
    unmapped = sorted(key for key, value in mapping.items() if not value)
    if unmapped and not args.allow_partial:
        raise SystemExit(
            "mapping is incomplete for: "
            f"{unmapped}. Complete config/fred_mrts_mapping.json or pass --allow-partial."
        )
    if unmapped:
        print(f"WARNING: partial fetch, skipping unmapped series: {unmapped}")

    selected = [(key, value) for key, value in sorted(mapping.items()) if value]
    vintage_start = args.vintage_start_year or args.start_year
    first_vintage = date(vintage_start, 1, 1)
    last_vintage = date.today() if args.end_year >= date.today().year else date(args.end_year, 12, 1)
    vintage_dates = pd.date_range(first_vintage, last_vintage, freq="MS").date
    print(f"Fetching {len(vintage_dates)} monthly vintages for {len(selected)} series...")

    ingested_at = datetime.now(timezone.utc)
    frames: list[pd.DataFrame] = []
    total = len(vintage_dates) * len(selected)
    completed = 0
    for vintage_date in vintage_dates:
        for mrts_id, fred_id in selected:
            payload = client.fetch_monthly_vintage(fred_id, vintage_date)
            rows: list[dict[str, object]] = []
            for observation in payload["observations"]:
                raw_value = str(observation.get("value", "."))
                if raw_value in {".", "", "NaN"}:
                    continue
                rows.append(
                    {
                        "series_id": mrts_id,
                        "observation_time": pd.Timestamp(observation["date"]),
                        "value": float(raw_value),
                        "release_time": pd.NaT,
                        "vintage_time": pd.Timestamp(vintage_date).tz_localize("UTC"),
                        "ingested_at": ingested_at,
                        "source": "fred_alfred",
                    }
                )
            frames.append(pd.DataFrame.from_records(rows, columns=[
                "series_id",
                "observation_time",
                "value",
                "release_time",
                "vintage_time",
                "ingested_at",
                "source",
            ]))
            completed += 1
            if completed % 25 == 0:
                print(f"  {completed}/{total} vintage fetches complete")
            time.sleep(0.4)  # FRED recommends no more than 120 requests per minute.

    if not frames:
        raise SystemExit("no vintage observations were returned")
    result = pd.concat(frames, ignore_index=True)
    write_monthly_csv(result, args.output)
    print(f"Wrote {len(result):,} versioned monthly observations to {args.output}")


if __name__ == "__main__":
    main()
