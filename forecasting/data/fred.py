"""FRED/ALFRED monthly vintage client with explicit point-in-time parameters."""

from __future__ import annotations

from datetime import date

import requests


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_VINTAGE_DATES_URL = "https://api.stlouisfed.org/fred/series/vintagedates"


class FREDError(RuntimeError):
    pass


class MissingFREDAPIKeyError(FREDError):
    pass


class FREDClient:
    def __init__(self, api_key: str | None, session: requests.Session | None = None) -> None:
        if not api_key:
            raise MissingFREDAPIKeyError("FRED_API_KEY is required for FRED/ALFRED ingestion")
        self.api_key = api_key
        self.session = session or requests.Session()

    def fetch_monthly_vintage(
        self,
        series_id: str,
        vintage_date: date,
        *,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "realtime_start": vintage_date.isoformat(),
            "realtime_end": vintage_date.isoformat(),
            "frequency": "m",
            "aggregation_method": "avg",
            "units": "lin",
            "output_type": 1,
        }
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            params["observation_end"] = observation_end.isoformat()
        response = self.session.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
        if response.status_code != 200:
            # FRED error bodies are plain text naming the cause (invalid key,
            # unknown series, bad parameter) and never echo the api_key.
            body = response.text.strip()[:200]
            raise FREDError(
                f"FRED request failed with HTTP {response.status_code} for series "
                f"{series_id!r} at vintage {vintage_date.isoformat()}: {body or '(no body)'}"
            )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise FREDError("FRED returned a non-JSON response") from exc
        if "observations" not in payload:
            raise FREDError("FRED response is missing observations")
        return payload

    def fetch_vintage_dates(self, series_id: str) -> list[date]:
        """List the exact vintage dates ALFRED archives for a series.

        Series without ALFRED archives (newer FRED additions) return 400 with
        an explanatory body, surfaced as FREDError.
        """
        response = self.session.get(
            FRED_VINTAGE_DATES_URL,
            params={
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            },
            timeout=30,
        )
        if response.status_code != 200:
            body = response.text.strip()[:200]
            raise FREDError(
                f"FRED vintage-dates request failed with HTTP {response.status_code} "
                f"for series {series_id!r}: {body or '(no body)'}"
            )
        payload = response.json()
        dates = payload.get("vintage_dates", [])
        return [date.fromisoformat(str(value)) for value in dates]
