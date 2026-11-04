"""Native-monthly Census MRTS ingestion with no synthetic fallback."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from forecasting.contracts import SeriesMetadata


MRTS_API_URL = "https://api.census.gov/data/timeseries/eits/mrts"
MRTS_FIELDS = "data_type_code,time_slot_id,seasonally_adj,category_code,cell_value,error_data"


class MRTSError(RuntimeError):
    """Base error for explicit MRTS ingestion failures."""


class MissingAPIKeyError(MRTSError):
    pass


class MRTSResponseError(MRTSError):
    pass


def parse_mrts_response(
    payload: list[list[Any]],
    series: SeriesMetadata,
    *,
    year: int,
    vintage_time: datetime,
) -> pd.DataFrame:
    """Parse one Census annual response into independent monthly observations."""
    if not payload or len(payload) < 2:
        raise MRTSResponseError(f"Census returned no observations for {series.series_id} in {year}")
    header, *rows = payload
    records = [dict(zip(header, row, strict=True)) for row in rows]
    frame = pd.DataFrame.from_records(records)
    required = {"data_type_code", "time_slot_id", "seasonally_adj", "category_code", "cell_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise MRTSResponseError(f"Census response is missing fields: {sorted(missing)}")

    adjustment = "yes" if series.seasonal_adjustment == "seasonally_adjusted" else "no"
    selected = frame.loc[
        frame["data_type_code"].eq("SM")
        & frame["category_code"].astype(str).eq(series.source_code)
        & frame["seasonally_adj"].str.lower().eq(adjustment)
    ].copy()
    selected["month"] = pd.to_numeric(selected["time_slot_id"], errors="coerce")
    selected["value"] = pd.to_numeric(selected["cell_value"], errors="coerce")
    selected = selected.loc[selected["month"].between(1, 12) & selected["value"].notna()]
    if selected.empty:
        raise MRTSResponseError(
            f"no monthly sales rows matched code={series.source_code}, adjustment={adjustment}, year={year}"
        )
    if selected["month"].duplicated().any():
        raise MRTSResponseError(f"duplicate monthly observations for {series.series_id} in {year}")

    ingested_at = datetime.now(timezone.utc)
    result = pd.DataFrame(
        {
            "series_id": series.series_id,
            "observation_time": pd.to_datetime(
                {"year": year, "month": selected["month"].astype(int), "day": 1}
            ),
            "value": selected["value"].astype(float),
            "vintage_time": vintage_time,
            "ingested_at": ingested_at,
            "source": series.source,
        }
    )
    return result.sort_values("observation_time", ignore_index=True)


class MRTSClient:
    def __init__(self, api_key: str | None, session: requests.Session | None = None) -> None:
        if not api_key:
            raise MissingAPIKeyError(
                "Census API key is required; set CENSUS_API_KEY. RetailPRED never substitutes synthetic MRTS data."
            )
        self.api_key = api_key
        self.session = session or requests.Session()
        self._year_payloads: dict[int, tuple[list[list[Any]], datetime]] = {}

    def fetch_year(self, series: SeriesMetadata, year: int) -> pd.DataFrame:
        if year not in self._year_payloads:
            requested_at = datetime.now(timezone.utc)
            response = self.session.get(
                MRTS_API_URL,
                params={"get": MRTS_FIELDS, "time": str(year), "key": self.api_key},
                timeout=30,
            )
            if response.status_code != 200:
                raise MRTSResponseError(f"Census request failed with HTTP {response.status_code}")
            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise MRTSResponseError("Census returned a non-JSON response") from exc
            self._year_payloads[year] = (payload, requested_at)
        payload, requested_at = self._year_payloads[year]
        return parse_mrts_response(payload, series, year=year, vintage_time=requested_at)

    def fetch_range(self, series: SeriesMetadata, start_year: int, end_year: int) -> pd.DataFrame:
        if start_year > end_year:
            raise ValueError("start_year cannot exceed end_year")
        frames = [self.fetch_year(series, year) for year in range(start_year, end_year + 1)]
        result = pd.concat(frames, ignore_index=True).sort_values("observation_time", ignore_index=True)
        if result[["series_id", "observation_time"]].duplicated().any():
            raise MRTSResponseError("duplicate series/month keys across Census responses")
        return result
