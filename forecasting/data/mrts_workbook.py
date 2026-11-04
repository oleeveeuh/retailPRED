"""Parser for the Census MRTS annual-revision sales workbook."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting.contracts import SeriesMetadata
from forecasting.data.mrts import MRTSResponseError


TOTAL_RETAIL_LABEL = "Retail sales, total"


def parse_adjusted_sheet(
    sheet: pd.DataFrame,
    series: SeriesMetadata,
    *,
    year: int,
    vintage_time: datetime,
) -> pd.DataFrame:
    """Extract a registered series from the adjusted section of one year sheet."""
    adjusted_rows = []
    for index, row in sheet.iterrows():
        cells = [str(value).strip().upper() for value in row if pd.notna(value)]
        if any(value.startswith("ADJUSTED") for value in cells):
            adjusted_rows.append(index)
    if len(adjusted_rows) != 1:
        raise MRTSResponseError(f"expected one adjusted section in workbook sheet {year}")
    adjusted = sheet.loc[adjusted_rows[0] + 1 :].copy()

    if series.source_code == "44000":
        matches = adjusted.iloc[:, 1].astype(str).str.strip().eq(TOTAL_RETAIL_LABEL)
    else:
        codes = adjusted.iloc[:, 0].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        matches = codes.eq(series.source_code)
    rows = adjusted.loc[matches]
    if len(rows) != 1:
        raise MRTSResponseError(
            f"expected one workbook row for {series.series_id} in {year}; found {len(rows)}"
        )

    values = pd.to_numeric(rows.iloc[0, 2:14], errors="coerce")
    records = [
        {
            "series_id": series.series_id,
            "observation_time": pd.Timestamp(year=year, month=month, day=1),
            "value": float(value),
            "vintage_time": vintage_time,
            "ingested_at": vintage_time,
            "source": "census_mrts_annual_revision",
        }
        for month, value in enumerate(values, start=1)
        if pd.notna(value)
    ]
    if not records:
        raise MRTSResponseError(f"no numeric workbook observations for {series.series_id} in {year}")
    return pd.DataFrame.from_records(records)


def load_revision_workbook(
    path: str | Path,
    series: SeriesMetadata,
    *,
    start_year: int,
    end_year: int,
    vintage_time: datetime,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("start_year cannot exceed end_year")
    workbook = pd.ExcelFile(path)
    years = [str(year) for year in range(start_year, end_year + 1)]
    missing = set(years).difference(workbook.sheet_names)
    if missing:
        raise MRTSResponseError(f"workbook is missing year sheets: {sorted(missing)}")
    frames = [
        parse_adjusted_sheet(
            pd.read_excel(workbook, sheet_name=year, header=None),
            series,
            year=int(year),
            vintage_time=vintage_time,
        )
        for year in years
    ]
    result = pd.concat(frames, ignore_index=True).sort_values("observation_time", ignore_index=True)
    if result[["series_id", "observation_time"]].duplicated().any():
        raise MRTSResponseError("revision workbook contains duplicate series/month keys")
    return result
