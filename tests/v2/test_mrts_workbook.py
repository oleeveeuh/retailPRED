from datetime import datetime, timezone

import numpy as np
import pandas as pd

from forecasting.data.mrts_workbook import parse_adjusted_sheet
from forecasting.registry import get_series


def _sheet() -> pd.DataFrame:
    rows: list[list[object]] = [
        ["NAICS Code", "Kind of Business", *([None] * 12)],
        [None, "NOT ADJUSTED", *([None] * 12)],
        [443, "Electronics and appliance stores", *range(1, 13)],
        [None, "ADJUSTED(2)", *([None] * 12)],
        [None, "Retail sales, total", *range(100, 112)],
        [443, "Electronics and appliance stores", *range(200, 212)],
    ]
    return pd.DataFrame(rows).replace({None: np.nan})


def test_workbook_parser_reads_only_adjusted_monthly_values() -> None:
    result = parse_adjusted_sheet(
        _sheet(),
        get_series("electronics_and_appliances"),
        year=2025,
        vintage_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert len(result) == 12
    assert result["value"].tolist() == list(range(200, 212))
    assert result["observation_time"].dt.month.tolist() == list(range(1, 13))


def test_workbook_parser_uses_label_for_total_retail() -> None:
    result = parse_adjusted_sheet(
        _sheet(),
        get_series("retail_total"),
        year=2025,
        vintage_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert result["value"].tolist() == list(range(100, 112))
