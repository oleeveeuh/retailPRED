from datetime import datetime, timezone

import pandas as pd
import pytest

from retailpred.platform.m5_pipeline import (
    build_m5_gold,
    ingest_m5_frames,
    M5SourceIds,
    normalize_m5_calendar,
    normalize_m5_prices,
    normalize_m5_sales,
    source_file_record,
)
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import BRONZE_SOURCE_FILES, GOLD_M5_DEMAND


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": ["d_1", "d_2"],
            "date": ["2011-01-29", "2011-01-30"],
            "wm_yr_wk": [11101, 11101],
            "weekday": ["Saturday", "Sunday"],
            "wday": [1, 2],
            "month": [1, 1],
            "year": [2011, 2011],
            "event_name_1": [None, None],
            "event_type_1": [None, None],
            "event_name_2": [None, None],
            "event_type_2": [None, None],
            "snap_CA": [0, 0],
            "snap_TX": [0, 0],
            "snap_WI": [0, 0],
        }
    )


def _sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_validation"],
            "item_id": ["FOODS_1_001"],
            "dept_id": ["FOODS_1"],
            "cat_id": ["FOODS"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [3],
            "d_2": [4],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["FOODS_1_001"],
            "wm_yr_wk": [11101],
            "sell_price": [2.50],
        }
    )


def test_m5_files_build_native_daily_gold_with_hierarchy_and_prices() -> None:
    calendar = normalize_m5_calendar(_calendar(), source_file_id="calendar-hash")
    sales = normalize_m5_sales(_sales(), calendar, source_file_id="sales-hash")
    prices = normalize_m5_prices(_prices(), source_file_id="prices-hash")
    gold = build_m5_gold(sales, calendar, prices)

    assert len(gold) == 2
    assert gold["units"].tolist() == [3, 4]
    assert gold["sell_price"].tolist() == [2.5, 2.5]
    assert gold["series_id"].nunique() == 1
    assert gold["sales_source_file_id"].eq("sales-hash").all()


def test_m5_sales_reject_fractional_units_and_missing_calendar_days() -> None:
    calendar = normalize_m5_calendar(_calendar(), source_file_id="calendar-hash")
    fractional = _sales()
    fractional["d_1"] = fractional["d_1"].astype(float)
    fractional.loc[0, "d_1"] = 1.5
    with pytest.raises(ValueError, match="whole numbers"):
        normalize_m5_sales(fractional, calendar, source_file_id="sales-hash")

    missing_day = _sales().rename(columns={"d_2": "d_3"})
    with pytest.raises(ValueError, match="missing sales days"):
        normalize_m5_sales(missing_day, calendar, source_file_id="sales-hash")


def test_source_file_registration_is_content_addressed(tmp_path) -> None:
    source = tmp_path / "sales_train_evaluation.csv"
    source.write_text("id,d_1\nexample,1\n")
    record = source_file_record(
        source,
        dataset="m5",
        source_url="https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
        license_scope="Kaggle competition rules",
        ingested_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert record.iloc[0]["source_file_id"] == record.iloc[0]["sha256"]
    assert record.iloc[0]["size_bytes"] == source.stat().st_size


def test_m5_replay_is_idempotent_across_all_layers(tmp_path) -> None:
    records = []
    source_ids = {}
    for kind in ("calendar", "sales", "prices"):
        path = tmp_path / f"{kind}.csv"
        path.write_text(f"fixture={kind}\n")
        record = source_file_record(
            path,
            dataset="m5",
            source_url="https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
            license_scope="Kaggle competition rules",
            ingested_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        records.append(record)
        source_ids[kind] = record.iloc[0]["source_file_id"]
    registry = pd.concat(records, ignore_index=True)
    ids = M5SourceIds(**source_ids)
    store = LocalTableStore(tmp_path / "lakehouse")

    first = ingest_m5_frames(
        store,
        sales=_sales(),
        calendar=_calendar(),
        prices=_prices(),
        source_files=registry,
        source_ids=ids,
    )
    second = ingest_m5_frames(
        store,
        sales=_sales(),
        calendar=_calendar(),
        prices=_prices(),
        source_files=registry,
        source_ids=ids,
    )
    assert [len(table) for table in first] == [2, 2, 1, 2]
    assert [len(table) for table in second] == [2, 2, 1, 2]
    assert len(store.read(BRONZE_SOURCE_FILES)) == 3
    assert len(store.read(GOLD_M5_DEMAND)) == 2
