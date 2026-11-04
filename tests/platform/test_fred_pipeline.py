from datetime import date, datetime, timezone

import pandas as pd
import pytest

from forecasting.data.fred import FREDClient, MissingFREDAPIKeyError
from forecasting.data.monthly_dataset import observations_as_of
from retailpred.platform.context import ExtractContext
from retailpred.platform.fred_pipeline import ingest_fred_extract
from retailpred.platform.local_store import LocalTableStore
from retailpred.platform.specs import BRONZE_FRED, SILVER_ECONOMIC


def _payload(first_value: str = "3.7", realtime_date: str = "2025-02-07") -> dict[str, object]:
    return {
        "realtime_start": realtime_date,
        "realtime_end": realtime_date,
        "observations": [
            {
                "realtime_start": realtime_date,
                "realtime_end": realtime_date,
                "date": "2024-12-01",
                "value": first_value,
            },
            {
                "realtime_start": realtime_date,
                "realtime_end": realtime_date,
                "date": "2025-01-01",
                "value": ".",
            },
        ],
    }


def _context(extract_id: str, vintage: datetime) -> ExtractContext:
    return ExtractContext(
        extract_id=extract_id,
        source_url="https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE",
        vintage_time=vintage,
        ingested_at=vintage,
    )


def test_fred_client_requires_credentials() -> None:
    with pytest.raises(MissingFREDAPIKeyError, match="FRED_API_KEY"):
        FREDClient(None)


def test_fred_client_declares_point_in_time_monthly_parameters() -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return _payload()

    class Session:
        params: dict[str, object] | None = None

        def get(self, url, *, params, timeout):
            self.params = params
            return Response()

    session = Session()
    FREDClient("a" * 32, session=session).fetch_monthly_vintage(
        "UNRATE", date(2025, 2, 7)
    )
    assert session.params is not None
    assert session.params["realtime_start"] == "2025-02-07"
    assert session.params["realtime_end"] == "2025-02-07"
    assert session.params["frequency"] == "m"
    assert session.params["output_type"] == 1


def test_fred_replay_is_idempotent_and_missing_values_are_not_facts(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    vintage = datetime(2025, 2, 7, tzinfo=timezone.utc)
    first = ingest_fred_extract(store, _payload(), "UNRATE", _context("unrate-1", vintage))
    second = ingest_fred_extract(store, _payload(), "UNRATE", _context("unrate-1", vintage))
    assert len(first[0]) == len(second[0]) == 2
    assert len(first[1]) == len(second[1]) == 1
    assert len(store.read(BRONZE_FRED)) == 2
    assert len(store.read(SILVER_ECONOMIC)) == 1


def test_fred_revision_is_selected_only_at_its_vintage(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    first_vintage = datetime(2025, 2, 7, tzinfo=timezone.utc)
    revised_vintage = datetime(2025, 3, 7, tzinfo=timezone.utc)
    ingest_fred_extract(store, _payload("3.7"), "UNRATE", _context("unrate-1", first_vintage))
    ingest_fred_extract(
        store,
        _payload("3.8", "2025-03-07"),
        "UNRATE",
        _context("unrate-2", revised_vintage),
    )
    silver = store.read(SILVER_ECONOMIC)
    first = observations_as_of(silver, first_vintage)
    revised = observations_as_of(silver, revised_vintage)
    assert first.iloc[0]["value"] == 3.7
    assert revised.iloc[0]["value"] == 3.8
    assert pd.api.types.is_datetime64_any_dtype(revised["observation_time"])


def test_fred_real_time_period_must_contain_requested_vintage(tmp_path) -> None:
    store = LocalTableStore(tmp_path)
    requested = datetime(2025, 3, 7, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="does not contain"):
        ingest_fred_extract(store, _payload(), "UNRATE", _context("wrong-vintage", requested))
    assert store.read(SILVER_ECONOMIC).empty
