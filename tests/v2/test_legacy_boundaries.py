import pytest

from project_root.etl.fetch_mrts import MRTSFetcher


class _FailedResponse:
    status_code = 503
    text = "unavailable"


def test_legacy_fetcher_does_not_generate_data_when_census_fails(monkeypatch) -> None:
    fetcher = object.__new__(MRTSFetcher)
    fetcher.base_url = "https://example.invalid"
    fetcher.series_info = {}
    fetcher.additional_series = {}
    monkeypatch.setattr("project_root.etl.fetch_mrts.requests.get", lambda *args, **kwargs: _FailedResponse())
    monkeypatch.setattr("project_root.etl.fetch_mrts.time.sleep", lambda _: None)
    assert not hasattr(fetcher, "_generate_sample_data")

    with pytest.raises(RuntimeError, match="synthetic fallback is disabled"):
        fetcher.fetch_series("MRTSSM44000USS", start_year=2025, end_year=2025)
