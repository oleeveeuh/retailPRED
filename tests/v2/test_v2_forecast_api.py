import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_v2_model_portfolio_exposes_frozen_decisions(client: TestClient) -> None:
    response = client.get("/api/v2/models")
    assert response.status_code == 200
    body = response.json()
    assert body["contract_version"] == "2.0"
    assert {model["dataset"] for model in body["models"]} == {"mrts", "m5"}
    assert all(model["production_available"] is False for model in body["models"])
    assert all(model["uncertainty_available"] is False for model in body["models"])


def test_v2_demo_catalog_is_explicitly_historical(client: TestClient) -> None:
    response = client.get("/api/v2/demo-catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "offline_demo"
    assert body["data_status"] == "historical_evaluation_fixture"
    assert "not live" in body["disclaimer"]
    assert body["fixtures"][0]["maximum_horizon"] == 4


def test_v2_offline_demo_is_deterministic_and_labeled(client: TestClient) -> None:
    request = {
        "dataset": "m5",
        "series_id": "FOODS_1_001_CA_1_evaluation",
        "forecast_origin": "2016-04-24",
        "horizon": 2,
        "requested_model": "m5_global_lightgbm",
        "mode": "offline_demo",
    }
    first = client.post("/api/v2/forecasts", json=request)
    second = client.post("/api/v2/forecasts", json=request)
    assert first.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert body["is_demo"] is True
    assert body["data_status"] == "historical_evaluation_fixture"
    assert body["uncertainty_status"] == "research_only_failed_segment_gate"
    assert body["points"][0]["lower"] is None
    assert body["points"][0]["upper"] is None
    assert body["provenance"]["source_sha256"]


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (
            {
                "dataset": "m5",
                "series_id": "FOODS_1_001_CA_1_evaluation",
                "forecast_origin": "2016-04-24",
                "horizon": 1,
                "mode": "production",
            },
            "production_model_unavailable",
        ),
        (
            {
                "dataset": "mrts",
                "series_id": "retail_total",
                "forecast_origin": "2025-12-01",
                "horizon": 1,
                "mode": "offline_demo",
            },
            "offline_demo_unavailable",
        ),
        (
            {
                "dataset": "m5",
                "series_id": "FOODS_1_001_CA_1_evaluation",
                "forecast_origin": "2016-04-24",
                "horizon": 1,
                "requested_model": "random_forest",
                "mode": "offline_demo",
            },
            "incompatible_model",
        ),
        (
            {
                "dataset": "m5",
                "series_id": "FOODS_1_001_CA_1_evaluation",
                "forecast_origin": "2016-04-24",
                "horizon": 1,
                "mode": "offline_demo",
                "quantiles": [0.1, 0.9],
            },
            "uncertainty_unavailable",
        ),
    ],
)
def test_v2_forecasts_fail_closed(
    client: TestClient, payload: dict[str, object], code: str
) -> None:
    response = client.post("/api/v2/forecasts", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == code
