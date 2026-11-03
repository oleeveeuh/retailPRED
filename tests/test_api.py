"""Default API surface and isolated legacy utility tests."""

import pytest
from fastapi.testclient import TestClient

from api.routes import normalize_model_metrics
from main import LEGACY_API_ENABLED, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_v2_health_distinguishes_service_model_and_data_state(client: TestClient) -> None:
    response = client.get("/api/v2/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "restricted"
    assert body["components"]["service"]["status"] == "healthy"
    assert body["components"]["models"]["status"] == "production_unavailable"
    assert body["components"]["data"]["status"] == "historical_only"


def test_root_advertises_only_v2_contract_by_default(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["legacy_api_enabled"] is False
    assert set(body["endpoints"]) == {
        "health",
        "v2_models",
        "v2_demo_catalog",
        "v2_forecasts",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/api/predict",
        "/api/categories/list",
        "/api/models",
        "/api/predictions/history",
        "/api/predictions/validate",
        "/api/train",
        "/api/scenarios/list",
        "/api/counterfactual",
    ],
)
def test_legacy_routes_are_not_mounted_by_default(
    client: TestClient, path: str
) -> None:
    assert LEGACY_API_ENABLED is False
    assert client.get(path).status_code == 404


def test_default_openapi_excludes_legacy_mutation_and_mock_routes(
    client: TestClient,
) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert "/api/v2/forecasts" in paths
    assert "/api/predict" not in paths
    assert "/api/predictions/validate" not in paths
    assert "/api/scenarios/analyze" not in paths


def test_legacy_aggregate_metrics_are_normalized_without_inventing_values() -> None:
    metrics = normalize_model_metrics(
        {
            "MAPE": {"mean": 7.5},
            "RMSE": {"mean": 12.25},
            "MAE": {"mean": 8.0},
            "MASE": {"mean": 1.1},
        }
    )
    assert metrics == {
        "rmse": 12.25,
        "mae": 8.0,
        "r2": 0.0,
        "mape": 7.5,
        "training_samples": 0,
    }
