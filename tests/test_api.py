"""API smoke tests.

Run against the FastAPI app with TestClient — no server, no real requests.
Skipped automatically when the backend dependencies are not installed.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402  (backend/main.py, path set up by conftest)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_categories_list(client):
    resp = client.get("/api/categories/list")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 11
    keys = {c["key"] for c in body["categories"]}
    assert "total_sales" in keys


def test_predictions_history_empty_or_valid(client):
    """Works whether the local DB is populated or freshly initialized."""
    resp = client.get("/api/predictions/history?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert "total_count" in body
    assert isinstance(body["predictions"], list)


def test_models_endpoint(client):
    resp = client.get("/api/models?active_only=true")
    assert resp.status_code == 200


def test_scenario_list(client):
    resp = client.get("/api/scenarios/list")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert isinstance(scenarios, (list, dict))


def test_predict_requires_params(client):
    """Missing required params must 422, not 500."""
    resp = client.get("/api/predict")
    assert resp.status_code == 422
