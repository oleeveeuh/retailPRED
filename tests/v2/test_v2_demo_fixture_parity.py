import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_frontend_and_backend_v2_demo_fixtures_match() -> None:
    backend = json.loads(
        (ROOT / "backend/demo/v2_forecasts.json").read_text(encoding="utf-8")
    )["fixtures"][0]
    frontend = json.loads(
        (ROOT / "frontend/public/demo-data/v2-forecast.json").read_text(
            encoding="utf-8"
        )
    )
    for field in (
        "dataset",
        "series_id",
        "forecast_origin",
        "data_cutoff",
        "model_name",
        "model_version",
    ):
        assert frontend[field] == backend[field]
    assert frontend["points"] == [
        {"horizon": horizon, "lower": None, "upper": None, **point}
        for horizon, point in enumerate(backend["points"], start=1)
    ]
    assert frontend["provenance"]["code_revision"] == backend["code_revision"]
    assert frontend["provenance"]["source_sha256"] == backend["source_sha256"]
