"""Versioned, fail-closed forecast and model-portfolio API."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from forecasting.portfolio import MODEL_PORTFOLIO, model_decision


router = APIRouter(prefix="/api/v2", tags=["v2 forecasts"])
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "demo" / "v2_forecasts.json"


class V2ForecastRequest(BaseModel):
    dataset: Literal["mrts", "m5"]
    series_id: str = Field(min_length=1)
    forecast_origin: date
    horizon: int = Field(ge=1, le=28)
    requested_model: str | None = None
    mode: Literal["production", "offline_demo"] = "production"
    quantiles: list[float] = Field(default_factory=list)


class V2ForecastPoint(BaseModel):
    horizon: int
    target_time: date
    prediction: float = Field(ge=0)
    lower: float | None = None
    upper: float | None = None
    evaluation_actual: float | None = Field(default=None, ge=0)


class V2ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    contract_version: Literal["2.0"] = "2.0"
    dataset: Literal["mrts", "m5"]
    use_case: str
    series_id: str
    forecast_origin: date
    data_cutoff: date
    model_name: str
    model_version: str
    model_status: str
    mode: Literal["offline_demo"]
    data_status: Literal["historical_evaluation_fixture"]
    uncertainty_status: str
    is_demo: Literal[True] = True
    points: list[V2ForecastPoint]
    provenance: dict[str, str]


def _unavailable(
    *, code: str, dataset: str, message: str, required_action: str
) -> HTTPException:
    decision = model_decision(dataset)
    return HTTPException(
        status_code=409,
        detail={
            "code": code,
            "message": message,
            "dataset": dataset,
            "required_action": required_action,
            "model_status": decision.point_status,
            "serving_mode": decision.serving_mode,
            "uncertainty_status": decision.uncertainty_status,
        },
    )


def _fixtures() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        raise RuntimeError("v2 forecast fixture contract is invalid")
    return fixtures


@router.get("/models")
async def list_model_portfolio() -> dict[str, object]:
    """Expose measured model decisions without scanning legacy artifacts."""
    models: list[dict[str, object]] = []
    for decision in MODEL_PORTFOLIO:
        record = decision.to_dict()
        record["production_available"] = False
        record["offline_demo_available"] = decision.dataset == "m5"
        record["uncertainty_available"] = False
        models.append(record)
    return {"contract_version": "2.0", "models": models}


@router.get("/demo-catalog")
async def list_demo_catalog() -> dict[str, object]:
    """List deterministic historical fixtures without presenting them as live."""
    fixtures = [
        {
            "dataset": fixture["dataset"],
            "series_id": fixture["series_id"],
            "forecast_origin": fixture["forecast_origin"],
            "data_cutoff": fixture["data_cutoff"],
            "model_name": fixture["model_name"],
            "maximum_horizon": len(fixture["points"]),
        }
        for fixture in _fixtures()
    ]
    return {
        "contract_version": "2.0",
        "mode": "offline_demo",
        "data_status": "historical_evaluation_fixture",
        "disclaimer": "Historical evaluation fixtures are not live or future forecasts.",
        "fixtures": fixtures,
    }


@router.post("/forecasts", response_model=V2ForecastResponse)
async def create_forecast(request: V2ForecastRequest) -> V2ForecastResponse:
    """Serve only evidence-backed fixtures; all other states fail explicitly."""
    decision = model_decision(request.dataset)
    if request.requested_model and request.requested_model != decision.point_model:
        raise _unavailable(
            code="incompatible_model",
            dataset=request.dataset,
            message=(
                f"{request.requested_model!r} is not the frozen model decision for "
                f"{request.dataset!r}."
            ),
            required_action=f"Request {decision.point_model!r} or inspect /api/v2/models.",
        )
    if request.quantiles:
        if any(not 0 < quantile < 1 for quantile in request.quantiles):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_quantiles", "message": "Quantiles must be between zero and one."},
            )
        raise _unavailable(
            code="uncertainty_unavailable",
            dataset=request.dataset,
            message="No uncertainty method passed the serving promotion gate.",
            required_action="Request a point forecast without quantiles.",
        )
    if request.mode == "production":
        raise _unavailable(
            code="production_model_unavailable",
            dataset=request.dataset,
            message="The frozen portfolio does not authorize production inference.",
            required_action=decision.serving_mode,
        )
    if request.dataset != "m5":
        raise _unavailable(
            code="offline_demo_unavailable",
            dataset=request.dataset,
            message="No evidence-backed offline fixture exists for this dataset.",
            required_action="Complete point-in-time validation and publish a governed fixture.",
        )

    fixture = next(
        (
            item
            for item in _fixtures()
            if item["dataset"] == request.dataset
            and item["series_id"] == request.series_id
            and item["forecast_origin"] == request.forecast_origin.isoformat()
        ),
        None,
    )
    if fixture is None:
        raise _unavailable(
            code="historical_fixture_not_found",
            dataset=request.dataset,
            message="The requested series and origin are not in the deterministic demo fixture.",
            required_action="Use the documented fixture returned by the demo catalog.",
        )
    raw_points = fixture["points"]
    assert isinstance(raw_points, list)
    if request.horizon > len(raw_points):
        raise _unavailable(
            code="fixture_horizon_unavailable",
            dataset=request.dataset,
            message=f"The fixture supports at most {len(raw_points)} forecast steps.",
            required_action=f"Request a horizon from 1 to {len(raw_points)}.",
        )
    points = [
        V2ForecastPoint(horizon=index, **point)
        for index, point in enumerate(raw_points[: request.horizon], start=1)
    ]
    return V2ForecastResponse(
        dataset="m5",
        use_case=decision.use_case,
        series_id=str(fixture["series_id"]),
        forecast_origin=date.fromisoformat(str(fixture["forecast_origin"])),
        data_cutoff=date.fromisoformat(str(fixture["data_cutoff"])),
        model_name=str(fixture["model_name"]),
        model_version=str(fixture["model_version"]),
        model_status=decision.point_status,
        mode="offline_demo",
        data_status="historical_evaluation_fixture",
        uncertainty_status=decision.uncertainty_status,
        points=points,
        provenance={
            "code_revision": str(fixture["code_revision"]),
            "source_sha256": str(fixture["source_sha256"]),
            "evidence": decision.evidence[0],
        },
    )
