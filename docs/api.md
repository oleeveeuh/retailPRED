# API Reference

FastAPI service, `backend/main.py`. Interactive docs at `http://localhost:8000/docs` (Swagger UI) when the backend is running.

Base URL (local): `http://localhost:8000`

## Health

### `GET /api/health`

Static liveness probe (does not check DB or models).

```json
{"status": "healthy"}
```

## Forecasting

### `GET /api/predict`

Generate a forecast for a category. Writes each point to `prediction_log`.

| Query param | Required | Values |
|---|---|---|
| `category` | ✔ | category key, e.g. `total_sales`, `automobile_dealers` (see `/api/categories/list`) |
| `weeks_ahead` | ✔ | 1–52 |
| `model_name` | | `LGBM` (default), `RandomForest` |
| `start_date` | | `YYYY-MM-DD` (first forecast week) |

```bash
curl "http://localhost:8000/api/predict?category=total_sales&model_name=LGBM&weeks_ahead=4&start_date=2026-01-05"
```

Response: `PredictionResponse` with `forecasts[]` (date, predicted value, confidence interval), `shap_values[]`, `features_used`, `metadata`.

### `GET /api/predictions/history`

Validated prediction log with accuracy summary.

| Query param | Default |
|---|---|
| `model_name`, `start_date`, `end_date` | — |
| `limit` | 100 |

Response: `{predictions[], total_count, accuracy_summary{avg_error_percentage, total_validated, ...}}`.

### `POST /api/predictions/validate`

Attach an actual value to a logged prediction.

```json
{"prediction_id": 123, "actual_value": 1525.75, "notes": "from Census backfill"}
```

### `POST /api/predictions/auto-validate`

Backfill actuals from stored reference data for a date range.

### `POST /api/counterfactual` / `POST /api/refresh-data` / `POST /api/train`

Counterfactual what-if analysis; manual data refresh; on-demand training. Training endpoints are synchronous and intended for single-category use.

## Models & explainability

### `GET /api/models?active_only=true`

All trained models with metadata (category, type, metrics).

### `GET /api/training-metrics/models`

Aggregated training/validation metrics, read from `training_outputs/validation_metrics.json`.

### `GET /api/shap-explain?category=...&model_name=...`

SHAP feature attributions for the latest prediction of a model.

## Categories

### `GET /api/categories/list`

```json
{"categories": [{"key": "total_sales", "display_name": "Total Retail Sales"}], "total_count": 11}
```

### `GET /api/categories/{category}/models`

Model types available for one category.

## Scenarios (what-if analysis)

Prefix `/api/scenarios` — see `backend/api/scenario_routes.py`:

- `GET /api/scenarios/list` — the 5 built-in scenarios (baseline +1%, recession −8%, recovery +6%, rate-hike −3%, inflation-surge −2%)
- `POST /api/scenarios/analyze` — `{category, scenario_type}`
- `POST /api/scenarios/model-prediction` — `{category, model_name, scenario_type}`
- `GET /api/scenarios/regime`, `GET /api/scenarios/similar-periods`, `POST /api/scenarios/sensitivity`, `POST /api/scenarios/custom`
- `GET /api/scenarios/indicators/current` — latest economic indicator values

Scenario multipliers are applied to base predictions client-visibly (`base_prediction` vs `prediction`); they are heuristics for exploration, not econometric forecasts.

## Exports

- `GET /api/export/predictions-csv`
- `GET /api/export/historical-csv`
- `GET /api/export/model-performance-csv`

## Economic indicators

- `GET /api/economic-indicators/current` — latest FRED-derived indicator snapshot.

## Error behavior

Validation errors return `422` (Pydantic). Missing models raise `FileNotFoundError` → `500` with the model path in `detail`. Missing database tables surface as `500` — initialize the DB per the README quick start.

Not mounted: `backend/api/context.py` (kept for reference, not routed).
