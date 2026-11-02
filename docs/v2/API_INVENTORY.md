# V1 API Inventory and V2 Disposition

This inventory freezes the intended disposition of the v1 FastAPI surface before implementation begins. `Keep` means retain the user capability, not necessarily the current implementation or response schema.

## Core forecast and metadata APIs

| V1 surface | Disposition | V2 direction |
|---|---|---|
| `GET /api/health` | Keep | Split service, model, and data-freshness health. |
| `GET /api/categories/list` | Replace | Serve the canonical series registry, dataset, hierarchy role, and availability. |
| `GET /api/categories/{category}/models` | Replace | Read promoted and challenger versions from MLflow. |
| `GET /api/models` | Replace | Return registry-backed versions, provenance, metrics, and lifecycle state. |
| `GET /api/predict` | Replace | Versioned forecast request with origin, horizon, quantiles, model version, and data cutoff. |
| `POST /api/categories/predict` | Merge | Fold into the versioned forecast endpoint. |
| `GET /api/predictions/history` | Keep | Query immutable forecast records with origin and realization status. |
| `POST /api/predictions/validate` | Replace | Realizations arrive through governed actuals ingestion rather than manual mutation. |
| `POST /api/predictions/auto-validate` | Remove | Replace with scheduled realization/evaluation jobs. |
| `GET /api/shap-explain` | Replace | Explain an immutable forecast/model pair; never synthesize explanations. |

## Training and data operations

| V1 surface | Disposition | V2 direction |
|---|---|---|
| `POST /api/refresh-data` | Remove from public API | Trigger through the orchestrator with run metadata and access controls. |
| `POST /api/train` | Remove from public API | Trigger through ML workflows; serving must not train models. |
| `/api/training-metrics/*` | Replace | Serve governed experiment and evaluation summaries from gold tables/MLflow. |
| `GET /api/historical-sales` | Keep | Serve gold observations with frequency, units, vintage, and release metadata. |
| `GET /api/economic-indicators/current` | Replace | Serve point-in-time economic observations with source and vintage. |

## Scenario APIs

| V1 surface | Disposition | V2 direction |
|---|---|---|
| `POST /api/counterfactual` | Rename/replace | Use `scenario` or `sensitivity`; do not imply causal identification. |
| `GET/POST /api/scenarios/analyze` | Replace | Apply declared inputs through models that actually consume those covariates. |
| `POST /api/scenarios/model-prediction` | Merge | Use the common forecast endpoint with a scenario payload. |
| `POST /api/scenarios/custom` | Replace | Validate known-future covariates and return provenance. |
| `GET /api/scenarios/similar-periods` | Reassess | Retain only with a defined similarity metric and evaluation. |
| `GET /api/scenarios/regime` | Reassess | Retain only with a documented regime method and uncertainty. |
| `POST /api/scenarios/sensitivity` | Keep/replace | Report conditional sensitivity, not causality. |
| `/api/scenarios/indicators/*` | Merge | Consolidate under the economic-data resource. |
| `GET /api/scenarios/health` | Remove | Covered by unified health with component status. |

## Export APIs

| V1 surface | Disposition | V2 direction |
|---|---|---|
| `GET /api/export/predictions-csv` | Keep | Export the governed forecast schema. |
| `GET /api/export/historical-csv` | Keep | Export observations with units and vintage metadata. |
| `GET /api/export/model-performance-csv` | Replace | Export evaluation facts using governed metric definitions. |

## V2 contract rules

- All new product routes live under `/api/v2` during migration.
- Training and ingestion are operational workflows, not unauthenticated serving endpoints.
- Forecast responses include dataset, series, forecast origin, horizon, target timestamp, quantiles, model name/version, data cutoff, and experiment/run identifiers.
- Unavailable data, models, or explanations return explicit typed errors; they never fall back to random or hard-coded values.
- V1 routes remain read-only during migration and are removed only after the frontend and exports use v2 contracts.
