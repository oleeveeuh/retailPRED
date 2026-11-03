# Milestone 5 legacy API isolation

Date: 2026-11-03

Status: implemented

## Outcome

The default FastAPI application now exposes only the evidence-backed v2
serving boundary. Preserved v1 modules remain available as historical source
evidence, but they are not imported, mounted, advertised at the root endpoint,
or included in the default OpenAPI document.

This prevents old randomized predictions, manual validation mutations,
request-triggered training, simulated scenarios, and legacy exports from being
mistaken for supported production behavior.

## Default boundary

The default application exposes:

- `GET /api/v2/health`;
- `GET /api/v2/models`;
- `GET /api/v2/demo-catalog`;
- `POST /api/v2/forecasts`.

Health is intentionally not synonymous with model availability. A running
service returns a healthy service component while separately reporting that
production models are unavailable and published data are historical only.

Representative v1 paths such as `/api/predict`, `/api/models`,
`/api/predictions/validate`, `/api/train`, and `/api/scenarios/*` return 404 by
default. Their absence from OpenAPI makes the supported contract visible to
clients and reviewers.

## Archival compatibility

Setting `RETAILPRED_ENABLE_LEGACY_API=true` mounts the preserved v1 routers.
This switch exists only for local historical comparison and compatibility
investigation. It is not a production deployment mode and does not promote any
legacy result into the v2 evidence chain.

## Verification

Automated tests verify:

- component-level v2 health semantics;
- a root document that advertises only v2 resources by default;
- 404 responses for representative legacy prediction, metadata, mutation,
  training, and scenario paths;
- exclusion of legacy mutation and mock routes from default OpenAPI;
- preservation of the isolated legacy metric-normalization utility without
  inventing missing measurements.

At this checkpoint, all 141 Python tests pass with no skips. Frontend
type-check, the lint gate, and the production build also pass.
