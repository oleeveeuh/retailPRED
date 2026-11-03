# Milestone 5 versioned serving contract

Date: 2026-11-03

Status: backend foundation implemented

## Outcome

The new `/api/v2` boundary serves only frozen, evidence-backed model decisions.
It does not scan legacy model files, auto-select retired algorithms, manufacture
confidence bands, or fall back to mock predictions.

Implemented resources:

- `GET /api/v2/models`: frozen MRTS and M5 model decisions, serving modes,
  uncertainty status, measured metric, cutoff, and evidence references.
- `GET /api/v2/demo-catalog`: the exact deterministic historical fixtures that
  can be requested in offline-demo mode.
- `POST /api/v2/forecasts`: versioned forecast request and response contract.

## Forecast contract

Requests declare dataset, series, origin, horizon, optional requested model,
production or offline-demo mode, freshness requirement, and any requested
quantiles. Successful responses always include:

- contract version;
- use case and dataset;
- series and forecast origin;
- model name, version, and promotion status;
- data cutoff and data status;
- explicit demo flag;
- horizon and target date for every point;
- uncertainty status;
- source hash, code revision, and evidence reference.

The current successful path is deliberately narrow: one historical M5
evaluation fixture, up to four days, requested with `offline_demo` and
`historical_allowed`. Its values were generated from the measured global
LightGBM artifact and frozen M5 evaluation source. The response includes the
realized evaluation actual only because it is explicitly historical.

## Fail-closed behavior

Typed HTTP 409 responses cover:

- `production_model_unavailable`;
- `offline_demo_unavailable`;
- `incompatible_model`;
- `uncertainty_unavailable`;
- `stale_data`;
- `historical_fixture_not_found`;
- `fixture_horizon_unavailable`.

Invalid quantile values return 422. No error state is converted into a number,
hard-coded confidence interval, alternate model, or randomized response.

## Verification

Contract tests cover model discovery, deterministic successful inference,
explicit demo labeling, production unavailability, stale-data rejection,
retired/incompatible model rejection, uncalibrated uncertainty rejection, and
MRTS demo unavailability. The complete Python suite passes with no skips.

## Remaining Milestone 5 work

- Migrate the React forecast page from the legacy API and simulated prediction
  generator to `/api/v2`.
- Add a permanent visible historical-demo banner and unavailable-state panels.
- Retire or isolate legacy mutation, manual validation, and mock scenario routes.
- Publish governed Tableau-ready forecast, accuracy, uncertainty, hierarchy,
  and freshness extracts from the same contracts.
