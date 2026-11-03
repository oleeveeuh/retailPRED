# Milestone 5 versioned serving contract

Date: 2026-11-03

Status: backend contract and primary React forecast surface implemented

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

## React forecast surface

The primary forecast page now consumes the same v2 contract. In hosted demo
mode it reads a static copy of the backend fixture; a parity test prevents the
two copies from drifting. With a backend configured it calls
`POST /api/v2/forecasts` directly.

The page:

- permanently displays a historical-offline-demo warning;
- exposes only the measured M5 LightGBM decision;
- fixes the series and origin to the cataloged fixture;
- limits the horizon to the four evidence-backed points;
- reports unit demand rather than incorrectly formatting M5 values as dollars;
- compares forecast with realized evaluation actuals;
- displays model status, data cutoff, code revision, evidence, and uncertainty
  status;
- omits confidence bands and SHAP claims that are not available under the v2
  contract.

Random model metrics were removed from the demo model list. Type-check, lint,
and production build pass; the legacy warning count falls from the frozen cap
of 215 to 194.

## Remaining Milestone 5 work

- Retire or isolate legacy mutation, manual validation, and mock scenario routes.
- Build the Tableau workbook from the now-governed local extracts and capture
  selected dashboard screenshots.

## Governed BI boundary

`make bi-extracts AS_OF_DATE=...` now produces contract-validated model
performance, M5 hierarchy, M5 uncertainty, MRTS regime, and data-status CSVs
plus an integrity manifest. The measured local build contains 6 performance
rows, 12 hierarchy rows, 47 uncertainty rows, 3 regime rows, and 2 freshness
rows. See `bi/README.md` for grains, fields, and the recommended Tableau
workbook.
