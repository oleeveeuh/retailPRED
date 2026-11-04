# Milestone 3 Point-in-Time Vintage Backtest

Status: gates predeclared; measured run pending FRED key rotation and ALFRED vintage pulls

## Purpose

The M3 LightGBM challenger is measured on revised history. Champion promotion and
live MRTS serving require the same model to win when it trains only on what was
knowable at each forecast origin. This document predeclares the protocol and the
promotion gate before any measured result exists, so the criteria cannot shift
after the numbers arrive.

## Evaluation contract

- Source: ALFRED monthly vintages via the FRED API for the 11 registered MRTS
  categories; each vintage date reproduces the full series history exactly as
  published on that date.
- Series mapping: `config/fred_mrts_mapping.json`. All 11 entries map to the
  seasonally adjusted monthly MRTS release family on FRED (`MRTSSM4xxxxxUSS`
  pattern, including `MRTSSM44000USS` for the total), so vintages stay within
  one release family. The first fetch run fails closed on any ID FRED rejects,
  and the mapping must be corrected before the backtest proceeds.
- Real-time availability: declared, not inferred. The as-of timestamp for
  forecast origin month *t* is the month-end of *t* plus 45 days, standing in
  for the revised-release cadence. Training rows are selected only by
  `observations_as_of(frame, as_of=...)`; no release timestamps from the data
  itself leak into origin selection.
- Actuals: the latest revised value for each series/month from the same panel,
  independent of the training vintage.
- Origins: identical expanding-origin grid for every model, minimum 60-month
  training window, horizons 1/3/6/12 months, one origin per month.
- Coverage rules: an origin is excluded, and recorded with its reason, when any
  series is missing at that vintage, the as-of panel is ragged across series,
  the most recent training month is not the origin month, or a future month is
  visible. Exclusions are reported in the summary rather than silently dropped.

## Predeclared promotion gate

Global LightGBM becomes the point-in-time champion only if all of the following
hold on identical origins:

1. Mean category WAPE improves over Seasonal Naive.
2. WAPE improves in at least 10 of the 11 categories.
3. Any regressing category stays within 2 percentage points of the Seasonal
   Naive baseline.

Failure of any criterion keeps the MRTS serving decision fail-closed
(`unavailable_until_point_in_time_validation`) and the live loop continues to
ingest and publish data status without scoring.

## Status

- Predeclared on 2026-11-04; no measured result exists yet.
- Blocked on: rotated `FRED_API_KEY`, completed series mapping, and the ALFRED
  fetch (`make fetch-alfred-mrts END_YEAR=<year>`), then
  `make vintage-backtest-mrts`.
- This section must be updated with the measured tables and the gate outcome in
  the same commit that first runs the backtest.

## Limitations

- The fixed 45-day availability lag is a declared approximation of the Census
  release cadence; it is applied identically to every model and every origin.
- FRED's seasonal adjustment is the Census process, but FRED is a redisseminator;
  the series mapping must record the exact FRED IDs used so vintages are auditable.
- The 11 national aggregates remain a small panel; this gate validates MRTS
  champion status only and makes no claim about M5 serving.

## Reproduce

```bash
# after FRED_API_KEY is rotated and the mapping is completed
make fetch-alfred-mrts END_YEAR=2026 FRED_API_KEY=$FRED_API_KEY
make vintage-backtest-mrts
```
