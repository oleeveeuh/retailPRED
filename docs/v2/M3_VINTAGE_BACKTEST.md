# Milestone 3 Point-in-Time Vintage Backtest

Status: **measured — gate FAILED**; the MRTS LightGBM challenger remains unauthorized for serving (2026-11-04)

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

## Measured result (2026-11-04)

Source: `data/v2/alfred_mrts_vintages.csv` — 11 series × 106 monthly ALFRED
vintages (last archived vintage per calendar month), 422,576 versioned
observations. Code revision at measurement: `2c1ac47`.

| Model | Folds | Mean category WAPE | Aggregate WAPE |
|---|---:|---:|---:|
| Seasonal Naive | 1,408 | **8.16%** | **5.43%** |
| Global LightGBM | 1,408 | 8.49% | 5.48% |

LightGBM improved 4 of 11 categories (building materials +1.05pp, food and
beverage +1.21pp, gasoline +3.07pp, general merchandise +3.41pp) and regressed
7, worst on clothing (−4.02pp), furniture (−2.32pp), and electronics (−2.10pp).

**The predeclared gate FAILED on all three criteria.** The MRTS LightGBM
challenger is not promoted; the serving decision stays fail-closed and the
scheduled pipeline continues to refuse scoring.

Notable context: point-in-time forecasting is measurably harder than
revised-history forecasting — the same seasonal-naive benchmark scores 6.35%
mean category WAPE on revised history versus 8.16% on real-time vintages, and
the LightGBM edge that existed on revised history (−1.86pp) disappears entirely
when only origin-visible data is used.

Origin coverage: 32 evaluated monthly origins (2018 → 2025-08), 313 exclusions
— 250 origins before vintage coverage began (ALFRED archives for these series
start ~2017-10) and 63 months where no archived vintage fell inside the as-of
window. Both models were evaluated on identical origins under identical vintage
conditions, so the comparison is fair; absolute staleness varies where archive
density is sparse.

## Status

- Predeclared 2026-11-04; measured and **failed** 2026-11-04.
- Consequence: live MRTS scoring stays unauthorized; snapshot publication stays
  dormant; `is_live` stays false. The scheduled pipeline keeps ingesting and
  accumulating fresh revisions.
- Path forward: re-run this gate as vintage coverage grows (the scheduled
  ingestion adds one vintage month per month) or after a documented model
  revision — the criteria above are fixed for this protocol; any protocol change
  requires a new predeclared document first.

## Limitations

- The fixed 45-day availability lag is a declared approximation of the Census
  release cadence; it is applied identically to every model and every origin.
- ALFRED archive density for these series is irregular (~10 vintages/year), so
  training-data staleness varies by origin even though the selection rule is
  uniform.
- FRED's seasonal adjustment is the Census process, but FRED is a redisseminator;
  the series mapping records the exact FRED IDs used (`MRTSSM44xxxxUSS` family)
  so vintages are auditable.
- The 11 national aggregates remain a small panel; this gate validates MRTS
  champion status only and makes no claim about M5 serving.

## Reproduce

```bash
# after FRED_API_KEY is rotated and the mapping is completed
make fetch-alfred-mrts END_YEAR=2026 FRED_API_KEY=$FRED_API_KEY
make vintage-backtest-mrts
```
