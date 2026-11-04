# Snapshot contract 2.1

The public snapshot (`v2-latest.json`) is the only live-loop artifact the
frontend consumes. It is produced by the Databricks publisher task from the
governed gold tables, committed to the machine-written `live-data` branch,
and deployed to GitHub Pages by the Pages workflow, which validates it
against this contract (`scripts/validate_snapshot.py`) before shipping.

## Claim vocabulary

| State | Meaning |
|---|---|
| implemented | code exists with tests |
| locally verified | executed against local store / local Delta |
| executed on Databricks | a real workspace run recorded in `DATABRICKS_RUN.md` |
| scheduled | the recurring job is enabled and has run on schedule |
| publicly deployed | the snapshot URL serves the file and the frontend retrieves it |

`is_live` in the snapshot must stay `false` until the recurring schedule has
executed end to end. It is a schedule-state flag, not a capability claim.

## Point fields

| Field | Meaning |
|---|---|
| `horizon`, `target_time` | forecast step and target month |
| `prediction` | model forecast, non-negative |
| `seasonal_naive_prediction` | actual of `target_time - 12 months` as published at or before the forecast origin; `null` when unavailable |
| `seasonal_naive_source_time` | the prior-year month the value was copied from |
| `seasonal_naive_source_vintage_time` | the vintage selected at forecast time |
| `seasonal_naive_status` | `available` or `unavailable` |
| `seasonal_naive_unavailable_reason` | `prior_year_observation_missing` or `prior_year_observation_not_published_at_origin` |
| `evaluation_actual`, `evaluation_actual_vintage_time` | realized target once published |
| `model_absolute_error`, `seasonal_naive_absolute_error` | against the realized actual |
| `model_beat_seasonal_naive` | strict error comparison; `null` when the benchmark is unavailable |
| `evaluation_status` | `first_release` or `revised` |

## Invariants

1. Forecast features, and the benchmark selection, use only data published at
   or before `origin_as_of`. Later revisions are invisible at generation time
   and cannot retroactively change stored benchmarks.
2. First-release evaluation and revised evaluation are separate immutable
   rows; nothing is overwritten.
3. Missing benchmarks and missing actuals stay missing: no zeros, no
   substitution with current revised values.
4. Uncertainty bounds appear only if the portfolio decision authorizes
   calibrated uncertainty; publication fails otherwise.
5. Publication fails without gold forecasts or if the frozen decision has not
   authorized the champion.

## Failure and recovery

- `build_snapshot` fails closed (no forecasts, unauthorized model, mixed
  vintages/models/origins, unauthorized bounds).
- The publisher skips the commit when content is unchanged.
- Pages deploys are re-runnable via `workflow_dispatch`; the data branch is
  never hand-edited.
