# Snapshot contract 2.1

The public snapshot (`v2-latest.json`) is the forecast artifact the frontend
consumes. It is produced by the Databricks publisher task from the governed
gold tables, committed to the machine-written `live-data` branch, and
deployed to GitHub Pages by the Pages workflow, which validates it
against this contract (`scripts/validate_snapshot.py`) before shipping.
A sibling status record (`v2-ingest-status.json`, contract `2.2-status`)
rides alongside it — see below.

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

## The sibling status record (contract `2.2-status`)

Every completed publish-task run — **including runs where the forecast
snapshot declines** — also writes two mechanical artifacts to the
`live-data` branch:

- `live/v2-ingest-status.json` — the current run's status record;
- `live/runs.jsonl` — the append-only log, one JSON line per run.

**The status record makes no model claims.** It carries run mechanics and
gate verdicts only — pipeline/job ids, UTC timestamps, the run's own
post-merge Delta reads (`rows_in_delta`, `data_cutoff`, `series_count`),
and one verdict per gate (`ingest` success/failed, `score`
authorized/refused with the frozen `model_status` and `required_action`,
`realized` skipped/evaluated with the immutable table's row count,
`publish` published/declined/skipped-identical). It contains no
predictions, no benchmark values, no model metrics; the validator rejects
any artifact that carries those keys under this schema version.
**Its presence or absence authorizes nothing**: a status file on the data
branch does not make the pipeline live, does not authorize scoring, and
does not satisfy the snapshot's fail-closed requirements. A deploy may
carry a status record without a snapshot — that combination is the honest
state while the vintage gate keeps scoring refused.

The record is assembled by the publish task from that run's own Delta
reads and `scoring_decision("mrts")` — never from invented values. Databricks
job-identity parameters are guarded: an unresolved `{{...}}` macro records
as null, never as literal text. The same publication discipline as the
snapshot applies: contract-validated assembly, atomic local write,
skip-on-identical content hash (independent of the snapshot's), the secret
scope token, and machine-written commits whose messages carry the run id.
`runs.jsonl` is never truncated; readers cap the displayed history instead.
A crashed task leaves no record — the absence of a status line for a run is
the honest signature of a run that never reached publication.

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
