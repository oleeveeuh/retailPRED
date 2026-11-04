# Databricks execution evidence

Live claims live here. Nothing below has executed on a Databricks workspace
yet; every entry stays at its current claim level until a real run records
evidence, and the level never silently upgrades.

| Capability | Claim level | Evidence required to upgrade |
|---|---|---|
| Table contracts and Delta binding | locally verified (`make spark-check` passes on a local delta-spark session) | workspace bootstrap run: job URL, table list |
| Bundle deploy (bootstrap job, scheduled pipeline) | implemented | `databricks bundle deploy -t dev` output; job links |
| Census ingestion task | **locally verified** — 17 years of live Census history ingested twice with identical values | workspace run of the same pipeline: row counts, manifest hashes |
| Scoring task | implemented; refuses while the candidate is an unauthorized challenger | a post-promotion run: forecast row counts, benchmark statuses |
| Realized-error task | implemented | first matured-target evaluation rows |
| Snapshot publication to `live-data` | implemented | the data-branch commit with its pipeline run id |
| Recurring schedule | declared paused | unpause commit + first scheduled-run URL, then `is_live=true` |
| GitHub Pages retrieval | implemented (workflow defined) | first Pages deployment URL + frontend fetch check |

## Run log

(append one row per real run; never delete rows)

| Date | Job / task | Result | Link or run id |
|---|---|---|---|
| 2026-11-04 | `make ingest-latest` (local, keyed) | Bronze 110,921 raw rows across 17 year-extracts; silver 4,400 rows = 11 series × 200 months × 2 runs; values identical across both runs; data cutoff 2026-08-01. Two API schema drifts found and fixed en route (ordinal time_slot_id with period in `time`; error_data no longer gettable). | local store `data/v2/store`, commits `bf88e76`, `7adc908`, `781f544` |
| — | — | — | — |
