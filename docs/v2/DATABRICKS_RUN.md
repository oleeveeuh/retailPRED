# Databricks execution evidence

Live claims live here. Nothing below has executed on a Databricks workspace
yet; every entry stays at its current claim level until a real run records
evidence, and the level never silently upgrades.

| Capability | Claim level | Evidence required to upgrade |
|---|---|---|
| Table contracts and Delta binding | **executed on Databricks** — 13 tables created in `workspace.bronze/silver/gold` by the bootstrap job | — |
| Bundle deploy (bootstrap job, scheduled pipeline) | **executed on Databricks** — bundle deployed to the dev target; 3 serverless jobs registered | schedule unpaused + first scheduled run |
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
| 2026-11-04 | `platform_bootstrap` (workspace, serverless) | **FAILED** — `NameError: __file__ is not defined`: serverless python tasks exec notebook-style, so the classic path-insert pattern cannot work. Fixed by passing the bundle-synced root as an explicit `--repo-root` parameter. | run 466144488641852, fix commit `9b85242` |
| 2026-11-04 | `platform_bootstrap` (workspace, serverless) | **SUCCESS** — all 13 declared tables created in `workspace.bronze/silver/gold` (including `gold.forecasts` v2 and `gold.realized_errors` v2). Root-cause detail: `${workspace.file_path}` already ends in `/files`, so the first parameter attempt doubled the segment; the job's workspace-layout fallback located the root and the parameter was corrected. | run 946654956135103, fix commit pending |
| 2026-11-04 | `dbt run` + `dbt test` (workspace, starter warehouse) | **SUCCESS** — 6 models materialized in `workspace.dbt_dev_*`, 20/20 data tests passing (vacuously: platform Delta tables not yet populated; the meaningful green follows the first pipeline run). Catalog default corrected from `retailpred` to Free Edition's `workspace`. | commits `e8314d3`+ |
| 2026-11-04 | `platform_ingest_latest` (workspace, serverless) | **FAILED** — ingest task ImportError: `scoring_decision` still imported from the pre-refactor module in the cluster wrapper. Fixed import to `retailpred.live_loop.scoring`. | run 1054816300992672 |
| 2026-11-04 | `platform_ingest_latest` (workspace, serverless) | **FAILED** — `ModuleNotFoundError: No module named 'lightgbm'`: the informational scoring-state import pulls the model chain, and lightgbm was not declared for the serverless environment. Added `lightgbm>=4.0` alongside `pydantic>=2` to every job environment. | fix commit pending |
| — | — | — | — |
