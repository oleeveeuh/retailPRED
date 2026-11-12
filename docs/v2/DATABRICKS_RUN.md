# Databricks execution evidence

Live claims live here. The claim level of every capability below reflects the
run log at the bottom of this file — nothing upgrades without a recorded run,
and no level moves silently.

| Capability | Claim level | Evidence required to upgrade |
|---|---|---|
| Table contracts and Delta binding | **executed on Databricks** — 13 tables created in `workspace.bronze/silver/gold` by the bootstrap job | — |
| Bundle deploy (bootstrap job, scheduled pipeline) | **executed on Databricks** — bundle deployed to the dev target; 3 serverless jobs registered | first scheduled run |
| Census ingestion task | **executed on Databricks** — 2,200 silver rows merged into `workspace.silver.monthly_observations` from live Census pulls | — |
| Scoring task | executed on Databricks (refusal path): refuses while the candidate is an unauthorized challenger | a post-promotion run: forecast row counts, benchmark statuses |
| Realized-error task | executed on Databricks (empty-skip path verified) | first matured-target evaluation rows |
| Snapshot publication to `live-data` | implemented; decline path verified on Databricks | the data-branch commit with its pipeline run id |
| Recurring schedule | declared, unpaused (2026-11-11) — awaiting its first scheduled fire | first scheduled-run URL, then `is_live=true` |
| Ingest-status record (2.2-status) | implemented; assembly and gate logic contract-tested without a hosted run (`--dry-run` + fixture mode) | first hosted run writing `live/v2-ingest-status.json` + `live/runs.jsonl` to `live-data` |
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
| 2026-11-04 | `platform_ingest_latest` (workspace, serverless) — **first end-to-end pipeline run** | **SUCCESS** — ingest: 2,200 silver rows merged into Delta from live Census, cutoff 2026-08-01; score: correctly refused the unauthorized challenger; realized: skipped with no forecasts; publish: declined without gold forecasts. The full four-task graph and every fail-closed gate verified on hosted infrastructure. | run 452582046469351, lightgbm commit `a89fc18` |
| 2026-11-10 | `make ingest-latest` (local, keyed) | **FAILED then SUCCESS** — first attempt rejected: `~/.zshrc` still held the pre-rotation Census key (fingerprint `58e3a68b`; deactivated by the 2026-11-04 rotation). After deploying the rotated key: 221,842 bronze rows (full provenance, exact double), silver 6,600 = 3 versioned batches, cutoff **2026-08-01 unchanged** — Census's 2026 payload ends at August (raw `time` tokens verified), September advance not yet published. Pull mechanism verified end-to-end; Databricks manual triggers blocked workspace-side ("disabled temporarily") — schedule unpause unaffected. | local store `data/v2/store`, fingerprint prefix `779f2edd` |
| 2026-11-11 | `databricks bundle deploy -t dev` — **schedule unpause** | **SUCCESS** — `pause_status: UNPAUSED` confirmed on workspace job 326179734702237 with the declared daily 14:30 ET schedule and the four-task graph intact; the publish task now carries guarded job-identity macros (`{{job.id}}`, `{{job.run_id}}`, `{{start_time}}` — unresolved records as null) for the 2.2-status record. Gated on the satellite's committed window pins (`0ed0504`: accumulated origins, recorded ingest timestamps) so the first scheduled origin starts the re-gate clock legally. `is_live` stays false until the recurring schedule executes end to end. **Caveat:** the org-side trigger block ("disabled temporarily") may also gate scheduled fires — no retries will be hammered; `jobs list-runs` checked the next day. The first successful scheduled fire upgrades this row to "first scheduled run", flips `is_live` at the next publish, and becomes the first real new-material ingest once Census publishes past August. | unpause commit `2f7b8e8`, deploy 2026-11-11 |
| 2026-11-12 | `jobs list-runs --job-id 326179734702237` — **next-day check per the unpause caveat** | **NO SCHEDULED FIRE.** The 14:30 ET slot of 2026-11-11 produced no run — the run list holds exactly the three 2026-11-04 ONE_TIME runs — so the org-side block gates scheduled fires as well as manual triggers. The workspace schedule itself is verified intact (`pause_status: UNPAUSED`, `0 30 14 * * ? America/New_York`), so fires resume automatically when the block lifts; no retries attempted, per the standing instruction. The first successful scheduled fire upgrades this ledger to "first scheduled run", writes the first 2.2-status record + `runs.jsonl` line to `live-data`, and starts the re-gate accumulation clock. | workspace job 326179734702237 |
| — | — | — | — |
