# RetailPRED v2 Status

Last updated: 2026-11-03

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Milestone 1: complete; native-monthly foundation is the default documented and tested path.
- Milestone 2: platform foundation complete; local and Databricks contracts cover MRTS, Census releases, FRED/ALFRED vintages, M5 operational demand, and the dbt analytics boundary.
- Milestone 3: locally complete except for a credential-dependent durable remote MLflow run; the common protocol, controlled model suite, evaluator, promotion policy, and local tracking proof are implemented.
- Milestone 4: complete; M5 point, stability, reconciliation, neural, foundation-model, and probabilistic gates are measured, and MRTS regime behavior is documented.
- Data-science/modeling completion: complete for the accepted local scope. Milestone 5 is complete for its local scope, including the governed Tableau workbook. Milestone 6 publication/reproducibility audit is in progress; external deployment and observability remain blocked on a selected target.

## Quality verification

- Branch: `v2-rebuild`; Milestone 0 commit `ad595d4`, Milestone 1 foundation commit `532754a`.
- Full Python 3.11 environment: 143 tests passed with no skips at this checkpoint.
- dbt Databricks project parse: passed; 6 models, 20 data tests, 4 sources; re-verified from a clean throwaway venv against `warehouse/requirements.txt` only.
- Frontend type-check: passed.
- Publication frontend lint: strict zero-warning gate passes. Full preserved frontend lint passes with 193 warnings, below the frozen 215-warning legacy cap; this remains explicit legacy debt rather than a repository-wide clean-lint claim.
- Frontend production build: passed; the existing large-chunk warning remains recorded debt.
- Production npm dependency audit: 0 known vulnerabilities for root and frontend dependencies; remaining audit findings are development-tool transitive dependencies requiring later breaking upgrades.
- API compatibility fix: `/api/models` now normalizes both legacy metric payload formats.

## Decisions recorded

- Maintain separate macro MRTS and operational M5 use cases on shared infrastructure.
- Core model suite: Seasonal Naive, global LightGBM, NHITS, Chronos-2.
- Treat v1 results as historical comparison evidence, not v2 baselines.
- Use explicit unavailable states instead of randomized production fallbacks.
- Build correctness and point-in-time evaluation before platform expansion.
- Standardize local and CI development on Python 3.11 with a compiled lockfile.
- Prevent dependency-based test skipping in the required verification environment.

## Next checkpoint

The fail-closed `/api/v2`, publication-only React forecast route, and five-view local Tableau workbook share governed evidence and decision semantics. The publication audit removed unsupported live-demo claims, corrected the README, isolated legacy frontend routes, replaced placeholder deployment language, added a minimal v2 container contract verified by a serving-only local smoke test, added a CI container smoke job, and upgraded the frontend path from end-of-life Node 20 to Node 24 LTS. Next complete verification and review the release diff before pushing `v2-rebuild`. Public hosting, Tableau publication, remote MLflow, Databricks execution, monitoring, and rollback remain unclaimed.

## Milestone 6 audit evidence to date

- `docs/v2/PUBLICATION_AUDIT.md` records recruiter-facing findings, fixes, open debt, blocked claims, and a release checklist.
- `scripts/audit_publication.py` fails on stale README links, unsupported public UI claims, legacy public routes, tracked real `.env` files, nonportable Tableau paths, and container regressions.
- The default frontend routes only to the deterministic historical M5 evidence page; legacy pages remain source-level provenance and are not public navigation.
- A serving-only smoke test (Python 3.11 venv restricted to `requirements-serving.txt`) served `backend.main:app` from the exact pruned build context (~14 MB, zero legacy pickles): health, models, demo catalog, offline-demo forecast, and fail-closed 409 production responses all verified.
- The publication gate checks `.dockerignore` resolution so the serving context must include the v2 contract and exclude `backend/ml`; it reproduces the pickle-in-context regression caught during verification.
- CI now includes a `container-smoke` job that builds the minimal v2 image and asserts the fail-closed contract; the first image-level proof lands on the pushed run because Docker is unavailable on the local machine.
- CI workflow names now distinguish build/publish actions from deployment; the placeholder deploy job was removed.
- The v2 API image excludes the 402 MB legacy pickle suite, runs as a non-root user, and health-checks `/api/v2/health`.
- Node 24 LTS replaces end-of-life Node 20 in CI and the frontend container.
- The completed rebuild remains local-only until an owner-authorized push and merge.

## Milestone 5 evidence to date

- The API reads frozen model decisions rather than discovering legacy files or auto-selecting retired algorithms.
- The default app mounts only v2 routes; preserved v1 routes require `RETAILPRED_ENABLE_LEGACY_API=true`, are excluded from default OpenAPI, and are covered by isolation tests.
- `/api/v2/health` distinguishes a healthy service from unavailable production models and historical-only data.
- Production inference is explicitly unavailable until each dataset satisfies its recorded serving requirement.
- One deterministic historical M5 fixture proves the success contract without claiming live inference.
- Every successful point includes origin, horizon, target date, model version/status, data cutoff/status, demo label, code revision, source hash, and evidence reference.
- Contract tests cover successful demo inference plus unavailable, stale-data, incompatible-model, and uncalibrated-uncertainty responses; see `M5_SERVING_CONTRACT.md`.
- The React forecast page now consumes the v2 contract, exposes only the frozen M5 model and fixture, reports provenance, compares evaluation actuals, and removes randomized predictions, retired model choices, invented confidence bands, and unsupported explanations.
- A parity test keeps backend and hosted static fixtures identical.
- Five Tableau-ready extracts cover model performance, M5 hierarchy, M5 uncertainty, MRTS regimes, and data status. Exact schemas, keys, numeric constraints, and a SHA-256 manifest fail closed before publication; see `bi/README.md`.
- A reproducible local Tableau workbook packages those contracts into five named views backed by a tracked aggregate-only Hyper extract. Structural tests guard its sources and sheets; no Tableau Public upload is claimed.
- Tableau Public 2026.3 opened every view successfully: 6 model-scorecard marks, 12 hierarchy marks, 47 uncertainty marks, 3 regime marks, and 2 data-status marks.

## Milestone 3 evidence to date

- Seasonal Naive and global LightGBM implement a shared fit, point-predict, quantile-predict, metadata, and serialization protocol.
- The global evaluator gives every series identical origins and horizons and verifies prediction timestamps against the monthly panel.
- Covariates declare historical, known-future, static, or unavailable-at-origin roles.
- Scale-free seasonal log-ratio training lets one deterministic model learn across categories without allowing large-dollar series to define the target scale.
- On the same 5,324 revised-history folds, LightGBM reduces mean category WAPE from 6.35% to 4.49% and improves all 11 categories.
- The numerical promotion gate passes, but final champion status remains blocked on untouched, point-in-time evidence. See `M3_LIGHTGBM.md`.
- MLflow tracking records the data hash, Git revision, parameters, metrics, runtime metadata, dataset, and artifacts. A complete local SQLite run finished and was read back through the client; see `MLFLOW_RUN.md`. No durable remote run is claimed.
- Native P10–P90 intervals achieve 86.0% coverage on a disjoint 2022–2025 target evaluation partition, versus an 80% target; fixed conformal calibration over-corrects to 92.4%, widens intervals, and worsens WIS, so neither method is promoted. See `M3_PROBABILISTIC.md`.
- Regime analysis shows LightGBM improves aggregate WAPE by 60.39% pre-COVID, 24.52% during disruption, and 42.26% post-disruption, but fails the strict all-category gate in the latter two regimes. It remains a development challenger; see `M3_MRTS_REGIMES.md`.

## Milestone 4 evidence to date

- The official 12-level M5 hierarchy is generated from bottom product/store metadata as a sparse summing matrix.
- Bottom-up forecasts are exactly coherent, and incomplete bottom panels fail closed.
- A nonnegative diagonal-variance weighted least-squares reconciler provides a scalable MinT-style challenger without claiming full covariance MinT.
- Revenue weights allocate equal total weight to each level and then weight nodes by recent dollar sales.
- WRMSSE requires fixed positive training scales and weights and rejects duplicate evaluation rows.
- The accepted-license M5 files are now supplied locally. A memory-bounded gate verified their hashes, 30,490 bottom series, 1,941 observed days, 6,841,121 price rows, and the official 42,840-node hierarchy; see `M4_M5_SOURCE.md`.
- The leakage-safe weekly Seasonal Naive baseline scores 0.869701 WRMSSE on `d_1914`–`d_1941`, reproducing the public M5 validation benchmark. Training scales and revenue weights end at `d_1913`; see `M4_M5_BASELINE.md`.
- Level WRMSSE ranges from 0.751430 at total to 1.100402 at item/store, establishing the frozen comparison target for global LightGBM.
- A direct pooled-horizon global LightGBM scores 0.813301 WRMSSE, a 6.48% overall improvement, with 15.21–23.37% gains across item, item/state, and item/store levels; see `M4_M5_LIGHTGBM.md`.
- LightGBM beats Seasonal Naive in all five rolling windows: worst improvement 6.27%, median 10.96%, and mean 13.77%. It is promoted as the current M5 point champion; see `M4_M5_STABILITY.md`.
- Independent level 1–9 forecasts plus nonnegative diagonal-variance WLS score 0.768502 WRMSSE, improving the bottom-up champion by 5.51% and every hierarchy level while restoring exact coherence; see `M4_M5_RECONCILIATION.md`.
- Reconciliation remains a challenger after its complete training-only blend and variance procedure failed the strict rolling-origin gate.
- The reconciliation audit improves four of five windows and mean WRMSSE by 4.79%, but its -1.82% worst fold fails the strict all-fold gate. Bottom-up LightGBM remains champion; see `M4_M5_RECONCILIATION_STABILITY.md`.
- On a predeclared 600-series revenue-stratified subset, NHITS improves Seasonal Naive by 7.98% but trails subset-trained LightGBM by 10.58%. The full-scale neural benchmark is rejected; see `M4_M5_NHITS_FEASIBILITY.md`.
- On the identical predeclared subset, zero-shot Chronos-2 scores 0.928768, improving subset-trained LightGBM by 3.11% and Seasonal Naive by 19.37%. It clears the accuracy and runtime gates and advances to the complete hierarchy benchmark; see `M4_M5_CHRONOS2_FEASIBILITY.md`.
- On the complete official hierarchy, zero-shot Chronos-2 scores 1.937468 WRMSSE and is rejected. It improves the item/state and item/store levels but underforecasts total units by 32.3%, causing severe aggregate-level error. The failure is accuracy rather than runtime; see `M4_M5_CHRONOS2.md`.
- Training-only scaled conformal M5 intervals achieve 80.38% official-weighted coverage against an 80% target and pass every hierarchy-level range, but high-volume bottom coverage is 69.38% and horizon coverage is unstable. The interval policy fails its predeclared segment gate and is not promoted; see `M4_M5_PROBABILISTIC.md`.

## Milestone 2 evidence to date

- Shared versioned table specifications declare bronze, silver, and gold grains, keys, columns, and nullability.
- Exact Census extract replays are idempotent across bronze and silver.
- FRED/ALFRED requests declare an explicit real-time date; revised economic values coexist and obey the same as-of selection rules.
- Census release records distinguish advance MARTS, revised MRTS, and annual revisions with timezone-aware availability.
- M5 ingestion preserves daily item/store demand, weekly prices, calendar features, file checksums, and the operational hierarchy.
- Later revisions coexist and point-in-time gold snapshots select only eligible vintages.
- Unknown schema changes and immutable-key conflicts fail closed.
- Local Parquet reads verify manifest schema version, row count, and SHA-256.
- Quality gates cover schema, nullability, uniqueness, registered-series relationships, value ranges, freshness, continuity, and generic hierarchy reconciliation.
- The architecture and Databricks parity contract are documented in `docs/architecture/data_platform.md`.
- Databricks Delta DDL and immutable merge predicates are generated from the shared table specifications.
- A Declarative Automation Bundle provides development/production targets and a platform bootstrap job without committed workspace secrets.
- dbt staging, dimension, fact, and reporting models expose governed monthly and M5 grains for Tableau-ready analysis.
- CI parses the dbt graph; source freshness and composite-grain tests are declared for connected-workspace execution.

## Milestone 1 evidence to date

- Canonical immutable contracts cover series, observations, releases, forecasts, and actuals.
- The Census parser selects independent monthly sales rows using `time_slot_id`; missing credentials and invalid responses fail explicitly.
- One 11-category registry replaces the inconsistent v1 mappings.
- Lag, rolling, difference, and change features are shifted before target-derived calculation.
- Expanding and sliding rolling-origin backtests record complete provenance.
- Point and probabilistic metric definitions are implemented and unit-tested.
- Seasonal Naive is deterministic at a 12-month period.
- The measured integrity comparison is in `LEAKAGE_REPORT.md`.
- The first measured v2 baseline is in `M1_BASELINE.md`: 5,324 native-monthly fold rows on the Census 2026 revision vintage.
- The canonical CSV boundary validates series/month/vintage uniqueness and supports as-of selection.
- CI runs the temporal-integrity suite explicitly before the full Python suite.
- The v1 interpolating builder requires an explicit legacy flag, and the old fetcher can no longer generate fallback observations.
