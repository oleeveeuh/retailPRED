# RetailPRED v2 Status

Last updated: 2026-11-02

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Milestone 1: complete; native-monthly foundation is the default documented and tested path.
- Milestone 2: platform foundation complete; local and Databricks contracts cover MRTS, Census releases, FRED/ALFRED vintages, M5 operational demand, and the dbt analytics boundary.
- Milestone 3: in progress; the common protocol, global LightGBM challenger, balanced-panel evaluator, promotion policy, and MLflow adapter are implemented.

## Quality verification

- Branch: `v2-rebuild`; Milestone 0 commit `ad595d4`, Milestone 1 foundation commit `532754a`.
- Full Python 3.11 environment: 104 tests passed with no skips at this checkpoint.
- dbt Databricks project parse: passed; 6 models, 20 data tests, 4 sources.
- Frontend type-check: passed.
- Frontend lint: blocking gate passes at the frozen 215-warning legacy cap.
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

Train and measure the global M5 LightGBM challenger on the frozen Seasonal Naive evaluation contract, then compare bottom-up and weighted reconciliation on identical forecasts. A durable remote MLflow run remains pending workspace credentials.

## Milestone 3 evidence to date

- Seasonal Naive and global LightGBM implement a shared fit, point-predict, quantile-predict, metadata, and serialization protocol.
- The global evaluator gives every series identical origins and horizons and verifies prediction timestamps against the monthly panel.
- Covariates declare historical, known-future, static, or unavailable-at-origin roles.
- Scale-free seasonal log-ratio training lets one deterministic model learn across categories without allowing large-dollar series to define the target scale.
- On the same 5,324 revised-history folds, LightGBM reduces mean category WAPE from 6.35% to 4.49% and improves all 11 categories.
- The numerical promotion gate passes, but final champion status remains blocked on untouched, point-in-time evidence. See `M3_LIGHTGBM.md`.
- MLflow tracking records the data hash, Git revision, parameters, metrics, runtime metadata, dataset, and artifacts. A complete local SQLite run finished and was read back through the client; see `MLFLOW_RUN.md`. No durable remote run is claimed.
- Native P10–P90 intervals achieve 86.0% coverage on a disjoint 2022–2025 target evaluation partition, versus an 80% target; fixed conformal calibration over-corrects to 92.4%, widens intervals, and worsens WIS, so neither method is promoted. See `M3_PROBABILISTIC.md`.

## Milestone 4 evidence to date

- The official 12-level M5 hierarchy is generated from bottom product/store metadata as a sparse summing matrix.
- Bottom-up forecasts are exactly coherent, and incomplete bottom panels fail closed.
- A nonnegative diagonal-variance weighted least-squares reconciler provides a scalable MinT-style challenger without claiming full covariance MinT.
- Revenue weights allocate equal total weight to each level and then weight nodes by recent dollar sales.
- WRMSSE requires fixed positive training scales and weights and rejects duplicate evaluation rows.
- The accepted-license M5 files are now supplied locally. A memory-bounded gate verified their hashes, 30,490 bottom series, 1,941 observed days, 6,841,121 price rows, and the official 42,840-node hierarchy; see `M4_M5_SOURCE.md`.
- The leakage-safe weekly Seasonal Naive baseline scores 0.869701 WRMSSE on `d_1914`–`d_1941`, reproducing the public M5 validation benchmark. Training scales and revenue weights end at `d_1913`; see `M4_M5_BASELINE.md`.
- Level WRMSSE ranges from 0.751430 at total to 1.100402 at item/store, establishing the frozen comparison target for global LightGBM.

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
