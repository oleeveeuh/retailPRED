# RetailPRED v2 Status

Last updated: 2026-11-02

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Milestone 1: complete; native-monthly foundation is the default documented and tested path.
- Milestone 2: platform foundation complete; local and Databricks contracts cover MRTS, Census releases, FRED/ALFRED vintages, M5 operational demand, and the dbt analytics boundary.

## Quality verification

- Branch: `v2-rebuild`; Milestone 0 commit `ad595d4`, Milestone 1 foundation commit `532754a`.
- Full Python 3.11 environment: 80 tests passed with no skips at this checkpoint.
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

Begin Milestone 3 modeling with the global LightGBM challenger, retaining Seasonal Naive as the required benchmark and the rolling-origin evaluation contract as the acceptance gate.

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
