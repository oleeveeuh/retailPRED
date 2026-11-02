# RetailPRED v2 Status

Last updated: 2026-11-02

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Milestone 1: complete; native-monthly foundation is the default documented and tested path.
- Milestone 2: in progress; local bronze/silver/gold contracts plus idempotent MRTS and FRED/ALFRED vintage ingestion are implemented.

## Quality verification

- Branch: `v2-rebuild`; Milestone 0 commit `ad595d4`, Milestone 1 foundation commit `532754a`.
- Full Python 3.11 environment: 69 tests passed with no skips at this checkpoint.
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

Add Census release/MARTS and M5 adapters to the shared contracts, then implement the equivalent declared-schema Delta tables and merges on Databricks.

## Milestone 2 evidence to date

- Shared versioned table specifications declare bronze, silver, and gold grains, keys, columns, and nullability.
- Exact Census extract replays are idempotent across bronze and silver.
- FRED/ALFRED requests declare an explicit real-time date; revised economic values coexist and obey the same as-of selection rules.
- Later revisions coexist and point-in-time gold snapshots select only eligible vintages.
- Unknown schema changes and immutable-key conflicts fail closed.
- Local Parquet reads verify manifest schema version, row count, and SHA-256.
- Quality gates cover schema, nullability, uniqueness, registered-series relationships, value ranges, freshness, continuity, and generic hierarchy reconciliation.
- The architecture and Databricks parity contract are documented in `docs/architecture/data_platform.md`.

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
