# RetailPRED v2 Status

Last updated: 2026-11-02

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Milestone 1: core foundation complete; integration cleanup remains before the milestone is closed.

## Baseline verification

- Git worktree: clean at `605d962` on `main`.
- Full Python 3.11 environment: 27 passed, 0 skipped.
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

Retire the v1 builder from the default workflow and add the Milestone 1 components to the full repository quality gate.

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
