# RetailPRED v2 Status

Last updated: 2026-11-02

## Current state

- Repository audit: complete.
- Implementation plan: accepted for execution.
- Milestone 0: complete and committed locally on `v2-rebuild`; not pushed.
- Active milestone: Milestone 0 verification complete.

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

Begin Milestone 1 by defining canonical monthly series/observation/forecast contracts and temporal-integrity tests before replacing the data builder.
