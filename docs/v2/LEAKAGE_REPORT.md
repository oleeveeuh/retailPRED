# Monthly Foundation: Before/After Integrity Report

Status: Milestone 1 implementation evidence

## Executive finding

The v1 training tables are not valid inputs for a v2 benchmark. They mix native monthly observations with interpolated daily/weekly rows, compute several target-derived features using the target on the same row, and count repeated monthly actuals as separate weekly validation samples. The v2 foundation preserves one row per category/month and derives every target-history feature from values strictly before the forecasted month.

## Measured repository evidence

The 11 tracked files in `project_root/data_multi_resolution/` contain:

- four daily files with 5,814 rows each but only 192 distinct calendar months;
- seven mixed-frequency files with 456 rows each but only 132 distinct calendar months;
- inclusive rolling calculations such as `df[target].rolling(...)` and same-row `pct_change`/`diff` in `project_root/etl/build_multi_resolution_dataset.py`;
- linear daily interpolation followed by hand-authored day-of-week multipliers; and
- validation in which one Census monthly actual is assigned to multiple weekly prediction rows.

These are preserved as v1 artifacts only. They are not consumed by the v2 modules.

## Before/after controls

| Integrity property | v1 behavior | v2 behavior | Verification |
|---|---|---|---|
| Target frequency | Monthly source expanded to daily/weekly/mixed rows | One independent row per category/month | Duplicate and missing-month validation |
| Rolling features at month *t* | Window includes `y[t]` | Shift by one month, then aggregate | Current/future target mutation test |
| Change features at month *t* | Same-row `pct_change` and `diff` include `y[t]` | Compare `y[t-1]` with `y[t-2]` | Exact feature-value tests |
| Missing source/API | Random sample data may be generated | Typed error; no synthetic fallback | Missing-key test |
| Category mapping | Multiple inconsistent maps and mislabeled codes | One validated 11-series registry | Uniqueness and known-code tests |
| Evaluation split | Mixed holdouts and pseudo-replicated weekly rows | Native-monthly rolling origins | Backtest provenance tests |
| Backtest provenance | Incomplete | Origin, target, horizon, cutoff, training start, vintage, model version, actual source | Required-column test |
| Baseline | Weekly seasonal period applied to mixed data | Deterministic 12-month seasonal naive | Reproducibility test |

## Remaining limitation

The Census time-series API provides revised values, not a complete historical release-vintage archive. Milestone 1 records the retrieval vintage explicitly; true point-in-time reconstruction using release files and ALFRED-style economic vintages belongs to Milestone 2. Until then, v2 results must be described as rolling-origin evaluation on a named revision vintage, not a real-time vintage backtest.

## Reproduce the checks

```bash
make test
```

The controlling tests are under `tests/v2/`, especially `test_monthly_features.py`, `test_mrts.py`, and `test_backtest.py`.
