# MRTS regime stability analysis

Date: 2026-11-03

Status: measured; development challenger retained without production promotion

## Decision

Global LightGBM remains the strongest MRTS development model but does not pass
the strict all-category regime gate. It improves aggregate and mean-category
WAPE in every economic regime and improves every tested horizon, but it does
not improve every category during disruption or post-disruption periods. The
model will not be patched with evaluation-selected category fallbacks.

## Frozen evidence

The analysis reuses the previously frozen 5,324 fold rows per model. Seasonal
Naive and LightGBM have identical series, forecast origins, target months, and
horizons. No model is retrained and no predictions are changed.

Regimes use target month:

- Pre-COVID: January 2015 through February 2020.
- Disruption: March 2020 through June 2021.
- Post-disruption: July 2021 through December 2025.

The gate requires positive aggregate WAPE improvement, positive equally
weighted mean-category WAPE improvement, and improvement in all 11 categories
within every regime.

## Results

| Regime | Rows/model | Seasonal Naive WAPE | LightGBM WAPE | Improvement | Mean-category improvement | Categories improved |
|---|---:|---:|---:|---:|---:|---:|
| Pre-COVID | 2,530 | 3.46% | **1.37%** | **60.39%** | **45.81%** | 11/11 |
| Disruption | 704 | 12.95% | **9.78%** | **24.52%** | **18.67%** | 10/11 |
| Post-disruption | 2,090 | 4.99% | **2.88%** | **42.26%** | **27.49%** | 8/11 |

LightGBM also improves every horizon in every regime. Its smallest advantage is
post-disruption at 12 months: 4.72% versus 4.75% WAPE, a 0.68% relative gain.

## Failures and interpretation

- During disruption, furniture/home furnishings is effectively tied but
  slightly worse: 20.87% versus 20.82% WAPE.
- Post-disruption, sporting goods/hobby is 26.17% worse than Seasonal Naive,
  electronics/appliances is 13.44% worse, and furniture/home furnishings is
  1.47% worse.
- The disruption period remains difficult for both models. Mean-category WAPE
  rises to 15.27% for LightGBM despite its clear improvement over the baseline.

The result supports a useful modeling claim—LightGBM materially improves the
portfolio across stable, disrupted, and recovery regimes—without supporting a
stronger claim that it dominates every retail category. The revised-history
backtest still cannot substitute for untouched point-in-time vintage evidence.

## Serving handoff

- Keep LightGBM as the MRTS development challenger.
- Do not label it a production champion.
- Return an explicit unavailable state for production MRTS inference until a
  point-in-time validation run earns promotion.
- Keep Seasonal Naive as the visible comparison baseline.
- Keep MRTS intervals research-only because the prior calibration gate failed.

These constraints are encoded in `forecasting/portfolio.py` for the serving
milestone.

## Reproduction

```bash
make regime-mrts
```

The ignored evidence directory contains regime summaries by category and
horizon, the exact fold evidence, its SHA-256, and the code revision.
