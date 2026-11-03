# M5 licensed-source validation

Date: 2026-11-02  
Status: passed; files remain local and are excluded from Git

## Evidence

The owner-supplied Kaggle competition files passed the memory-bounded source gate:

| Check | Measured result |
|---|---:|
| Bottom item/store series | 30,490 |
| Observed sales days | 1,941 (`d_1`–`d_1941`) |
| Calendar days | 1,969 (2011-01-29–2016-06-19) |
| Weekly price rows | 6,841,121 |
| Hierarchy nodes | 42,840 |
| Hierarchy levels | 12/12 match official counts |
| Total observed units | 66,927,173 |
| Observed sell-price range | 0.01–107.32 |

The source identities are fixed by SHA-256:

- `calendar.csv`: `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5`
- `sales_train_evaluation.csv`: `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0`
- `sell_prices.csv`: `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef`

The generated machine-readable report is
`data/v2/evidence/m5_source_validation.json`. Both the licensed inputs and the
report's local source path are intentionally ignored by Git.

## Reproduction

```bash
make validate-m5
```

The validator reads sales and prices in bounded chunks. It checks official file
dimensions, continuous day columns, nonnegative whole-unit demand, positive
prices, calendar coverage, store/item relationships, evaluation-series IDs,
file hashes, and the official hierarchy shape without first materializing the
roughly 59 million-row long sales table.

## Evidence boundary

This gate establishes source integrity, not forecast accuracy. The next evidence
artifact must run Seasonal Naive on the fixed 28-day evaluation horizon, compute
training-only RMSSE scales and pre-horizon revenue weights, and report WRMSSE at
all 12 levels. LightGBM and reconciliation comparisons must reuse that identical
evaluation contract.
