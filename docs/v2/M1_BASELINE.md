# Milestone 1 Seasonal-Naive Baseline

Status: measured v2 baseline

## Evaluation contract

- Source: U.S. Census Bureau, *Estimates of Monthly Retail and Food Services Sales by Kind of Business*, 2026 annual-revision workbook (`benchsales26.xlsx`).
- Source URL: <https://www.census.gov/retail/mrts/www/benchmark/2026/excel/benchsales26.xlsx>
- File SHA-256: `e1243966e8e40ea1b99ddb7158f2224ba91147bc40b7a47c2ab7b398077d8796`
- Data used: complete seasonally adjusted months from January 2010 through December 2025.
- Model: Seasonal Naive v2.0.0, 12-month season length.
- Backtest: expanding window, minimum 60 months, monthly origins, horizons 1, 3, 6, and 12 months.
- RMSSE scale: seasonal differences from the initial 60-month training window only.
- Forecast origins: December 2014 through December 2024.
- Targets: January 2015 through December 2025.
- Fold rows: 5,324, representing 11 categories × 121 origins × 4 horizons. Each target is an independent category/month observation; there are no weekly replicas.
- Important limitation: this is a backtest on the 2026 revision vintage, not a real-time vintage reconstruction.

## Results by category

| Series | Fold rows | MAE (USD millions) | WAPE | Bias (USD millions) | RMSSE |
|---|---:|---:|---:|---:|---:|
| Retail total | 484 | 23,629.9 | 4.81% | -21,611.9 | 1.964 |
| Motor vehicle and parts | 484 | 6,629.4 | 5.87% | -4,646.3 | 1.868 |
| Furniture and home furnishings | 484 | 659.2 | 6.56% | -217.0 | 3.366 |
| Electronics and appliances | 484 | 540.1 | 7.41% | 134.9 | 3.781 |
| Building material and garden | 484 | 2,021.3 | 5.64% | -1,616.4 | 2.098 |
| Food and beverage | 484 | 2,836.5 | 4.08% | -2,674.1 | 1.868 |
| Health and personal care | 484 | 1,427.1 | 4.74% | -1,328.3 | 2.080 |
| Gasoline stations | 484 | 5,748.2 | 12.62% | -1,070.3 | 1.912 |
| Clothing and accessories | 484 | 1,721.6 | 7.94% | -355.3 | 4.442 |
| Sporting goods and hobby | 484 | 420.1 | 5.91% | -104.7 | 3.798 |
| General merchandise | 484 | 2,808.0 | 4.23% | -2,105.8 | 2.523 |

The unweighted mean category WAPE is 6.35%. Gasoline is the clearest difficult series for this baseline. Negative bias in 10 of 11 categories reflects the limitation of copying last year's level through a generally growing nominal-sales period.

## Results by horizon

| Horizon | Fold rows | MAE (USD millions) | WAPE | Bias (USD millions) |
|---:|---:|---:|---:|---:|
| 1 month | 1,331 | 4,379.0 | 5.45% | -3,167.1 |
| 3 months | 1,331 | 4,386.0 | 5.42% | -3,199.4 |
| 6 months | 1,331 | 4,408.5 | 5.39% | -3,248.1 |
| 12 months | 1,331 | 4,441.6 | 5.32% | -3,329.1 |

The aggregate horizon table is scale-weighted toward large categories and should not replace the category table for model selection.

## Reproduce

Download the workbook from the exact source URL above, then run:

```bash
make baseline-mrts \
  WORKBOOK=/absolute/path/to/benchsales26.xlsx \
  VINTAGE_TIME=2026-11-02T00:00:00+00:00
```

The fold-level CSV is written under ignored `data/v2/`; the tracked report contains only the declared aggregate evidence.
