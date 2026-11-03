# M5 hierarchy and reconciliation

Status: implementation complete; real-data benchmark pending source files

## Official hierarchy

RetailPRED implements the 12 aggregation levels defined by the M5 competition:

| Level | Grain | Official series |
|---:|---|---:|
| 1 | Total | 1 |
| 2 | State | 3 |
| 3 | Store | 10 |
| 4 | Category | 3 |
| 5 | Department | 7 |
| 6 | State–category | 9 |
| 7 | State–department | 21 |
| 8 | Store–category | 30 |
| 9 | Store–department | 70 |
| 10 | Product | 3,049 |
| 11 | Product–state | 9,147 |
| 12 | Product–store | 30,490 |
|  | **Total** | **42,840** |

These counts and grains follow the competition paper's hierarchy table ([M5 Accuracy paper](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf)) and the organizers' [M5 methods repository](https://github.com/Mcompetitions/M5-methods). The official submission format contains the 30,490 bottom series because coherent higher levels can be obtained by summing them.

## Sparse summing matrix

`M5Hierarchy.from_bottom` derives the hierarchy from governed product/store metadata and constructs a SciPy CSR summing matrix `S`:

- rows are all hierarchy nodes;
- columns are bottom product/store series;
- `S @ bottom_forecast` produces every aggregation level;
- each bottom series belongs to exactly one node at each of 12 levels.

For the complete dataset the expected matrix shape is 42,840 × 30,490 with 365,880 nonzero memberships. Construction validates required identifiers, nulls, duplicate bottom series, duplicate node IDs, and—when requested—the official node counts.

## Reconciliation methods

### Bottom-up

Bottom forecasts are multiplied by `S`. This is exactly coherent by construction and remains the lowest-risk operational baseline.

### Diagonal-variance weighted least squares

The MinT-style challenger solves the sparse projection

```text
minimize || W^(-1/2) (S b - y_hat) ||²
```

where `y_hat` contains independently produced forecasts for all nodes, `b` is the reconciled bottom vector, and `W` is a declared diagonal error-variance approximation. Nonnegative constrained least squares is the default, after which `S b` guarantees coherence.

This is deliberately described as diagonal-variance MinT-style reconciliation, not full covariance MinT. Estimating and inverting a reliable 42,840-series covariance matrix is neither implemented nor justified by the current evidence.

## Evaluation contract

- Coherence is the maximum absolute difference between submitted node forecasts and the aggregates implied by their bottom level.
- M5 revenue weights allocate one-twelfth of total score weight to each hierarchy level, then distribute that level's weight according to recent dollar sales.
- WRMSSE consumes fixed training-period scales and revenue weights. Scales may not be recomputed from evaluation targets.
- Bottom-up and weighted reconciliation must use identical base forecasts, horizons, scales, and weights.

The official M5 results paper emphasizes that the competition evaluates 42,840 hierarchical series and that higher-level forecasts must align with the bottom level ([International Journal of Forecasting](https://doi.org/10.1016/j.ijforecast.2021.11.013)).

## Remaining evidence gate

No real M5 accuracy result is claimed yet. The repository does not contain the competition files, and RetailPRED will not silently download or republish them. A measured benchmark requires the owner to provide the accepted-license files locally:

- `sales_train_evaluation.csv`
- `calendar.csv`
- `sell_prices.csv`

Once available, the next run will validate the official 42,840-node shape, compare unreconciled and bottom-up forecasts, fit reconciliation variances using training residuals only, and report WRMSSE by all 12 levels.
