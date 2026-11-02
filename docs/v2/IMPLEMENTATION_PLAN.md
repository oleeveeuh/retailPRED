# RetailPRED v2 Implementation Plan

Status: proposed

Source audit: [`AUDIT.md`](AUDIT.md)

## Product definition

RetailPRED v2 will be an evidence-first retail forecasting platform with two explicitly separated use cases:

1. **Macro planning:** monthly U.S. retail-category forecasting using MRTS and point-in-time economic vintages.
2. **Operational demand:** store/item forecasting at M5 scale with a valid retail hierarchy.

The shared platform will provide versioned ingestion, tested feature construction, reproducible experiments, probabilistic forecasts, model promotion, and serving. The two datasets will share infrastructure and evaluation conventions without pretending that they represent the same forecasting problem.

## Non-negotiable principles

- Correctness precedes model sophistication.
- No target-derived feature may use information unavailable at the forecast origin.
- Every observation records when it occurred, when it was released, and when it was ingested where the source permits.
- Backtests use only information available as of each forecast origin.
- Missing models or data produce explicit errors or unavailable states, never randomized production responses.
- Model promotion is based on declared metrics and constraints, not model novelty.
- A simple benchmark remains in every result table.
- Generated data and trained models are not committed to Git.
- Documentation distinguishes measured results, demonstrations, and future work.

## Target repository structure

```text
retailPRED/
├── platform/                 # Spark/Delta ingestion and transformations
│   ├── ingestion/
│   ├── transformations/
│   ├── quality/
│   └── orchestration/
├── warehouse/                # dbt project and semantic reporting models
├── forecasting/              # Shared datasets, features, models, evaluation
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── reconciliation/
│   └── registry/
├── serving/                  # FastAPI application and batch inference
├── frontend/                 # React application
├── bi/                       # Tableau-ready extracts and dashboard docs
├── infrastructure/           # Minimal validated infrastructure as code
├── tests/                    # Unit, integration, data, and contract tests
├── docs/
│   ├── architecture/
│   ├── methodology/
│   └── v2/
└── pyproject.toml
```

The existing paths will be migrated incrementally. They will not be moved mechanically before their replacement interfaces exist.

## Model suite

| Model | Purpose | Core datasets |
|---|---|---|
| Seasonal Naive | Minimum auditable baseline | MRTS, M5 |
| Global LightGBM | Practical feature-based champion | MRTS, M5 |
| NHITS | Task-trained probabilistic neural model | M5; MRTS only if data sufficiency is demonstrated |
| Chronos-2 | Zero-shot/fine-tuned foundation-model challenger | MRTS, M5 |

Random Forest, AutoARIMA, AutoETS, PatchTST, and TimesNet will be removed from the primary product interface. Historical results will remain documented as v1 evidence, not presented as v2 models.

## Milestone 0: Preserve and stabilize v1

### Work

- Tag the current revision as `v1-undergrad`.
- Create the v2 implementation branch/worktree.
- Add durable decision, status, and benchmark documentation.
- Standardize the Python runtime and dependency lock.
- Inventory the API routes and mark keep, replace, or remove.
- Make lint, type-check, unit tests, API tests, and model-discovery tests mandatory.
- Remove silent test skips from CI.
- Capture the current published result tables and screenshots as the v1 comparison baseline.

### Acceptance criteria

- The v1 tag resolves to `605d962`.
- A clean clone can install dependencies and run all baseline checks with one documented command.
- Required CI jobs pass without `continue-on-error` or dependency-based skips.
- No existing published result is lost before replacement metrics exist.

## Milestone 1: Correct monthly forecasting foundation

### Work

- Define canonical schemas for series metadata, observations, releases, forecasts, and actuals.
- Rebuild MRTS ingestion at native monthly frequency.
- Eliminate interpolated or repeated targets from model training and evaluation.
- Consolidate category and NAICS configuration into one validated registry.
- Implement causal lag, rolling, difference, and change features using explicit shifts.
- Implement rolling-origin backtesting with configurable horizons and expanding/sliding windows.
- Add WAPE, MAE, RMSSE, bias, pinball loss, weighted interval score, coverage, and interval width.
- Produce a before/after leakage report.

### Acceptance criteria

- Temporal-integrity tests fail if a feature depends on the forecasted target or a future timestamp.
- Every backtest row records forecast origin, horizon, training cutoff, data vintage, model version, and actual source.
- Monthly sample counts equal independent monthly forecast targets rather than repeated weekly rows.
- Seasonal Naive results are reproducible from a clean environment.
- The README no longer presents v1 metrics as authoritative v2 results.

## Milestone 2: Versioned data platform

### Work

- Implement incremental Census MRTS, MARTS/release, FRED/ALFRED, and M5 ingestion.
- Store `observation_time`, `release_time`, `vintage_time`, and `ingested_at` where applicable.
- Build bronze, silver, and gold Delta tables.
- Add idempotent merge behavior and schema-evolution policies.
- Add source-level, schema, uniqueness, range, relationship, freshness, and reconciliation tests.
- Create dbt staging, fact, dimension, and reporting models.
- Add an SCD2 category dimension only for a demonstrated classification change.
- Document and implement one orchestrator; prefer Databricks Workflows unless portability requirements justify Airflow.

### Acceptance criteria

- Re-running ingestion for the same release produces no duplicate facts.
- A point-in-time query can reconstruct the values known at a selected historical date.
- Gold tables expose stable contracts for training and BI.
- Data-quality failures stop downstream training.
- The local/sample pipeline and Databricks pipeline use the same logical contracts.

## Milestone 3: Unified experiment and model pipeline

### Work

- Define a common forecaster protocol for fit, predict, predict-quantiles, serialize, and metadata.
- Implement Seasonal Naive, global LightGBM, NHITS, and Chronos-2 adapters.
- Separate historical, known-future, static, and unavailable-at-origin covariates.
- Add MLflow tracking for data version, code revision, features, folds, parameters, metrics, runtime, and artifacts.
- Implement champion/challenger promotion with minimum baseline improvement, calibration, and runtime constraints.
- Use deterministic seeds and record hardware/runtime metadata.

### Acceptance criteria

- All models consume identical folds and forecast horizons for a dataset.
- Every reported result links to an MLflow run and immutable data version.
- Chronos-2 is evaluated zero-shot before any fine-tuning.
- A more complex model is not promoted unless it satisfies the declared criteria.
- Model-loading failures cannot return mock predictions.

## Milestone 4: Hierarchical and probabilistic forecasting

### Work

- Implement M5 hierarchy construction and summing matrices.
- Compare unreconciled, bottom-up, and MinT-style reconciled forecasts.
- Train or derive P10/P50/P90 forecasts.
- Evaluate native quantile calibration and add conformal calibration where needed.
- Report metrics by horizon, hierarchy level, category, volume, and intermittency.
- Add regime-aware MRTS analysis for pre-COVID, disruption, and post-disruption periods.

### Acceptance criteria

- Reconciled child forecasts sum to parents within a documented tolerance.
- Interval coverage and width are reported together.
- Calibration uses data disjoint from final evaluation windows.
- Results identify where reconciliation or foundation models hurt as well as help.

## Milestone 5: Serving and user-facing analytics

### Work

- Replace overlapping inference modules with one model-registry-backed service.
- Version the forecast API contract.
- Return explicit provenance, model version, data cutoff, and interval semantics.
- Rename counterfactual endpoints and UI copy to scenario/sensitivity analysis unless causal identification is added.
- Remove randomized fallbacks, simulated metrics, and simulated learning curves from production paths.
- Connect the React application to real gold/API contracts with deterministic demo fixtures as an explicit offline mode.
- Produce Tableau-ready gold extracts for forecast accuracy, hierarchy drill-down, uncertainty, and freshness.

### Acceptance criteria

- API contract tests cover successful, unavailable, stale-data, and incompatible-model cases.
- Every displayed forecast includes its origin, horizon, model, data cutoff, and uncertainty level.
- Demo mode is visually labeled and contains no claims of live inference.
- Tableau and React report the same governed metric definitions.

## Milestone 6: Deployment, observability, and reproducibility

### Work

- Build minimal production images for training and serving.
- Deploy the API and connect an environment-specific frontend.
- Add service health, data freshness, inference latency, failure rate, drift, and realized-error monitoring.
- Validate infrastructure code against the selected deployment architecture; remove speculative resources.
- Add rollback documentation for data, model, and service versions.
- Add scheduled retraining only after promotion and rollback are tested.

### Acceptance criteria

- A deployment is reproducible from documented configuration without committed secrets.
- Health checks distinguish service health, model availability, and data freshness.
- A prior model version can be restored and verified.
- CI builds, tests, scans, and deploys the exact artifact promoted to the environment.

## Milestone 7: Portfolio and résumé delivery

### Work

- Publish a concise top-level README with Data Science and ML Platform reading paths.
- Add architecture, data lineage, evaluation, and model-promotion diagrams.
- Publish the leakage before/after table and honest model benchmark.
- Document limitations, compute cost, licensing, and blocked claims.
- Add selected dashboard screenshots and a short reproducible demo flow.
- Draft separate outcome-based résumé bullets only from shipped, measured results.

### Acceptance criteria

- Every headline claim maps to a tracked result or deployed surface.
- The repository can be understood within a recruiter skim and withstand a technical deep dive.
- No résumé metric uses placeholders, synthetic values, or pseudo-replicated sample counts.

## Verification strategy

The final quality gate will include:

- Python formatting, linting, typing, unit tests, and integration tests;
- frontend lint, type-check, component tests, and production build;
- data-contract and dbt tests;
- causal-feature and point-in-time join tests;
- ingestion idempotency tests;
- forecast metric reconciliation tests;
- probabilistic calibration tests;
- hierarchy coherence tests;
- API contract and container smoke tests;
- a clean-clone reproducibility run.

## Decision gates requiring owner input

Implementation can proceed locally through Milestone 1 without external accounts. Later gates require decisions or credentials for:

1. Databricks workspace and cloud/storage location.
2. Tableau Public versus another Tableau environment.
3. Deployment target and acceptable ongoing cost.
4. Whether the M5 source files may be downloaded and retained locally.
5. Whether trained Chronos-2/NHITS artifacts will be hosted publicly or regenerated on demand.
6. Whether Git history should eventually be rewritten to remove legacy binary weight after v2 is stable.

## Immediate next action

Execute Milestone 0, then begin Milestone 1 with the canonical monthly data and evaluation contracts. Do not begin Spark, MLflow, or neural-model integration until the temporal-integrity suite and monthly Seasonal Naive baseline are green.
