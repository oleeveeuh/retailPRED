# RetailPRED v2 Repository Audit

Audit date: 2026-11-02

Audited revision: `605d962` (`main`)

## Executive assessment

RetailPRED already contains a substantial full-stack prototype, useful documentation, and an unusually candid limitations section. The v2 rebuild should preserve that history while replacing the current forecasting core rather than layering additional tools over it.

The primary technical risk is not missing functionality. It is the number of overlapping and partially simulated paths for ingestion, training, inference, validation, and UI presentation. The first implementation work should therefore establish one authoritative data contract, one training interface, one inference interface, and one evaluation pipeline.

## Baseline status

- Git worktree: clean on `main`, tracking `origin/main`.
- Tags: none. The proposed `v1-undergrad` tag does not yet exist.
- Working directory: approximately 1.0 GB, including local dependencies and generated state.
- Git object pack: approximately 147 MB; `.git` directory approximately 243 MB.
- Python surface: 61 modules and approximately 23,472 lines, excluding notebooks and dependencies.
- Current documented runtime: Python 3.11. The local test environment currently uses Python 3.14.7.
- Frontend type-check: passes.
- Frontend lint: fails with 105 errors and 110 warnings.
- Python tests in the available minimal environment: 18 pass and 2 skip. The skipped coverage includes API behavior and model discovery because FastAPI and pandas are absent from that environment.

## Critical findings

### 1. Evaluation and feature correctness

- Rolling, difference, percentage-change, and year-over-year target features include the current target value during training.
- Monthly MRTS observations are expanded into synthetic daily or weekly rows. Four tracked category files contain 5,814 generated observations while the others contain 456 rows.
- The current weekly validation repeats one monthly actual across multiple weekly predictions, inflating the apparent number of independent validation samples.
- Fixed category scaling factors are part of the validation path, including at least one explicitly estimated factor.
- MAPE is the principal published metric despite its weaknesses for heterogeneous and low-volume series.

Impact: current model rankings and training metrics cannot serve as the v2 baseline. The raw monthly data and rolling-origin evaluator must be rebuilt before model development.

### 2. Multiple competing implementations

The repository contains several overlapping training and serving systems, including:

- `train.py`
- `backend/ml/train_73_features.py`
- `backend/ml/train_statistical_models.py`
- `backend/ml/train_model.py`
- `backend/ml/train_lgbm_es.py`
- `backend/ml/inference.py`
- `backend/ml/multi_resolution_inference.py`
- `backend/ml/unified_inference.py`
- `backend/services/model_loader.py`
- `backend/services/prediction_service.py`

Category names, display names, NAICS codes, scaling factors, model discovery rules, and fallback behavior are repeated across these modules.

Impact: fixes can affect one path while leaving another path unchanged. V2 needs a single package with explicit interfaces and one registry of datasets, series, models, and metrics.

### 3. Silent mock and fallback behavior

Production-facing paths can return randomized or simulated results when data or models are unavailable. Examples include:

- mock predictions and training metrics;
- synthetic historical data;
- heuristic outputs labeled as PatchTST or TimesNet;
- random economic indicator histories;
- fixed `$50K` scenario baselines;
- simulated frontend learning curves;
- hard-coded scenario multipliers presented through counterfactual-style interfaces.

Impact: silent fallbacks undermine recruiter trust and make operational failures appear successful. V2 should fail explicitly or return a typed `unavailable` state. Demonstration fixtures must be deterministic and visibly labeled.

### 4. Model and data artifacts in Git

- `backend/ml/models/` contains approximately 390 MB of committed model files.
- Random Forest artifacts are approximately 36 MB each.
- A second legacy artifact layout exists under `backend/training_outputs/models/`.
- Generated feature CSVs, predictions, metrics, and diagnostic images are committed.
- `predictions.csv` is tracked even though the README describes it as a local, regenerated artifact.

Impact: the repository is slow to clone and model lineage is unclear. V2 should store models through MLflow/object storage and keep only small deterministic fixtures, benchmark summaries, and selected documentation graphics in Git.

### 5. Model suite does not match repository claims

- Random Forest is redundant with the stronger LightGBM path and accounts for most repository weight.
- PatchTST and TimesNet implementations and weights are absent; similarly named prediction rows were produced by heuristics.
- Statistical model configuration is inconsistent across modules. AutoETS is removed in one path but remains available in others.
- Current models are trained independently by category despite the small amount of genuine monthly data.

Impact: v2 should replace the current menu with a controlled suite representing four distinct approaches: Seasonal Naive, global LightGBM, NHITS, and Chronos-2.

### 6. Test and CI gaps

- Type-checking passes, but lint is intentionally non-blocking in CI and currently reports 215 findings.
- The minimal local Python environment permits API and model tests to skip when dependencies are absent.
- Existing tests primarily cover schema/configuration and metric reconciliation; they do not prove temporal causality, idempotent ingestion, point-in-time joins, interval calibration, reconciliation coherence, or model promotion behavior.
- The Docker deployment workflow validates image creation but retains a placeholder production deployment step.

Impact: v2 acceptance criteria must be encoded in tests. Required checks should not silently skip in CI.

### 7. Data and taxonomy inconsistencies

- The canonical backend category module contains 10 component categories plus total sales.
- Other configuration paths also include nonstore retailers, but no corresponding tracked training CSV or model is present.
- Category keys vary (`total_sales` versus `total_retail_sales`), and display-name transformations are repeated.
- The current set is not proven to be a complete additive hierarchy beneath total retail sales.

Impact: hierarchical reconciliation must not be applied to the current MRTS set until a validated additive hierarchy exists. M5 is the safer initial reconciliation dataset.

### 8. Scenario and explainability semantics

- Economic series are displayed by the application but are not inputs to the shipped forecast models.
- Scenario endpoints mainly apply predefined multipliers rather than estimating causal effects.
- Some SHAP paths fall back to feature importance or equal weights.

Impact: v2 should use the terms `scenario` and `sensitivity analysis`, not `counterfactual`, unless a causal design is added. SHAP responses must identify the exact model and explanation method or return unavailable.

## Assets worth preserving

- The honest limitations and methodology documentation.
- The React/FastAPI delivery concept.
- Existing API schemas that remain useful after contract review.
- Census/FRED ingestion knowledge and category mappings after consolidation.
- The clean-chronological-split intent and existing validation database schema as migration references.
- Existing CI, Docker, and Terraform work as references rather than assumed production-ready infrastructure.
- The visible history from undergraduate prototype to rebuilt system.

## Required preconditions for implementation

1. Tag revision `605d962` as `v1-undergrad` before structural changes.
2. Create a dedicated v2 branch or worktree.
3. Adopt one supported Python version and a reproducible locked environment.
4. Define authoritative data contracts before rewriting feature or model code.
5. Preserve benchmark outputs needed for the before/after methodology report.
6. Do not rewrite Git history during the initial rebuild. Artifact history cleanup can be considered separately after v2 is stable.

## Audit conclusion

The project is a strong candidate for a rebuild, but not for incremental expansion of the existing architecture. The recommended approach is a staged replacement: first establish correctness and contracts, then add the lakehouse and modern model suite, and only then reconnect serving and presentation layers.
