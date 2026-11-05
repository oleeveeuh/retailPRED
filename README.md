# retailpred — field notes

**An evidence-first forecasting platform** rebuilt from an undergraduate project. Every number below is measured, every model decision went through a predeclared gate, and the serving contract stays dark until a model earns the right to forecast.

[![CI](https://github.com/oleeveeuh/retailPRED/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/oleeveeuh/retailPRED/actions/workflows/ci-cd.yml)
[![Showcase](https://img.shields.io/badge/showcase-live%20on%20Pages-89927a)](https://oleeveeuh.github.io/retailPRED/)
[![Release](https://img.shields.io/badge/release-v2.0.0-a43b36)](https://github.com/oleeveeuh/retailPRED/releases/tag/v2.0.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**Live showcase → [oleeveeuh.github.io/retailPRED](https://oleeveeuh.github.io/retailPRED/)** · Portfolio → [oleeveeuh.github.io](https://oleeveeuh.github.io/)

## The scoreboard

| The question | The measured answer | Verdict | Evidence |
|---|---|---|---|
| Can one model forecast 30,490 Walmart item/store series better than naive, across an official 42,840-node hierarchy? | **WRMSSE 0.8133 vs 0.8697** (+6.48%), won all 5 rolling stability windows | ✅ **Promoted** — serves as the site's offline demo | [M4 benchmark](docs/v2/M4_M5_LIGHTGBM.md) · [stability](docs/v2/M4_M5_STABILITY.md) |
| Can global LightGBM beat naive forecasting 11 Census retail categories on revised history? | Mean category **WAPE 4.49% vs 6.35%** (+29%), all 11 categories improved, 5,324 folds | ✅ On revised history only | [M3 benchmark](docs/v2/M3_LIGHTGBM.md) |
| Then the harder question: what could it know at *forecast time*? | On 106 real ALFRED vintages: **8.49% vs 8.16%** — the edge disappears | ❌ **Gate refused the model** | [Vintage gate](docs/v2/M3_VINTAGE_BACKTEST.md) |
| Does the pipeline actually run — end to end, on real infrastructure? | **Executed on Databricks**: 13 Delta tables, dbt 20/20 green on populated tables, live Census pull merged (2,200 rows, cutoff Aug 2026) | ✅ [Execution ledger](docs/v2/DATABRICKS_RUN.md) | — |

**The failed gate is the headline, not a footnote.** A promotion gate written *before* measurement refused the model its author wanted to promote — and the system stayed honestly dark. Four other approaches were rejected by their own predeclared gates: zero-shot Chronos-2 (1.937 WRMSSE on the full hierarchy), NHITS, MinT-style reconciliation, and conformal intervals (69.38% coverage on high-volume series). [The gate document →](docs/v2/M3_VINTAGE_BACKTEST.md)

## Two datasets, two lessons

| | **Census MRTS** — macro planning | **M5 (Kaggle)** — operational demand |
|---|---|---|
| Grain | 11 national monthly categories, millions of dollars | 30,490 item/store daily series |
| The catch | **Alive**: every month is published, then revised | **Frozen**: fixed snapshot, ends May 2016 |
| What it taught | Real-time forecasting is measurably harder than revised history — and the revision process itself is the next modeling frontier | Related-series pooling beats per-series models at scale; the hierarchy must add up exactly |

The tracks stay separate on purpose: national dollar categories are never presented as store/SKU demand.

## How it works

```mermaid
flowchart LR
    A[Census MRTS + ALFRED vintages] --> B[Immutable bronze → versioned silver → point-in-time gold]
    M[M5 licensed source files] --> B
    B --> C[Causal features · rolling origins]
    C --> D[Seasonal Naive / global LightGBM / gated challengers]
    D --> E[Predeclared promotion + calibration gates]
    E --> F[Fail-closed FastAPI · typed 409 refusals]
    E --> G[Governed BI → 5-view Tableau workbook]
    B --> P[[Scheduled Databricks pipeline:<br/>ingest → score → realized errors → publish]]
    P --> S[Snapshot on live-data branch]
    S --> W[GitHub Pages · contract-validated]
    E -. refused .-> X[Records the refusal — serves nothing]
```

Delivery target: **[oleeveeuh.github.io/retailPRED](https://oleeveeuh.github.io/retailPRED/)** — the Pages workflow validates the snapshot contract before anything deploys, and the UI reports a missing snapshot as unavailable instead of inventing one.

## What is implemented

- **Immutable contracts** for series, observations, releases, forecasts, actuals, model decisions, and BI extracts — with observation time, vintage time, explicit as-of selection, and no synthetic fallback.
- **Causal monthly features** — target-derived lags, rolling statistics, and changes shifted before aggregation — plus comparable rolling-origin folds with WAPE, MAE, RMSSE, bias, pinball loss, WIS, and coverage.
- **The official sparse 12-level M5 hierarchy**: revenue weights, WRMSSE, exact bottom-up aggregation, and nonnegative diagonal-variance WLS reconciliation.
- **Governed model selection**: Seasonal Naive and global LightGBM measured; NHITS, Chronos-2, reconciliation, and conformal intervals each rejected by their own gates. Retired v1 labels are excluded from v2 selection.
- **A live-loop foundation, executed on Databricks**: ALFRED vintage pulls behind a fail-closed series mapping, a four-task scheduled pipeline (ingest → fail-closed scoring → realized errors → snapshot publish), vintage-safe seasonal-naive benchmarks frozen beside every forecast, immutable first-release-plus-revised realized tracking, and a [versioned snapshot contract](docs/v2/SNAPSHOT_CONTRACT.md). Scoring stays refused until the [vintage gate](docs/v2/M3_VINTAGE_BACKTEST.md) promotes a champion — by design.
- **A fail-closed FastAPI `/api/v2`** returning typed unavailable states for unapproved production, stale-data, incompatible-model, and uncalibrated-uncertainty requests.
- **Five governed Tableau views** backed by an aggregate-only Hyper extract, checked against the same frozen model decisions.
- **A React evidence page** (this deployment) backed by a deterministic 2016 M5 fixture — explicitly historical, never a current forecast.
- **MLflow tracking proof** locally with data hash, Git revision, parameters, metrics, and artifacts.

## Run and verify

Prerequisites: Python 3.11, Node.js 24 LTS, Git.

```bash
git clone https://github.com/oleeveeuh/retailPRED.git && cd retailPRED
make setup          # Python 3.11 venv + frontend dependencies
make verify         # publication gate → 220 Python tests → frontend type-check, lint, build
```

Run the restricted API and frontend locally:

```bash
.venv-v2/bin/python -m uvicorn backend.main:app --port 8000   # terminal 1
cd frontend && npm run dev                                     # terminal 2
```

`GET /api/v2/health` separates service health from model availability and data freshness; `GET /api/v2/models` returns frozen decisions; `POST /api/v2/forecasts` serves only the documented fixture — production mode fails closed with a typed 409.

## Reproduce the measurements

Census benchmarks require the named annual-revision workbook and an explicit vintage timestamp:

```bash
make benchmark-mrts WORKBOOK=/path/to/benchsales26.xlsx VINTAGE_TIME=2026-11-02T00:00:00+00:00
make probabilistic-mrts WORKBOOK=/path/to/benchsales26.xlsx VINTAGE_TIME=2026-11-02T00:00:00+00:00
```

M5 requires the four competition files under `data/v2/source/m5/` (user-supplied, not redistributed — checksums and validation in [the source record](docs/v2/M4_M5_SOURCE.md)):

```bash
make validate-m5 && make baseline-m5 && make rolling-m5 && make rolling-reconcile-m5 && make probabilistic-m5
```

Real-time vintage evaluation requires a FRED key and the completed series mapping (`config/fred_mrts_mapping.json`):

```bash
make fetch-alfred-mrts END_YEAR=2026 FRED_API_KEY=$FRED_API_KEY   # ~15 min, 106 vintages × 11 series
make vintage-backtest-mrts
```

The Tableau workbook rebuilds from tracked aggregate reports: `make bi-extracts AS_OF_DATE=2026-11-03 && make setup-tableau && make tableau-workbook`.

## Repository map

```text
forecasting/          contracts, causal features, models, evaluation, hierarchy, benchmarks
retailpred/           platform (Delta binding, snapshot contract) + live loop (ingest, scoring, realized)
databricks/           serverless jobs and Asset Bundle resources; warehouse/ holds the dbt project
backend/              restricted FastAPI v2 surface; legacy v1 code isolated behind a flag
frontend/             showcase landing + governed evidence page (GitHub Pages)
bi/                   governed extracts, manifest, Hyper file, Tableau workbook
scripts/              reproducible benchmark, ingestion, and publication-gate entry points
tests/                temporal, platform, model, serving, BI, and parity tests
docs/v2/              decisions, measured results, limitations, execution ledger
project_root/         preserved undergraduate pipeline; never a v2 input
```

## Limits — stated, not hidden

- MRTS benchmark numbers use a 2026 revised history, not origin-available values. LightGBM remains a **development challenger**: the real-time gate refused it.
- M5 ends in 2016 — its champion is valid for that benchmark, not current retail behavior.
- No uncertainty method passed its gate; nothing on this site shows invented confidence bands.
- dbt and Databricks code is executed on a Free Edition workspace; no paid-tier features, remote MLflow server, or production SLA is claimed.
- The live snapshot chain activates only after the vintage gate promotes a champion and the schedule runs; until then the site serves the historical fixture and labels it as such.

## The v1 → v2 story

**v1 (GCP + Airflow).** An Airflow-scheduled pipeline expanded monthly Census values into synthetic weekly rows, computed features with inclusive rolling windows (the target month leaked into its own features), served per-category model pickles through a SQLite-backed FastAPI, and presented the results on a Vercel dashboard. The orchestration worked — jobs ran, data flowed, the site rendered. That sharpens the diagnosis: **v1's failures were not operational. They were epistemic.** Its validation snapshot showed 9.76% MAPE for LightGBM against 14.11% for seasonal naive, but the 429 "validation rows" were monthly actuals repeated across weekly targets — pseudo-replication flattering every model — and the neural entries were heuristic outputs from a retired pipeline. The published demo claimed 90.2% accuracy from static data no deployed model had produced. The stack delivered numbers; nothing in it could say what the numbers meant.

**The rebuild decision.** Orchestration was never the problem, so v2 asked what the pipeline should *enforce*, not just schedule. Three requirements drove the platform choice:

1. **Revisions first-class.** Census values change after publication, so storage needed immutable vintage-keyed rows and point-in-time selection — semantics Delta Lake implements natively and a relational warehouse only provides by hand-building slowly-changing tables.
2. **Auditable governance.** Declared schemas, quality gates that fail closed, and an execution ledger — Unity Catalog and Delta table properties provide both without a separate audit stack.
3. **The repository is the deployment.** One bundle definition in CI, no console drift, zero-cost serverless for a monthly pipeline.

The honest tradeoffs are stated: v2 accepts platform lock-in in exchange for native versioning, and it drops v1's always-on serving posture, because the measurement work showed no MRTS model had earned live inference. Airflow remains a fine orchestrator.

**What the rebuild proved.** The same 11 categories, measured properly: LightGBM beats naive by 29% on revised history — then loses in real time, where a gate written before the measurement refused it. M5's global LightGBM was promoted; Chronos-2, NHITS, MinT-style reconciliation, and conformal intervals were rejected by their own gates. **v1 could not have produced any of those sentences, because nothing in it could fail.**

## Data, artifacts, and licensing

Tracked v2 assets are contracts, tests, checksums, aggregate reports, the aggregate-only Hyper extract, and the deterministic demo fixture. Raw downloads, canonical tables, fold-level predictions, trained weights, databases, logs, and secrets are ignored. M5 source files are user-supplied and not redistributed; FRED series retain their own provider terms; the MIT license covers repository code only — it does not relicense upstream datasets.

An earlier commit exposed a local `.env`; the credentials were rotated and the path was removed from reachable history. Report any residual secret privately before opening a public issue.

The undergraduate implementation is retained on the `v1-undergrad` tag; its weekly expansions, inclusive features, randomized fallbacks, and heuristic outputs are not v2 evidence. The full diagnosis is in [the original audit](docs/v2/AUDIT.md).
