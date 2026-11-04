PYTHON ?= python3.11
VENV ?= .venv-v2
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
EXPERIMENT_VENV ?= .venv-experiment
EXPERIMENT_PYTHON := $(EXPERIMENT_VENV)/bin/python
EXPERIMENT_PIP := $(EXPERIMENT_VENV)/bin/pip
NEURAL_VENV ?= .venv-neural
NEURAL_PYTHON := $(NEURAL_VENV)/bin/python
NEURAL_PIP := $(NEURAL_VENV)/bin/pip
FOUNDATION_VENV ?= .venv-foundation
FOUNDATION_PYTHON := $(FOUNDATION_VENV)/bin/python
FOUNDATION_PIP := $(FOUNDATION_VENV)/bin/pip
TABLEAU_VENV ?= .venv-tableau
TABLEAU_PYTHON := $(TABLEAU_VENV)/bin/python
TABLEAU_PIP := $(TABLEAU_VENV)/bin/pip
SPARK_VENV ?= .venv-spark
SPARK_PYTHON := $(SPARK_VENV)/bin/python
SPARK_PIP := $(SPARK_VENV)/bin/pip
NPM_CACHE ?= /tmp/retailpred-npm-cache

.PHONY: setup setup-experiment setup-neural setup-foundation setup-tableau setup-spark lock test milestone1-check platform-check warehouse-check fetch-mrts fetch-alfred-mrts vintage-backtest-mrts ingest-latest realized-errors publish-snapshot spark-check baseline-mrts benchmark-mrts track-mrts probabilistic-mrts regime-mrts validate-m5 baseline-m5 benchmark-m5 rolling-m5 reconcile-m5 rolling-reconcile-m5 feasibility-nhits-m5 feasibility-chronos-m5 benchmark-chronos-m5 probabilistic-m5 bi-extracts tableau-workbook frontend-check publication-check verify clean-generated

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install pip==25.3
	$(VENV_PIP) install -r requirements-dev.lock
	NPM_CONFIG_CACHE=$(NPM_CACHE) npm ci
	cd frontend && NPM_CONFIG_CACHE=$(NPM_CACHE) npm ci

setup-experiment:
	$(PYTHON) -m venv $(EXPERIMENT_VENV)
	$(EXPERIMENT_PIP) install pip==25.3
	$(EXPERIMENT_PIP) install -r requirements-dev.lock
	$(EXPERIMENT_PIP) install -r requirements-experiment.txt

setup-neural:
	$(PYTHON) -m venv $(NEURAL_VENV)
	$(NEURAL_PIP) install pip==26.2.1
	$(NEURAL_PIP) install -r requirements-neural.txt

setup-foundation:
	$(PYTHON) -m venv $(FOUNDATION_VENV)
	$(FOUNDATION_PIP) install pip==26.2.1
	$(FOUNDATION_PIP) install -r requirements-foundation.txt

setup-tableau:
	$(PYTHON) -m venv $(TABLEAU_VENV)
	$(TABLEAU_PIP) install pip==26.2.1
	$(TABLEAU_PIP) install -r requirements-tableau.txt

# Requires a Java 17 runtime on the host (brew install openjdk@17).
setup-spark:
	$(PYTHON) -m venv $(SPARK_VENV)
	$(SPARK_PIP) install pip==26.2.1
	$(SPARK_PIP) install -r requirements-spark.txt

spark-check:
	@test -x "$(SPARK_PYTHON)" || (echo "Run make setup-spark first" && exit 1)
	PYSPARK_PYTHON=$$(realpath $(SPARK_VENV)/bin/python) \
	PYSPARK_DRIVER_PYTHON=$$(realpath $(SPARK_VENV)/bin/python) \
	JAVA_HOME=$${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home} \
	$(SPARK_PYTHON) -m scripts.spark_local_check

lock:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install pip==25.3 pip-tools==7.5.2
	PIP_TOOLS_CACHE_DIR=/tmp/retailpred-pip-tools-cache $(VENV_PYTHON) -m piptools compile --resolver=backtracking --strip-extras --output-file=requirements-dev.lock requirements-dev.in

test:
	$(VENV_PYTHON) -m pytest

milestone1-check:
	$(VENV_PYTHON) -m pytest tests/v2

platform-check:
	$(VENV_PYTHON) -m pytest tests/platform

warehouse-check:
	.venv-dbt/bin/dbt parse --project-dir warehouse --profiles-dir warehouse

fetch-mrts:
	@test -n "$(END_YEAR)" || (echo "END_YEAR is required, e.g. make fetch-mrts END_YEAR=2025" && exit 1)
	$(VENV_PYTHON) -m scripts.fetch_mrts_v2 --end-year $(END_YEAR)

# Real-time vintage pulls from ALFRED; requires FRED_API_KEY in the environment
# and a completed FRED series mapping in config/fred_mrts_mapping.json.
fetch-alfred-mrts:
	@test -n "$(FRED_API_KEY)" || (echo "FRED_API_KEY is required in the environment" && exit 1)
	@test -n "$(END_YEAR)" || (echo "END_YEAR is required, e.g. make fetch-alfred-mrts END_YEAR=2026 FRED_API_KEY=..." && exit 1)
	$(VENV_PYTHON) -m scripts.fetch_alfred_mrts --end-year $(END_YEAR) --output data/v2/alfred_mrts_vintages.csv

vintage-backtest-mrts:
	@test -f data/v2/alfred_mrts_vintages.csv || (echo "Run make fetch-alfred-mrts first" && exit 1)
	$(VENV_PYTHON) -m scripts.run_mrts_vintage_backtest data/v2/alfred_mrts_vintages.csv

ingest-latest:
	@test -n "$(CENSUS_API_KEY)" || (echo "CENSUS_API_KEY is required in the environment" && exit 1)
	$(VENV_PYTHON) -m scripts.ingest_latest --store data/v2/store

realized-errors:
	$(VENV_PYTHON) -m scripts.realized_errors --store data/v2/store

publish-snapshot:
	@test -f data/v2/store/gold/forecasts.parquet || (echo "No gold forecasts in the local store; snapshot publishing is unavailable" && exit 1)
	$(VENV_PYTHON) -m scripts.publish_snapshot --store data/v2/store --output data/v2/snapshots/v2-latest.json

baseline-mrts:
	@test -n "$(WORKBOOK)" || (echo "WORKBOOK is required" && exit 1)
	@test -n "$(VINTAGE_TIME)" || (echo "VINTAGE_TIME is required, including a UTC offset" && exit 1)
	$(VENV_PYTHON) -m scripts.run_mrts_baseline "$(WORKBOOK)" --vintage-time "$(VINTAGE_TIME)"

benchmark-mrts:
	@test -n "$(WORKBOOK)" || (echo "WORKBOOK is required" && exit 1)
	@test -n "$(VINTAGE_TIME)" || (echo "VINTAGE_TIME is required, including a UTC offset" && exit 1)
	$(VENV_PYTHON) -m scripts.run_mrts_lightgbm "$(WORKBOOK)" --vintage-time "$(VINTAGE_TIME)"

track-mrts:
	@test -n "$(WORKBOOK)" || (echo "WORKBOOK is required" && exit 1)
	@test -n "$(VINTAGE_TIME)" || (echo "VINTAGE_TIME is required, including a UTC offset" && exit 1)
	@test -n "$(TRACKING_URI)" || (echo "TRACKING_URI is required" && exit 1)
	MPLCONFIGDIR=/tmp/retailpred-matplotlib MLFLOW_DISABLE_AGENT_HINT=1 $(EXPERIMENT_PYTHON) -m scripts.run_mrts_lightgbm "$(WORKBOOK)" --vintage-time "$(VINTAGE_TIME)" --tracking-uri "$(TRACKING_URI)"

probabilistic-mrts:
	@test -n "$(WORKBOOK)" || (echo "WORKBOOK is required" && exit 1)
	@test -n "$(VINTAGE_TIME)" || (echo "VINTAGE_TIME is required, including a UTC offset" && exit 1)
	$(VENV_PYTHON) -m scripts.run_mrts_probabilistic "$(WORKBOOK)" --vintage-time "$(VINTAGE_TIME)"

regime-mrts:
	$(VENV_PYTHON) -m scripts.analyze_mrts_regimes data/v2/mrts_model_folds.csv

validate-m5:
	$(VENV_PYTHON) -m scripts.validate_m5_sources data/v2/source/m5

baseline-m5:
	$(VENV_PYTHON) -m scripts.run_m5_baseline data/v2/source/m5

benchmark-m5:
	$(VENV_PYTHON) -m scripts.run_m5_lightgbm data/v2/source/m5

rolling-m5:
	$(VENV_PYTHON) -m scripts.run_m5_rolling_lightgbm data/v2/source/m5

reconcile-m5:
	$(VENV_PYTHON) -m scripts.run_m5_reconciliation data/v2/source/m5

rolling-reconcile-m5:
	$(VENV_PYTHON) -m scripts.run_m5_rolling_reconciliation data/v2/source/m5

feasibility-nhits-m5:
	$(NEURAL_PYTHON) -m scripts.run_m5_nhits_feasibility data/v2/source/m5

feasibility-chronos-m5:
	$(VENV_PYTHON) -m scripts.run_m5_chronos_feasibility data/v2/source/m5 --foundation-python $(FOUNDATION_PYTHON)

benchmark-chronos-m5:
	$(VENV_PYTHON) -m scripts.run_m5_chronos_benchmark data/v2/source/m5 --foundation-python $(FOUNDATION_PYTHON)

probabilistic-m5:
	$(VENV_PYTHON) -m scripts.run_m5_probabilistic data/v2/source/m5

bi-extracts:
	@test -n "$(AS_OF_DATE)" || (echo "AS_OF_DATE is required, e.g. make bi-extracts AS_OF_DATE=2026-11-03" && exit 1)
	$(VENV_PYTHON) -m scripts.build_bi_extracts --as-of-date "$(AS_OF_DATE)"

tableau-workbook:
	@test -x "$(TABLEAU_PYTHON)" || (echo "Run make setup-tableau first" && exit 1)
	$(TABLEAU_PYTHON) scripts/build_tableau_workbook.py

frontend-check:
	cd frontend && npm run type-check
	cd frontend && npm run lint:publication
	cd frontend && npm run lint
	cd frontend && npm run build

publication-check:
	$(VENV_PYTHON) scripts/audit_publication.py

verify: publication-check test frontend-check

clean-generated:
	rm -rf frontend/dist .pytest_cache warehouse/target warehouse/logs warehouse/.user.yml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
