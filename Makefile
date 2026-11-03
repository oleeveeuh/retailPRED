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
NPM_CACHE ?= /tmp/retailpred-npm-cache

.PHONY: setup setup-experiment setup-neural setup-foundation lock test milestone1-check platform-check warehouse-check fetch-mrts baseline-mrts benchmark-mrts track-mrts probabilistic-mrts regime-mrts validate-m5 baseline-m5 benchmark-m5 rolling-m5 reconcile-m5 rolling-reconcile-m5 feasibility-nhits-m5 feasibility-chronos-m5 benchmark-chronos-m5 probabilistic-m5 frontend-check verify clean-generated

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

frontend-check:
	cd frontend && npm run type-check
	cd frontend && npm run lint
	cd frontend && npm run build

verify: test frontend-check

clean-generated:
	rm -rf frontend/dist .pytest_cache warehouse/target warehouse/logs warehouse/.user.yml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
