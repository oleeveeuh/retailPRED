PYTHON ?= python3.11
VENV ?= .venv-v2
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
NPM_CACHE ?= /tmp/retailpred-npm-cache

.PHONY: setup lock test milestone1-check platform-check fetch-mrts baseline-mrts frontend-check verify clean-generated

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install pip==25.3
	$(VENV_PIP) install -r requirements-dev.lock
	NPM_CONFIG_CACHE=$(NPM_CACHE) npm ci
	cd frontend && NPM_CONFIG_CACHE=$(NPM_CACHE) npm ci

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

fetch-mrts:
	@test -n "$(END_YEAR)" || (echo "END_YEAR is required, e.g. make fetch-mrts END_YEAR=2025" && exit 1)
	$(VENV_PYTHON) -m scripts.fetch_mrts_v2 --end-year $(END_YEAR)

baseline-mrts:
	@test -n "$(WORKBOOK)" || (echo "WORKBOOK is required" && exit 1)
	@test -n "$(VINTAGE_TIME)" || (echo "VINTAGE_TIME is required, including a UTC offset" && exit 1)
	$(VENV_PYTHON) -m scripts.run_mrts_baseline "$(WORKBOOK)" --vintage-time "$(VINTAGE_TIME)"

frontend-check:
	cd frontend && npm run type-check
	cd frontend && npm run lint
	cd frontend && npm run build

verify: test frontend-check

clean-generated:
	rm -rf frontend/dist .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
