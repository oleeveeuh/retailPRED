PYTHON ?= python3.11
VENV ?= .venv-v2
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
NPM_CACHE ?= /tmp/retailpred-npm-cache

.PHONY: setup lock test frontend-check verify clean-generated

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

frontend-check:
	cd frontend && npm run type-check
	cd frontend && npm run lint
	cd frontend && npm run build

verify: test frontend-check

clean-generated:
	rm -rf frontend/dist .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
