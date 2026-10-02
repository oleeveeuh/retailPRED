"""Shared pytest fixtures/path setup for RetailPRED tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# backend modules import each other as `from ml... import ...` / `from db... import ...`,
# so the backend directory must be on sys.path
for p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
