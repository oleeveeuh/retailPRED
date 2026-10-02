"""Configuration loading without real secrets.

Guards the fix for the hardcoded Census/MRTS API key (config.py). The key must
only ever come from the environment; the source must not contain an embedded
credential.
"""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


def test_config_imports_cleanly():
    """config.py must import without env vars or a real .env present."""
    assert config.PROJECT_ROOT.exists()
    assert config.DATABASE_PATH.name == "retailpred.db"


def test_mrts_key_empty_by_default(monkeypatch):
    """With no MRTS_API_KEY in the environment the default must be empty."""
    monkeypatch.delenv("MRTS_API_KEY", raising=False)
    import importlib

    importlib.reload(config)
    assert config.MRTS_API_KEY == ""


def test_mrts_key_from_environment(monkeypatch):
    monkeypatch.setenv("MRTS_API_KEY", "test-key-123")
    import importlib

    importlib.reload(config)
    assert config.MRTS_API_KEY == "test-key-123"


def test_no_hardcoded_census_key():
    """Regression test: the Census key must not reappear as a source literal."""
    source = (REPO_ROOT / "config.py").read_text()
    # 40+ char hex literals are credential-shaped; the tracked config must have none
    assert not re.search(r'["\'][0-9a-f]{40,}["\']', source), (
        "config.py contains a credential-shaped hex literal"
    )


def test_no_machine_specific_absolute_paths():
    source = (REPO_ROOT / "config.py").read_text()
    assert "/home/oliau" not in source
    assert "/Users/olivialiau" not in source
