"""Shared pytest fixtures/path setup for RetailPRED tests."""

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# backend modules import each other as `from ml... import ...` / `from db... import ...`,
# so the backend directory must be on sys.path
for p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_database_schema() -> None:
    """Initialize an empty data/retailpred.db from data/db/schema.sql if absent.

    The populated DB is not tracked; CI and clean clones need at least the
    table structure for the API smoke tests (mirrors the README quick start).
    """
    schema_path = REPO_ROOT / "data" / "db" / "schema.sql"
    db_path = REPO_ROOT / "data" / "retailpred.db"
    if not schema_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            has_tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0] > 0
            conn.close()
        except sqlite3.DatabaseError:
            has_tables = False
        if has_tables:
            return
    conn = sqlite3.connect(db_path)
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


_ensure_database_schema()
