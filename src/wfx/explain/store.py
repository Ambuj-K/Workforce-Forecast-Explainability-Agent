"""The agent's query layer: a DuckDB database file built from a validated extract.

The agent only ever gets a connection from ``connect_readonly``:
* the database file is opened read-only (no DDL, no inserts),
* external access is disabled (no reading local files, URLs or other databases),
* the configuration is locked (the session cannot re-enable any of the above).
SQL written by an LLM therefore cannot escape the extract, even if it is malicious.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from wfx.explain.extract import Extract, validate

READONLY_CONFIG = {"enable_external_access": False, "lock_configuration": True}


def build_database(extract: Extract, db_path: Path) -> Path:
    """Validate ``extract`` and write it into a fresh DuckDB file at ``db_path``."""
    validate(extract)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        for name, frame in extract.tables().items():
            con.register("incoming", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")
    finally:
        con.close()
    return db_path


def connect_readonly(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open the agent's database: read-only, no external access, configuration locked."""
    if not db_path.exists():
        raise FileNotFoundError(f"no agent database at {db_path}; run build_database first")
    return duckdb.connect(str(db_path), read_only=True, config=READONLY_CONFIG)
