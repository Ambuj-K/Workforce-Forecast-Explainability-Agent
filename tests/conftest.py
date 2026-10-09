"""Shared fixtures."""

from datetime import date

import pytest

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import build_extract
from wfx.explain.store import build_database
from wfx.forecasting.model import ModelParams


@pytest.fixture(scope="session")
def api_db(tmp_path_factory):
    """A small agent database (4 stores + new stores, 2 recent + 3 scored runs) for API-level tests."""
    ds = generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 12, 29), n_stores=4, seed=5))
    extract = build_extract(ds, recent_runs=2, scored_runs=3, params=ModelParams(max_iter=60))
    return build_database(extract, tmp_path_factory.mktemp("api") / "agent.duckdb")
