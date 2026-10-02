"""Tests for the explainability extract (contract, as-of correctness) and the locked-down agent database."""

from dataclasses import replace
from datetime import date

import duckdb
import pandas as pd
import pytest

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import Extract, ExtractContractError, build_extract, check_invariants
from wfx.explain.store import build_database, connect_readonly
from wfx.forecasting.model import ModelParams

FAST = ModelParams(max_iter=60)


@pytest.fixture(scope="module")
def ds():
    return generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 9, 29), n_stores=5, seed=3))


@pytest.fixture(scope="module")
def extract(ds):
    return build_extract(ds, n_backtest_runs=2, params=FAST)


@pytest.fixture(scope="module")
def live(extract):
    return extract.runs[extract.runs["is_live"]].iloc[0]


@pytest.fixture(scope="module")
def db(extract, tmp_path_factory):
    return build_database(extract, tmp_path_factory.mktemp("agent") / "agent.duckdb")


# ---------------------------------------------------------------------- extract


def test_extract_satisfies_its_own_contract(extract):
    assert check_invariants(extract) == []


def test_one_live_run_the_day_after_the_data(extract, ds, live):
    assert extract.runs["is_live"].sum() == 1
    assert live["origin"] == ds.volumes["date"].max() + pd.Timedelta(days=1)
    assert live["data_through"] < live["origin"]


def test_live_forecast_covers_the_planned_future(extract, live, ds):
    rows = extract.forecasts[extract.forecasts["run_id"] == live["run_id"]]
    assert rows["date"].min() == live["origin"]
    assert rows["lead_days"].max() == live["horizon_days"] - 1
    # every live forecast row is a store-day that really happens (the answer key exists)
    keys = ["store_id", "driver", "date"]
    assert len(rows.merge(ds.future_true_volumes[keys], on=keys)) == len(rows)


def test_extract_never_reads_ground_truth(ds, extract):
    blind = replace(
        ds,
        true_volumes=ds.true_volumes.iloc[0:0],
        caveats=ds.caveats.iloc[0:0],
        future_true_volumes=ds.future_true_volumes.iloc[0:0],
    )
    again = build_extract(blind, n_backtest_runs=2, params=FAST)
    for name in ("forecasts", "eligibility", "accuracy", "quality_issues"):
        pd.testing.assert_frame_equal(getattr(again, name), getattr(extract, name))


def test_accuracy_is_as_of_the_live_run(extract, live):
    assert (extract.accuracy["window_end"] < live["origin"]).all()
    assert set(extract.accuracy["scored_run_id"]) == set(extract.runs.loc[~extract.runs["is_live"], "run_id"])


def test_live_eligibility_flags_the_late_feed_store(extract, ds, live):
    late_store = ds.caveats.loc[ds.caveats["caveat_type"] == "late_feed", "store_id"].iloc[0]  # answer key, test only
    flags = extract.eligibility[extract.eligibility["run_id"] == live["run_id"]]
    stale = set(flags.loc[flags["data_is_stale"], "store_id"])
    assert stale == {late_store}


def test_tampered_contribution_breaks_the_contract(extract):
    broken = extract.contributions.copy()
    broken.loc[broken.index[0], "contribution"] += 5.0
    with pytest.raises(ExtractContractError, match="base \\+ contributions"):
        from wfx.explain.extract import validate

        validate(replace(extract, contributions=broken))


def test_write_and_load_round_trip_with_validation(extract, tmp_path):
    extract.write(tmp_path)
    loaded = Extract.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.forecasts, extract.forecasts, check_dtype=False)


# --------------------------------------------------------------------- database


def test_agent_can_query_every_table(db, extract):
    con = connect_readonly(db)
    for name, frame in extract.tables().items():
        assert con.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == len(frame)
    con.close()


@pytest.mark.parametrize(
    "attack",
    [
        "CREATE TABLE x AS SELECT 1",
        "DELETE FROM runs",
        "SELECT * FROM read_csv('{secret}')",
        "SELECT * FROM '{secret}'",
        "COPY runs TO '{out}'",
        "SET enable_external_access = true",
        "ATTACH '{other}' AS other",
        "INSTALL httpfs",
    ],
)
def test_locked_down_connection_blocks_escapes(db, tmp_path, attack):
    secret = tmp_path / "secret.env"
    secret.write_text("API_KEY,leaked\n")
    sql = attack.format(secret=secret, out=tmp_path / "exfil.csv", other=tmp_path / "other.duckdb")
    con = connect_readonly(db)
    with pytest.raises(duckdb.Error):
        con.execute(sql).fetchall()
    con.close()
    assert not (tmp_path / "exfil.csv").exists()
