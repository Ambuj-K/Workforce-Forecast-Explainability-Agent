"""Tests for the synthetic workload generator: determinism, structure and every injected caveat."""

from datetime import date

import pandas as pd
import pytest

from wfx.data.config import CaveatConfig, GeneratorConfig
from wfx.data.synthetic import CAVEAT_COLUMNS, generate

KEYS = ["date", "store_id", "driver"]


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 6, 30), n_stores=6, seed=11)


@pytest.fixture(scope="module")
def ds(config):
    return generate(config)


def caveats_of(ds, caveat_type: str) -> pd.DataFrame:
    return ds.caveats[ds.caveats["caveat_type"] == caveat_type]


def test_same_seed_same_data(config, ds):
    again = generate(config)
    pd.testing.assert_frame_equal(ds.volumes, again.volumes)
    pd.testing.assert_frame_equal(ds.caveats, again.caveats)


def test_different_seed_different_data(config, ds):
    other = generate(GeneratorConfig(start=config.start, end=config.end, n_stores=6, seed=12))
    assert not ds.true_volumes["volume"].equals(other.true_volumes["volume"])


def test_rejects_impossible_store_mix():
    with pytest.raises(ValueError, match="neither new nor closed"):
        generate(GeneratorConfig(n_stores=3, n_new_stores=2, n_closed_stores=1))


def test_tables_have_expected_columns(ds):
    assert list(ds.volumes.columns) == ["date", "store_id", "department", "driver", "volume"]
    assert list(ds.caveats.columns) == list(CAVEAT_COLUMNS)
    assert {"is_trading_day", "holiday", "week_start"} <= set(ds.calendar.columns)
    assert list(ds.labour_standards["driver"]) == [spec.driver for spec in GeneratorConfig().drivers]


def test_true_volumes_are_unique_and_non_negative(ds):
    assert not ds.true_volumes.duplicated(KEYS).any()
    assert (ds.true_volumes["volume"] >= 0).all()


def test_no_volumes_outside_store_trading_life(ds):
    merged = ds.true_volumes.merge(ds.stores, on="store_id")
    assert (merged["date"] >= merged["open_date"]).all()
    closed = merged[merged["close_date"].notna()]
    assert (closed["date"] < closed["close_date"]).all()


def test_non_trading_days_have_zero_volume(ds):
    non_trading = ds.calendar.loc[~ds.calendar["is_trading_day"], "date"]
    assert len(non_trading) > 0
    assert (ds.true_volumes.loc[ds.true_volumes["date"].isin(non_trading), "volume"] == 0).all()


def test_new_store_ramps_up(ds, config):
    ramp = caveats_of(ds, "ramp_up").iloc[0]
    store = ds.true_volumes[
        (ds.true_volumes["store_id"] == ramp["store_id"]) & (ds.true_volumes["driver"] == "transactions")
    ]
    days_open = (store["date"] - ramp["start_date"]).dt.days
    first_fortnight = store.loc[days_open < 14, "volume"].mean()
    after_ramp = store.loc[(days_open >= config.ramp_up_days) & (days_open < config.ramp_up_days + 28), "volume"].mean()
    assert first_fortnight < 0.5 * after_ramp


def test_promotions_lift_grocery_volume(ds):
    grocery = ds.true_volumes[ds.true_volumes["driver"] == "cases_filled"].merge(
        ds.promotions[ds.promotions["department"] == "grocery"], on=["date", "department"]
    )
    grocery = grocery[grocery["volume"] > 0]
    assert grocery.loc[grocery["on_promotion"], "volume"].mean() > grocery.loc[~grocery["on_promotion"], "volume"].mean()


def test_every_caveat_type_is_injected(ds, config):
    counts = ds.caveats["caveat_type"].value_counts()
    assert counts["gap"] == config.caveats.n_gaps
    assert counts["outlier"] == config.caveats.n_outliers
    assert counts["duplicate_load"] == config.caveats.n_duplicate_days
    assert counts["late_feed"] == 1
    assert counts["ramp_up"] == config.n_new_stores
    assert counts["store_closed"] == config.n_closed_stores


def test_gaps_are_missing_from_reported_but_present_in_truth(ds):
    for gap in caveats_of(ds, "gap").itertuples():
        in_window = lambda df: df[  # noqa: E731
            (df["store_id"] == gap.store_id)
            & (df["driver"] == gap.driver)
            & df["date"].between(gap.start_date, gap.end_date)
        ]
        assert in_window(ds.volumes).empty
        assert len(in_window(ds.true_volumes)) == gap.affected_rows > 0


def test_late_feed_drops_most_recent_days(ds):
    late = caveats_of(ds, "late_feed").iloc[0]
    reported = ds.volumes[ds.volumes["store_id"] == late["store_id"]]
    assert reported["date"].max() < late["start_date"]
    assert late["end_date"] == ds.true_volumes["date"].max()


def test_outliers_differ_from_truth_as_recorded(ds):
    merged = ds.volumes.drop_duplicates(KEYS).merge(ds.true_volumes, on=KEYS, suffixes=("_reported", "_true"))
    for outlier in caveats_of(ds, "outlier").itertuples():
        row = merged[
            (merged["store_id"] == outlier.store_id)
            & (merged["driver"] == outlier.driver)
            & (merged["date"] == outlier.start_date)
        ].iloc[0]
        assert row["volume_reported"] != row["volume_true"]
        assert f"reported {row['volume_reported']}, true {row['volume_true']}" in outlier.detail


def test_duplicates_are_exact_copies_of_one_store_day(ds):
    dupes = ds.volumes[ds.volumes.duplicated(keep=False)]
    for dup in caveats_of(ds, "duplicate_load").itertuples():
        store_day = dupes[(dupes["store_id"] == dup.store_id) & (dupes["date"] == dup.start_date)]
        assert len(store_day) == 2 * dup.affected_rows
    assert len(dupes) == 2 * caveats_of(ds, "duplicate_load")["affected_rows"].sum()


def test_only_injected_rows_differ_from_truth(ds):
    """Outside the recorded caveats, reported volumes match the truth exactly."""
    merged = ds.volumes.drop_duplicates(KEYS).merge(ds.true_volumes, on=KEYS, suffixes=("_reported", "_true"))
    changed = merged[merged["volume_reported"] != merged["volume_true"]]
    assert len(changed) == len(caveats_of(ds, "outlier"))


def test_caveats_never_overlap_on_a_store_day(ds):
    days = []
    for c in ds.caveats[ds.caveats["caveat_type"].isin(["gap", "outlier", "duplicate_load", "late_feed"])].itertuples():
        for d in pd.date_range(c.start_date, c.end_date):
            days.append((c.store_id, d))
    assert len(days) == len(set(days))


def test_write_round_trips(ds, tmp_path):
    paths = ds.write(tmp_path)
    assert set(paths) == {
        "calendar", "stores", "promotions", "prices", "true_volumes", "volumes", "labour_standards", "caveats"
    }
    pd.testing.assert_frame_equal(pd.read_parquet(paths["volumes"]), ds.volumes, check_dtype=False)


def test_small_caveat_config_is_respected():
    small = generate(
        GeneratorConfig(
            end=date(2023, 12, 31),
            n_stores=4,
            caveats=CaveatConfig(n_gaps=1, n_duplicate_days=1, n_outliers=2),
        )
    )
    counts = small.caveats["caveat_type"].value_counts()
    assert (counts["gap"], counts["duplicate_load"], counts["outlier"]) == (1, 1, 2)
