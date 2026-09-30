"""Tests for data preparation (cleaning + detection, scored against ground truth) and features."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wfx.data.synthetic import generate
from wfx.forecasting.features import (
    DEFAULT_HORIZON_DAYS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    KNOWN_IN_ADVANCE,
    build_features,
)
from wfx.forecasting.prepare import OutlierRule, prepare_observations

KEYS = ["store_id", "driver", "date"]
HISTORY = [name for name in FEATURE_NAMES if name not in KNOWN_IN_ADVANCE]


@pytest.fixture(scope="module")
def ds():
    return generate()


@pytest.fixture(scope="module")
def prepared(ds):
    return prepare_observations(ds)


@pytest.fixture(scope="module")
def features(prepared, ds):
    return build_features(prepared.frame, ds)


def caveat_days(ds, caveat_type: str) -> pd.DataFrame:
    """Expand caveats of one type to (store_id, driver, date) rows, as in the reported data."""
    rows = []
    drivers = ds.labour_standards["driver"].tolist()
    for c in ds.caveats[ds.caveats["caveat_type"] == caveat_type].itertuples():
        for day in pd.date_range(c.start_date, c.end_date):
            for driver in [c.driver] if isinstance(c.driver, str) else drivers:
                rows.append({"store_id": c.store_id, "driver": driver, "date": day})
    return pd.DataFrame(rows, columns=KEYS)


def issues(prepared, *types: str) -> pd.DataFrame:
    return prepared.quality[prepared.quality["issue_type"].isin(types)][KEYS]


# ---------------------------------------------------------------- preparation


def test_grid_covers_every_trading_day_and_driver_once(ds, prepared):
    start, end = ds.calendar["date"].min(), ds.calendar["date"].max()
    expected = 0
    for store in ds.stores.itertuples():
        first = max(start, store.open_date)
        last = end if pd.isna(store.close_date) else min(end, store.close_date - pd.Timedelta(days=1))
        expected += ((last - first).days + 1) * len(ds.labour_standards)
    assert len(prepared.frame) == expected
    assert not prepared.frame.duplicated(KEYS).any()


def test_duplicate_loads_removed_and_reported(ds, prepared):
    expected = ds.caveats.loc[ds.caveats["caveat_type"] == "duplicate_load", "affected_rows"].sum()
    assert len(issues(prepared, "duplicate_load")) == expected


def test_missing_days_match_gaps_and_late_feed_exactly(ds, prepared):
    expected = pd.concat([caveat_days(ds, "gap"), caveat_days(ds, "late_feed")]).sort_values(KEYS, ignore_index=True)
    found = issues(prepared, "missing").sort_values(KEYS, ignore_index=True)
    pd.testing.assert_frame_equal(found, expected, check_dtype=False)


def test_outlier_detection_scores_perfectly_on_default_data(ds, prepared):
    injected = caveat_days(ds, "outlier")
    detected = issues(prepared, "outlier_spike", "outlier_dropout")
    assert len(injected.merge(detected)) == len(injected)  # recall 1.0
    assert len(detected) == len(injected)  # no false positives


def test_new_stores_are_not_judged_during_ramp_up(ds, prepared):
    rule = OutlierRule()
    flagged = issues(prepared, "outlier_spike", "outlier_dropout").merge(ds.stores, on="store_id")
    assert ((flagged["date"] - flagged["open_date"]).dt.days >= rule.min_days_trading).all()


def test_target_excludes_exactly_the_flagged_rows(prepared):
    frame = prepared.frame
    excluded = frame["is_missing"] | frame["is_spike"] | frame["is_dropout"]
    assert frame.loc[excluded, "target"].isna().all()
    assert frame.loc[~excluded, "target"].notna().all()
    assert (frame.loc[~excluded, "target"] == frame.loc[~excluded, "reported_volume"]).all()


def test_preparation_never_reads_ground_truth(ds, prepared):
    blind = replace(ds, true_volumes=ds.true_volumes.iloc[0:0], caveats=ds.caveats.iloc[0:0])
    again = prepare_observations(blind)
    pd.testing.assert_frame_equal(again.frame, prepared.frame)
    pd.testing.assert_frame_equal(again.quality, prepared.quality)


# ------------------------------------------------------------------- features


def test_every_feature_has_a_column_and_a_known_group(features):
    assert set(FEATURE_NAMES) <= set(features.columns)
    assert set(FEATURE_GROUPS.values()) == {
        "calendar", "holiday", "promotion", "price", "store", "recent_level", "seasonal_history"
    }


def test_known_in_advance_features_are_always_available(features):
    assert features[sorted(KNOWN_IN_ADVANCE)].notna().all().all()


def test_lag_feature_is_the_target_one_horizon_earlier(features):
    series = features[(features["store_id"] == "S001") & (features["driver"] == "transactions")].set_index("date")
    day = series.index[400]
    earlier = day - pd.Timedelta(days=DEFAULT_HORIZON_DAYS)
    assert series.loc[day, "lag_h"] == series.loc[earlier, "target"]


def test_history_features_never_see_the_forecast_window(prepared, ds, features):
    """Changing volumes on or after a forecast origin must not change any history
    feature for dates less than one horizon after that origin."""
    origin = pd.Timestamp("2024-06-03")
    tampered = prepared.frame.copy()
    future = tampered["date"] >= origin
    tampered.loc[future, "target"] = tampered.loc[future, "target"] * 10 + 1
    again = build_features(tampered, ds)
    visible = features["date"] < origin + pd.Timedelta(days=DEFAULT_HORIZON_DAYS)
    pd.testing.assert_frame_equal(
        features.loc[visible, HISTORY].reset_index(drop=True),
        again.loc[visible, HISTORY].reset_index(drop=True),
    )
    # ...and the tampering does reach rows further out, so the test has teeth.
    beyond = features["date"] >= origin + pd.Timedelta(days=DEFAULT_HORIZON_DAYS)
    assert not np.allclose(features.loc[beyond, "lag_h"].fillna(0), again.loc[beyond, "lag_h"].fillna(0))


def test_horizon_must_keep_weekday_alignment(prepared, ds):
    with pytest.raises(ValueError, match="multiple of 7"):
        build_features(prepared.frame, ds, horizon_days=10)
