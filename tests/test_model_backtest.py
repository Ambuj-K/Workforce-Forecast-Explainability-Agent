"""Tests for the per-driver models, exact contributions and the walk-forward back-test."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.forecasting.backtest import BacktestConfig, accuracy, forecast_origins, seasonal_naive, walk_forward
from wfx.forecasting.features import FEATURE_GROUPS, FEATURE_NAMES, build_features
from wfx.forecasting.model import KEYS, ModelParams, explain, predict, train, training_rows
from wfx.forecasting.prepare import prepare_observations

FAST = ModelParams(max_iter=60)


@pytest.fixture(scope="module")
def ds():
    return generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 9, 29), n_stores=4, seed=3))


@pytest.fixture(scope="module")
def features(ds):
    return build_features(prepare_observations(ds).frame, ds)


@pytest.fixture(scope="module")
def origin(features):
    return forecast_origins(features["date"], BacktestConfig())[-1]


@pytest.fixture(scope="module")
def models(features, origin):
    return train(features, before=origin, params=FAST)


@pytest.fixture(scope="module")
def window(features, origin):
    return features[(features["date"] >= origin) & (features["date"] < origin + pd.Timedelta(days=28))]


@pytest.fixture(scope="module")
def backtest(features, ds):
    return walk_forward(features, ds, BacktestConfig(max_origins=2, params=FAST))


# ---------------------------------------------------------------------- models


def test_one_model_per_driver_trained_only_on_the_past(models, features, origin):
    assert set(models) == set(features["driver"])
    for driver, model in models.items():
        assert model.trained_until < origin
        assert model.n_training_rows == len(training_rows(features, driver, origin))


def test_training_skips_non_trading_days_and_excluded_targets(features, origin):
    rows = training_rows(features, "transactions", origin)
    assert rows["is_trading_day"].all()
    assert rows["target"].notna().all()


def test_training_too_early_names_the_missing_history(features):
    too_early = features["date"].min() + pd.Timedelta(days=200)
    with pytest.raises(ValueError, match="lag_364"):
        train(features, before=too_early, params=FAST)


def test_forecast_is_zero_on_non_trading_days(models, features):
    closed = features[~features["is_trading_day"]]
    assert len(closed) > 0
    assert (predict(models, closed) == 0).all()


def test_contributions_add_up_to_the_forecast_exactly(models, window):
    contributions = explain(models, window)
    totals = contributions.groupby(KEYS).agg(total=("contribution", "sum"), base=("base_value", "first"))
    forecast = window.assign(forecast=predict(models, window)).set_index(KEYS)["forecast"]
    error = (totals["total"] + totals["base"] - forecast.loc[totals.index]).abs().max()
    assert error < 1e-6  # guards against the categorical-split SHAP bug (error was 1,641 units)


def test_every_trading_row_gets_every_feature_once(models, window):
    contributions = explain(models, window)
    trading_rows = int(window["is_trading_day"].sum())
    assert len(contributions) == trading_rows * len(FEATURE_NAMES)
    assert set(contributions["group"]) <= set(FEATURE_GROUPS.values())


# -------------------------------------------------------------------- back-test


def test_origins_are_mondays_a_step_apart_with_a_full_horizon(features):
    origins = forecast_origins(features["date"], BacktestConfig())
    assert all(o.dayofweek == 0 for o in origins)
    assert all((b - a).days == 28 for a, b in zip(origins, origins[1:]))
    assert origins[-1] + pd.Timedelta(days=27) <= features["date"].max()


def test_each_origin_forecasts_exactly_one_horizon(backtest):
    spans = backtest.groupby("origin")["lead_days"].agg(["min", "max"])
    assert (spans["min"] == 0).all() and (spans["max"] == 27).all()


def test_backtest_cannot_see_the_future(features, ds):
    """Tampering with volumes on/after the last origin must not change its forecasts."""
    config = BacktestConfig(max_origins=1, params=FAST)
    last = forecast_origins(features["date"], config)[-1]
    tampered = prepare_observations(ds).frame
    future = tampered["date"] >= last
    tampered.loc[future, "target"] = tampered.loc[future, "target"] * 5
    before = walk_forward(features, ds, config)
    after = walk_forward(build_features(tampered, ds), ds, config)
    np.testing.assert_allclose(before["forecast"], after["forecast"])


def test_seasonal_naive_falls_back_and_respects_closures(window):
    baseline = seasonal_naive(window)
    assert (baseline[~window["is_trading_day"]] == 0).all()
    missing_lag = window["is_trading_day"] & window["lag_h"].isna() & window["same_weekday_mean_4w"].notna()
    assert (baseline[missing_lag] == window.loc[missing_lag, "same_weekday_mean_4w"]).all()


def test_model_beats_the_seasonal_naive_baseline_overall(backtest):
    overall = accuracy(backtest.assign(all="all"), by=("all",)).iloc[0]
    assert overall["wape_model"] < overall["wape_baseline"]


def test_accuracy_reports_skill_by_driver_and_lead_week(backtest):
    table = accuracy(backtest)
    assert set(table["lead_week"]) == {1, 2, 3, 4}
    assert {"wape_model", "wape_baseline", "skill"} <= set(table.columns)
