"""Tests for the volume -> weekly hours conversion and its exact explanation."""

from datetime import date

import pandas as pd
import pytest

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.forecasting.backtest import BacktestConfig, forecast_origins, walk_forward
from wfx.forecasting.features import build_features
from wfx.forecasting.hours import WEEK_KEYS, hours_accuracy, weekly_hours, weekly_hours_contributions
from wfx.forecasting.model import ModelParams, explain, predict, train
from wfx.forecasting.prepare import prepare_observations

FAST = ModelParams(max_iter=60)
LABOUR = pd.DataFrame(
    {
        "department": ["grocery", "grocery", "checkouts"],
        "driver": ["cases_filled", "shelf_gaps", "transactions"],
        "minutes_per_unit": [1.5, 3.0, 1.0],
        "fixed_hours_per_week": [14.0, 7.0, 21.0],
        "standard_version": "test",
    }
)


def daily(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["store_id", "department", "driver", "date", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_hand_computed_week():
    week = daily(
        [("S1", "grocery", "cases_filled", f"2025-01-{d:02d}", 40.0) for d in range(6, 13)]
        + [("S1", "grocery", "shelf_gaps", f"2025-01-{d:02d}", 10.0) for d in range(6, 13)]
    )
    row = weekly_hours(week, LABOUR).iloc[0]
    assert row["days"] == 7
    assert row["variable_hours"] == pytest.approx(7 * (40 * 1.5 + 10 * 3.0) / 60)  # 10.5
    assert row["fixed_hours"] == pytest.approx(21.0)  # both grocery standards' fixed hours
    assert row["total_hours"] == pytest.approx(31.5)


def test_partial_week_prorates_fixed_hours():
    opening = daily([("S1", "checkouts", "transactions", f"2025-01-{d:02d}", 60.0) for d in (10, 11, 12)])
    row = weekly_hours(opening, LABOUR).iloc[0]
    assert row["days"] == 3
    assert row["fixed_hours"] == pytest.approx(21.0 * 3 / 7)
    assert row["week_start"] == pd.Timestamp("2025-01-06")


def test_unknown_driver_is_rejected():
    with pytest.raises(ValueError, match="no labour standard"):
        weekly_hours(daily([("S1", "fresh", "units_prepared", "2025-01-06", 5.0)]), LABOUR)


@pytest.fixture(scope="module")
def ds():
    return generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 9, 29), n_stores=4, seed=3))


@pytest.fixture(scope="module")
def features(ds):
    return build_features(prepare_observations(ds).frame, ds)


def test_hour_contributions_add_up_to_forecast_hours_exactly(ds, features):
    origin = forecast_origins(features["date"], BacktestConfig())[-1]
    models = train(features, before=origin, params=FAST)
    window = features[(features["date"] >= origin) & (features["date"] < origin + pd.Timedelta(days=28))].copy()
    window["forecast"] = predict(models, window)
    hours = weekly_hours(window, ds.labour_standards, volume_col="forecast").set_index(WEEK_KEYS)
    parts = weekly_hours_contributions(explain(models, window), ds.labour_standards)
    total = parts.groupby(WEEK_KEYS)["hours"].sum()
    assert (total - hours.loc[total.index, "variable_hours"]).abs().max() < 1e-6
    assert "base" in set(parts["group"])


def test_hours_accuracy_scores_model_and_baseline_by_department(ds, features):
    forecasts = walk_forward(features, ds, BacktestConfig(max_origins=1, params=FAST))
    table = hours_accuracy(forecasts, ds.labour_standards)
    assert set(table["department"]) == set(ds.labour_standards["department"])
    assert set(table["lead_week"]) == {1, 2, 3, 4}
    assert (table["wape_model"] >= 0).all()
