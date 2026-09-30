"""Leakage-safe features for forecasting daily workload up to ``horizon_days`` ahead.

Rule: a row dated ``t`` may use
* planned / known-in-advance inputs dated ``t`` (calendar, holidays, promotions,
  prices, store facts), and
* past volume only from ``t - horizon_days`` or earlier,
so one model can forecast every day of the horizon from a single forecast origin.

Each feature carries a group and a description. The groups are the coarse
buckets the explainer reports contributions in; the descriptions are its feature
dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfx.data.synthetic import SyntheticDataset

DEFAULT_HORIZON_DAYS = 28


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("day_of_week", "calendar", "Day of the week (0 = Monday)."),
    FeatureSpec("day_of_year_sin", "calendar", "Position in the year (sine), for the yearly cycle."),
    FeatureSpec("day_of_year_cos", "calendar", "Position in the year (cosine), for the yearly cycle."),
    FeatureSpec("is_trading_day", "holiday", "Whether stores trade that day (false on closure days)."),
    FeatureSpec("holiday_code", "holiday", "Which holiday the day is, if any (-1 = none)."),
    FeatureSpec("days_to_christmas", "holiday", "Days until Christmas Day, capped at 31."),
    FeatureSpec("on_promotion", "promotion", "Whether the department runs a planned promotion that day."),
    FeatureSpec("price_index", "price", "Planned relative price level of the department (1.0 = start)."),
    FeatureSpec("price_change_28d", "price", "Planned price level vs 28 days earlier (ratio)."),
    FeatureSpec("size_factor", "store", "Relative store size."),
    FeatureSpec("region_code", "store", "Store region."),
    FeatureSpec("days_since_open", "store", "Days the store has been trading, capped at 730 (captures ramp-up)."),
    FeatureSpec("lag_h", "recent_level", "Volume exactly one horizon earlier (same weekday)."),
    FeatureSpec("same_weekday_mean_4w", "recent_level", "Mean volume on the same weekday over the 4 weeks ending one horizon earlier."),
    FeatureSpec("rolling_mean_28", "recent_level", "Mean daily volume over the 28 days ending one horizon earlier."),
    FeatureSpec("rolling_std_28", "recent_level", "Day-to-day variability over the 28 days ending one horizon earlier."),
    FeatureSpec("lag_364", "seasonal_history", "Volume on the same weekday 52 weeks earlier."),
    FeatureSpec("lag_364_mean_7", "seasonal_history", "Mean volume over the week around the same date last year."),
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURES)
FEATURE_GROUPS: dict[str, str] = {spec.name: spec.group for spec in FEATURES}
KNOWN_IN_ADVANCE: frozenset[str] = frozenset(
    spec.name for spec in FEATURES if spec.group in {"calendar", "holiday", "promotion", "price", "store"}
)

_REGIONS = {"north": 0, "south": 1, "east": 2, "west": 3}


def build_features(
    prepared: pd.DataFrame, ds: SyntheticDataset, horizon_days: int = DEFAULT_HORIZON_DAYS
) -> pd.DataFrame:
    """Add every feature in ``FEATURES`` to the prepared daily grid.

    ``prepared`` is ``PreparedData.frame``: one row per store x driver x day with
    ``target`` (clean volume, NaN where excluded). Returns a copy with feature columns.
    """
    if horizon_days < 7 or horizon_days % 7:
        raise ValueError("horizon_days must be a positive multiple of 7 (keeps weekday alignment)")
    frame = prepared.sort_values(["store_id", "driver", "date"], ignore_index=True).copy()
    frame = _add_known_in_advance(frame, ds)
    frame = _add_history(frame, horizon_days)
    return frame


def _add_known_in_advance(frame: pd.DataFrame, ds: SyntheticDataset) -> pd.DataFrame:
    calendar = ds.calendar[["date", "day_of_week", "holiday"]].copy()
    day_of_year = calendar["date"].dt.dayofyear.to_numpy()
    calendar["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    calendar["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    names = sorted(calendar["holiday"].dropna().unique())
    calendar["holiday_code"] = calendar["holiday"].map({name: i for i, name in enumerate(names)}).fillna(-1).astype("int64")
    christmas = pd.to_datetime(calendar["date"].dt.year.astype(str) + "-12-25")
    to_christmas = (christmas - calendar["date"]).dt.days
    calendar["days_to_christmas"] = to_christmas.where(to_christmas.between(0, 31), 31).astype("int64")
    frame = frame.merge(calendar.drop(columns=["holiday"]), on="date", how="left")

    prices = ds.prices.sort_values(["department", "date"]).copy()
    prices["price_change_28d"] = prices["price_index"] / prices.groupby("department")["price_index"].shift(28)
    prices["price_change_28d"] = prices["price_change_28d"].fillna(1.0)
    frame = frame.merge(ds.promotions, on=["date", "department"], how="left")
    frame = frame.merge(prices, on=["date", "department"], how="left")

    stores = ds.stores[["store_id", "size_factor", "region", "open_date"]]
    frame = frame.merge(stores, on="store_id", how="left")
    frame["region_code"] = frame["region"].map(_REGIONS).astype("int64")
    frame["days_since_open"] = (frame["date"] - frame["open_date"]).dt.days.clip(upper=730)
    return frame.drop(columns=["region", "open_date"])


def _add_history(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """Past-volume features, all ending at least ``h`` days before the row's date."""
    series = frame.groupby(["store_id", "driver"])["target"]
    frame["lag_h"] = series.shift(h)
    same_weekday = [series.shift(h + 7 * w) for w in range(4)]
    frame["same_weekday_mean_4w"] = pd.concat(same_weekday, axis=1).mean(axis=1, skipna=True)
    shifted = series.shift(h)
    frame["rolling_mean_28"] = shifted.groupby([frame["store_id"], frame["driver"]]).transform(
        lambda s: s.rolling(28, min_periods=14).mean()
    )
    frame["rolling_std_28"] = shifted.groupby([frame["store_id"], frame["driver"]]).transform(
        lambda s: s.rolling(28, min_periods=14).std()
    )
    frame["lag_364"] = series.shift(364)
    frame["lag_364_mean_7"] = pd.concat([series.shift(364 + k) for k in range(-3, 4)], axis=1).mean(axis=1, skipna=True)
    return frame
