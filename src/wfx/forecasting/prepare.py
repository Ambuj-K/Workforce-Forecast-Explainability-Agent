"""Turn reported volumes into a clean, complete daily series per store and driver.

Steps, each recorded in a quality report so the agent can later disclose them:
1. Build the full grid of trading-life days per store x driver.
2. Remove duplicate loads (exact repeated rows).
3. Mark days with no reported volume as missing (gaps, late feeds).
4. Detect spikes and dropouts against a same-weekday baseline (confirmed against
   the recent level) and exclude them from training. Stores in their first weeks
   of trading are not judged: their ramp-up is disclosed instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfx.data.synthetic import SyntheticDataset

KEYS = ["store_id", "driver", "date"]
QUALITY_COLUMNS = ("issue_type", "store_id", "driver", "date", "reported_volume", "baseline", "detail")


@dataclass(frozen=True)
class OutlierRule:
    """Same-weekday baseline rule for flagging suspicious reported values."""

    history_weeks: int = 8  # same-weekday observations used for the baseline
    min_history: int = 3  # need at least this many to judge
    spike_ratio: float = 4.0  # value > ratio x same-weekday baseline is a spike candidate
    recent_days: int = 7  # previous days used for the recent-level baseline
    recent_ratio: float = 2.0  # ...confirmed only if also > ratio x recent level (rules out ramp-ups)
    dropout_min_baseline: float = 10.0  # a zero on a trading day with baseline >= this is a dropout
    min_days_trading: int = 91  # new stores are not judged while they ramp up (disclosed as ramp-up instead)


@dataclass(frozen=True)
class PreparedData:
    """Daily grid with the cleaned target and a report of every quality issue found."""

    frame: pd.DataFrame  # store_id, department, driver, date, reported_volume, target, flags
    quality: pd.DataFrame  # one row per issue


def prepare_observations(ds: SyntheticDataset, rule: OutlierRule | None = None) -> PreparedData:
    """Clean the reported volumes of ``ds``. Never reads the ground-truth tables."""
    rule = rule or OutlierRule()
    issues: list[dict[str, object]] = []

    reported, duplicate_issues = _drop_duplicate_loads(ds.volumes)
    issues.extend(duplicate_issues)

    frame = _trading_grid(ds).merge(reported, on=[*KEYS, "department"], how="left")
    frame = frame.merge(ds.calendar[["date", "is_trading_day"]], on="date", how="left")
    frame = frame.merge(ds.stores[["store_id", "open_date"]], on="store_id", how="left")
    frame = frame.sort_values(KEYS, ignore_index=True)
    frame = frame.rename(columns={"volume": "reported_volume"})

    frame["is_missing"] = frame["reported_volume"].isna()
    for row in frame[frame["is_missing"]].itertuples():
        issues.append(_issue("missing", row.store_id, row.driver, row.date, None, None, "no volume reported"))

    judged = (frame["date"] - frame["open_date"]).dt.days >= rule.min_days_trading
    frame["baseline"] = _same_weekday_baseline(frame, rule).where(judged)
    recent = _recent_baseline(frame, rule)
    frame["is_spike"] = (frame["reported_volume"] > rule.spike_ratio * frame["baseline"]) & (
        frame["reported_volume"] > rule.recent_ratio * recent
    )
    frame["is_dropout"] = (
        (frame["reported_volume"] == 0)
        & frame["is_trading_day"]
        & (frame["baseline"] >= rule.dropout_min_baseline)
    )
    for kind, column in (("outlier_spike", "is_spike"), ("outlier_dropout", "is_dropout")):
        for row in frame[frame[column]].itertuples():
            issues.append(
                _issue(
                    kind,
                    row.store_id,
                    row.driver,
                    row.date,
                    row.reported_volume,
                    round(float(row.baseline), 1),
                    f"reported {int(row.reported_volume)} vs same-weekday baseline {row.baseline:.1f}",
                )
            )

    excluded = frame["is_missing"] | frame["is_spike"] | frame["is_dropout"]
    frame["target"] = frame["reported_volume"].where(~excluded)
    frame = frame.drop(columns=["baseline", "open_date"])

    quality = pd.DataFrame(issues, columns=list(QUALITY_COLUMNS))
    return PreparedData(frame=frame, quality=quality)


def _drop_duplicate_loads(volumes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    duplicated = volumes.duplicated(keep="first")
    issues = [
        _issue("duplicate_load", r.store_id, r.driver, r.date, r.volume, None, "duplicate row removed")
        for r in volumes[duplicated].itertuples()
    ]
    return volumes[~duplicated].reset_index(drop=True), issues


def _trading_grid(ds: SyntheticDataset) -> pd.DataFrame:
    """Every day each store traded (inside the data range) x every driver."""
    start, end = ds.calendar["date"].min(), ds.volumes["date"].max()  # calendar extends into the future
    drivers = ds.labour_standards[["department", "driver"]]
    parts = []
    for store in ds.stores.itertuples():
        first = max(start, store.open_date)
        last = end if pd.isna(store.close_date) else min(end, store.close_date - pd.Timedelta(days=1))
        if first > last:
            continue
        days = pd.DataFrame({"date": pd.date_range(first, last, freq="D"), "store_id": store.store_id})
        parts.append(days.merge(drivers, how="cross"))
    return pd.concat(parts, ignore_index=True)[["store_id", "department", "driver", "date"]]


def _same_weekday_baseline(frame: pd.DataFrame, rule: OutlierRule) -> pd.Series:
    """Median of the previous ``history_weeks`` same-weekday reported values (excluding today)."""
    return _lagged_median(frame, [7 * w for w in range(1, rule.history_weeks + 1)], rule.min_history)


def _recent_baseline(frame: pd.DataFrame, rule: OutlierRule) -> pd.Series:
    """Median of the previous ``recent_days`` reported values; tracks fast ramp-ups."""
    return _lagged_median(frame, list(range(1, rule.recent_days + 1)), rule.min_history)


def _lagged_median(frame: pd.DataFrame, lags: list[int], min_obs: int) -> pd.Series:
    series = frame.groupby(["store_id", "driver"])["reported_volume"]
    history = np.column_stack([series.shift(lag).to_numpy(dtype=float) for lag in lags])
    enough = np.sum(~np.isnan(history), axis=1) >= min_obs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are handled by ``enough``
        median = np.nanmedian(history, axis=1)
    return pd.Series(np.where(enough, median, np.nan), index=frame.index)


def _issue(
    issue_type: str,
    store_id: str,
    driver: str,
    day: pd.Timestamp,
    reported: float | None,
    baseline: float | None,
    detail: str,
) -> dict[str, object]:
    return {
        "issue_type": issue_type,
        "store_id": store_id,
        "driver": driver,
        "date": day,
        "reported_volume": reported,
        "baseline": baseline,
        "detail": detail,
    }
