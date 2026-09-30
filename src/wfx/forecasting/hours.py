"""Convert daily workload volumes into weekly labour hours per store and department.

    variable_hours = sum over days and drivers of volume x minutes_per_unit / 60
    fixed_hours    = fixed_hours_per_week x (days trading in the week / 7)
    total_hours    = fixed_hours + variable_hours

Hours are linear in volume, so per-feature volume contributions convert to exact
per-feature hour contributions:

    base_hours + sum(group hours) == variable_hours   (exact, up to float error)
"""

from __future__ import annotations

import pandas as pd

WEEK_KEYS = ["store_id", "department", "week_start"]


def _with_standards(daily: pd.DataFrame, labour: pd.DataFrame) -> pd.DataFrame:
    unknown = set(daily["driver"]) - set(labour["driver"])
    if unknown:
        raise ValueError(f"no labour standard for drivers: {sorted(unknown)}")
    columns = ["driver", "minutes_per_unit"] + ([] if "department" in daily.columns else ["department"])
    out = daily.merge(labour[columns], on="driver", how="left")
    out["week_start"] = out["date"] - pd.to_timedelta(out["date"].dt.dayofweek, unit="D")
    return out


def weekly_hours(daily: pd.DataFrame, labour: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
    """Weekly hours per store x department from daily volumes in ``volume_col``.

    ``daily`` needs store_id, department, driver, date and ``volume_col``, with one row per
    day the store traded (the rows themselves define partial weeks at openings/closures).
    """
    rows = _with_standards(daily, labour)
    rows["variable_hours"] = rows[volume_col] * rows["minutes_per_unit"] / 60
    weekly = rows.groupby(WEEK_KEYS, as_index=False).agg(
        days=("date", "nunique"),
        volume=(volume_col, "sum"),
        variable_hours=("variable_hours", "sum"),
    )
    fixed = labour.groupby("department", as_index=False).agg(
        fixed_hours_per_week=("fixed_hours_per_week", "sum"),
        standard_version=("standard_version", "first"),
    )
    weekly = weekly.merge(fixed, on="department", how="left")
    weekly["fixed_hours"] = weekly["fixed_hours_per_week"] * weekly["days"] / 7
    weekly["total_hours"] = weekly["fixed_hours"] + weekly["variable_hours"]
    return weekly[[*WEEK_KEYS, "days", "volume", "fixed_hours", "variable_hours", "total_hours", "standard_version"]]


def weekly_hours_contributions(contributions: pd.DataFrame, labour: pd.DataFrame) -> pd.DataFrame:
    """Convert per-row volume contributions (from ``model.explain``) into weekly hours by group.

    Returns store_id, department, week_start, group, hours, where group ``base`` is the
    models' base value converted to hours. Their sum equals the forecast variable hours.
    """
    rows = _with_standards(contributions, labour)
    rate = rows["minutes_per_unit"] / 60
    rows["hours"] = rows["contribution"] * rate
    by_group = rows.groupby([*WEEK_KEYS, "group"], as_index=False)["hours"].sum()

    per_row = rows.drop_duplicates(["store_id", "driver", "date"]).copy()
    per_row["hours"] = per_row["base_value"] * per_row["minutes_per_unit"] / 60
    base = per_row.groupby(WEEK_KEYS, as_index=False)["hours"].sum().assign(group="base")

    out = pd.concat([base, by_group], ignore_index=True)
    return out.sort_values([*WEEK_KEYS, "group"], ignore_index=True)[[*WEEK_KEYS, "group", "hours"]]


def hours_accuracy(forecasts: pd.DataFrame, labour: pd.DataFrame) -> pd.DataFrame:
    """Weekly hours WAPE of model and baseline vs true volumes, by department and lead week.

    ``forecasts`` is ``backtest.walk_forward`` output. Fixed hours are excluded (they are
    identical in all three), so this measures the error planners actually feel.
    """
    scored = forecasts[forecasts["true_volume"].notna() & forecasts["baseline"].notna()].copy()
    scored["lead_week"] = scored["lead_days"] // 7 + 1
    tables = {}
    for col in ("forecast", "baseline", "true_volume"):
        weekly = _with_standards(scored, labour)
        weekly["hours"] = weekly[col] * weekly["minutes_per_unit"] / 60
        tables[col] = weekly.groupby(["origin", *WEEK_KEYS, "lead_week"])["hours"].sum()
    frame = pd.DataFrame(tables).reset_index()
    frame["err_model"] = (frame["forecast"] - frame["true_volume"]).abs()
    frame["err_baseline"] = (frame["baseline"] - frame["true_volume"]).abs()
    out = frame.groupby(["department", "lead_week"], as_index=False).agg(
        weeks=("true_volume", "size"),
        true_hours=("true_volume", "sum"),
        err_model=("err_model", "sum"),
        err_baseline=("err_baseline", "sum"),
    )
    out["wape_model"] = out["err_model"] / out["true_hours"]
    out["wape_baseline"] = out["err_baseline"] / out["true_hours"]
    out["skill"] = 1 - out["wape_model"] / out["wape_baseline"]
    return out[["department", "lead_week", "weeks", "wape_model", "wape_baseline", "skill"]]
