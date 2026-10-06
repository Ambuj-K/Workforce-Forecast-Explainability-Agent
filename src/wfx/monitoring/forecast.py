"""Forecast monitoring over runs: accuracy and data-quality trends, plus alert rules.

Reads only the locked-down agent database, so monitoring sees exactly what the agent sees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import duckdb
import pandas as pd


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: str  # warning | critical
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastRules:
    drift_ratio: float = 1.25  # latest run error > ratio x median of earlier runs...
    # ...AND skill vs the baseline below its earlier median. Raw error alone false-alarms in
    # hard periods (e.g. Christmas) where the simple method gets worse too.
    min_history_runs: int = 2  # earlier scored runs needed before judging drift
    consecutive_worse_runs: int = 2  # store-driver worse than baseline this many runs in a row


def accuracy_trend(con: duckdb.DuckDBPyConnection, by_lead_week: bool = False) -> pd.DataFrame:
    """Error of model and baseline per scored run (optionally per lead week), oldest first."""
    group = ", lead_week" if by_lead_week else ""
    return con.execute(
        f"""
        SELECT scored_run_id AS run_id, min(window_end) AS window_end{group},
               sum(abs_err_model) / sum(actual) AS wape_model,
               sum(abs_err_baseline) / sum(actual) AS wape_baseline,
               1 - sum(abs_err_model) / sum(abs_err_baseline) AS skill
        FROM accuracy GROUP BY scored_run_id{group} ORDER BY window_end{group}
        """
    ).fetch_df()


def data_quality_trend(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per run: how much of the data needed caution as of that run's origin."""
    return con.execute(
        """
        SELECT r.run_id, r.origin, r.is_live,
               count(*) FILTER (WHERE e.data_is_stale) AS stale_store_drivers,
               sum(e.missing_days_28) AS missing_days_28,
               sum(e.outliers_excluded_28) AS outliers_excluded_28,
               sum(e.duplicate_rows_removed_28) AS duplicate_rows_removed_28,
               count(*) FILTER (WHERE e.is_ramping) AS ramping_store_drivers
        FROM runs r JOIN eligibility e USING (run_id)
        GROUP BY r.run_id, r.origin, r.is_live ORDER BY r.origin
        """
    ).fetch_df()


def forecast_alerts(con: duckdb.DuckDBPyConnection, rules: ForecastRules | None = None) -> list[Alert]:
    rules = rules or ForecastRules()
    alerts: list[Alert] = []
    trend = accuracy_trend(con)

    if len(trend) > rules.min_history_runs:
        latest, earlier = trend.iloc[-1], trend.iloc[:-1]
        typical, typical_skill = earlier["wape_model"].median(), earlier["skill"].median()
        if latest["wape_model"] > rules.drift_ratio * typical and latest["skill"] < typical_skill:
            alerts.append(Alert(
                "accuracy_drift", "critical",
                f"Run {latest['run_id']}: error {latest['wape_model']:.1%} vs typical {typical:.1%}, and its edge over the "
                f"same-weekday method fell to {latest['skill']:.0%} (typical {typical_skill:.0%}). Check recent data changes.",
            ))
    if len(trend) and trend.iloc[-1]["wape_model"] > trend.iloc[-1]["wape_baseline"]:
        latest = trend.iloc[-1]
        alerts.append(Alert(
            "worse_than_baseline", "critical",
            f"Run {latest['run_id']}: model error {latest['wape_model']:.1%} is worse than the same-weekday method "
            f"({latest['wape_baseline']:.1%}).",
        ))

    by_pair = con.execute(
        """
        SELECT store_id, driver, scored_run_id, min(window_end) AS window_end,
               sum(abs_err_model) > sum(abs_err_baseline) AS worse
        FROM accuracy GROUP BY ALL ORDER BY store_id, driver, window_end
        """
    ).fetch_df()
    for (store, driver), runs in by_pair.groupby(["store_id", "driver"]):
        recent = runs.tail(rules.consecutive_worse_runs)
        if len(recent) == rules.consecutive_worse_runs and recent["worse"].all():
            alerts.append(Alert(
                "persistently_worse_than_baseline", "warning",
                f"{store} {driver}: the model lost to the same-weekday method in the last "
                f"{rules.consecutive_worse_runs} scored runs.",
            ))

    stale = con.execute(
        """
        SELECT e.store_id, count(*) AS drivers, min(e.latest_actual_date) AS latest_actual, any_value(r.origin) AS origin
        FROM eligibility e JOIN runs r USING (run_id)
        WHERE r.is_live AND e.data_is_stale GROUP BY e.store_id ORDER BY e.store_id
        """
    ).fetch_df()
    for row in stale.itertuples():
        lag = (pd.Timestamp(row.origin) - pd.Timestamp(row.latest_actual)).days
        alerts.append(Alert(
            "stale_data_live", "warning",
            f"{row.store_id}: data for {row.drivers} driver(s) is {lag} days old at the live run "
            f"(latest actuals {pd.Timestamp(row.latest_actual).date()}).",
        ))
    return alerts
