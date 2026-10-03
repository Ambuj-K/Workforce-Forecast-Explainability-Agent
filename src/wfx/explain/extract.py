"""Explainability extract: the only interface between the forecasting pipeline and the agent.

One extract holds several forecast runs, all on Mondays: the live run made the day after
the last reported data; recent weekly runs whose windows overlap it (for "what changed
since the last run"); and older runs whose windows have fully played out (for accuracy). Every table is
built "as of" its run's origin from reported data only; the ground-truth tables are
never read. The contract (schemas + invariants) is enforced when the extract is built
and again when it is loaded.

Invariants:
* forecast = base_value + sum(contributions) for every trading-day forecast row
* non-trading days are forecast as 0 by rule and have no contributions
* weekly variable hours = sum of weekly hour contributions (including base)
* total hours = fixed hours + variable hours
* exactly one live run; accuracy only from windows that ended before the live origin
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from wfx.data.synthetic import SyntheticDataset
from wfx.forecasting.backtest import seasonal_naive
from wfx.forecasting.features import (
    DEFAULT_HORIZON_DAYS,
    FEATURE_GROUPS,
    FEATURES,
    KNOWN_IN_ADVANCE,
    build_features,
    extend_into_future,
)
from wfx.forecasting.hours import weekly_hours, weekly_hours_contributions
from wfx.forecasting.model import KEYS, ModelParams, explain, predict, train
from wfx.forecasting.prepare import OutlierRule, prepare_observations

TOLERANCE = 1e-6


class ExtractContractError(ValueError):
    """The extract broke its schema or an invariant. ``violations`` lists every problem."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("extract contract violated:\n- " + "\n- ".join(violations))


@dataclass(frozen=True)
class Extract:
    runs: pd.DataFrame
    forecasts: pd.DataFrame
    contributions: pd.DataFrame
    weekly_hours: pd.DataFrame
    weekly_hour_contributions: pd.DataFrame
    eligibility: pd.DataFrame
    quality_issues: pd.DataFrame
    accuracy: pd.DataFrame
    feature_dictionary: pd.DataFrame
    labour_standards: pd.DataFrame

    def write(self, out_dir: Path) -> dict[str, Path]:
        """Validate, then write every table as parquet under ``out_dir``."""
        validate(self)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for table in fields(self):
            path = out_dir / f"{table.name}.parquet"
            getattr(self, table.name).to_parquet(path, index=False)
            paths[table.name] = path
        return paths

    @classmethod
    def load(cls, in_dir: Path) -> Extract:
        """Read an extract and validate it before anyone can use it."""
        extract = cls(**{t.name: pd.read_parquet(in_dir / f"{t.name}.parquet") for t in fields(cls)})
        validate(extract)
        return extract

    def tables(self) -> dict[str, pd.DataFrame]:
        return {t.name: getattr(self, t.name) for t in fields(self)}


# ------------------------------------------------------------------------- build


def build_extract(
    ds: SyntheticDataset,
    recent_runs: int = 3,
    scored_runs: int = 4,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    params: ModelParams | None = None,
) -> Extract:
    """Build and validate an extract: the live run, ``recent_runs`` weekly runs before it and
    ``scored_runs`` runs (one horizon apart) whose windows ended before the live run."""
    params = params or ModelParams()
    prepared = prepare_observations(ds)
    features = build_features(extend_into_future(prepared.frame, ds, horizon_days), ds, horizon_days)
    last_day = ds.volumes["date"].max()
    live_origin = last_day + pd.Timedelta(days=1)

    if live_origin.dayofweek != 0:
        raise ValueError("the live run must start on a Monday (data should end on a Sunday)")
    recent = [live_origin - pd.Timedelta(days=7 * k) for k in range(recent_runs, 0, -1)]
    scored = [live_origin - pd.Timedelta(days=horizon_days * k) for k in range(scored_runs, 0, -1)]
    origins = sorted(set(recent) | set(scored) | {live_origin})

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in ("runs", "forecasts", "contributions", "weekly_hours", "weekly_hour_contributions", "eligibility")}
    for origin in origins:
        run = _build_run(features, prepared.quality, ds, origin, horizon_days, params, is_live=origin == live_origin)
        for name, frame in run.items():
            parts[name].append(frame)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}

    extract = Extract(
        **tables,
        quality_issues=prepared.quality[prepared.quality["date"] < live_origin].reset_index(drop=True),
        accuracy=_accuracy(tables["forecasts"], tables["runs"], features),
        feature_dictionary=pd.DataFrame(
            {
                "feature": [f.name for f in FEATURES],
                "group": [f.group for f in FEATURES],
                "description": [f.description for f in FEATURES],
                "known_in_advance": [f.name in KNOWN_IN_ADVANCE for f in FEATURES],
            }
        ),
        labour_standards=ds.labour_standards.copy(),
    )
    validate(extract)
    return extract


def _version(prefix: str, payload: object) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{prefix}-{digest}"


def _build_run(
    features: pd.DataFrame,
    quality: pd.DataFrame,
    ds: SyntheticDataset,
    origin: pd.Timestamp,
    horizon_days: int,
    params: ModelParams,
    *,
    is_live: bool,
) -> dict[str, pd.DataFrame]:
    run_id = f"run-{origin:%Y-%m-%d}"
    end = origin + pd.Timedelta(days=horizon_days)
    models = train(features, before=origin, params=params)
    window = features[(features["date"] >= origin) & (features["date"] < end)].copy()
    window["forecast"] = predict(models, window)
    window["baseline"] = seasonal_naive(window)
    window["lead_days"] = (window["date"] - origin).dt.days
    window["lead_week"] = window["lead_days"] // 7 + 1

    contributions = explain(models, window)
    base = contributions.drop_duplicates(KEYS)[[*KEYS, "base_value"]]
    forecasts = window.merge(base, on=KEYS, how="left")
    forecasts["method"] = np.where(forecasts["is_trading_day"], "model", "non_trading_rule")
    forecasts = forecasts[
        ["store_id", "department", "driver", "date", "lead_days", "lead_week", "is_trading_day", "method", "forecast", "baseline", "base_value"]
    ].assign(run_id=run_id)

    hours = weekly_hours(window, ds.labour_standards, volume_col="forecast")
    baseline_hours = weekly_hours(window.assign(baseline=window["baseline"].fillna(np.inf)), ds.labour_standards, volume_col="baseline")
    hours["baseline_variable_hours"] = baseline_hours["variable_hours"].replace(np.inf, np.nan).to_numpy()
    hours["lead_week"] = ((hours["week_start"] - origin).dt.days // 7 + 1).astype("int64")
    hour_parts = weekly_hours_contributions(contributions, ds.labour_standards)

    runs = pd.DataFrame(
        {
            "run_id": [run_id],
            "origin": [origin],
            "is_live": [is_live],
            "horizon_days": [horizon_days],
            "data_through": [features.loc[features["target"].notna() & (features["date"] < origin), "date"].max()],
            "trained_until": [max(m.trained_until for m in models.values())],
            "training_rows": [sum(m.n_training_rows for m in models.values())],
            "model_version": [_version("gbm", params.__dict__)],
            "feature_version": [_version("feat", [(f.name, f.group) for f in FEATURES])],
            "standard_version": [ds.labour_standards["standard_version"].iloc[0]],
        }
    )
    return {
        "runs": runs,
        "forecasts": forecasts,
        "contributions": contributions.assign(run_id=run_id),
        "weekly_hours": hours.assign(run_id=run_id),
        "weekly_hour_contributions": hour_parts.assign(run_id=run_id),
        "eligibility": _eligibility(features, quality, ds, origin, end, run_id),
    }


def _eligibility(
    features: pd.DataFrame, quality: pd.DataFrame, ds: SyntheticDataset, origin: pd.Timestamp, end: pd.Timestamp, run_id: str
) -> pd.DataFrame:
    """Per store x driver in the run: what a planner should know before trusting it."""
    in_run = features[(features["date"] >= origin) & (features["date"] < end)][["store_id", "driver"]].drop_duplicates()
    out = in_run.merge(ds.stores[["store_id", "open_date", "close_date"]], on="store_id", how="left")
    out["days_since_open"] = (origin - out["open_date"]).dt.days
    out["is_ramping"] = out["days_since_open"] < OutlierRule().min_days_trading
    out["closes_in_horizon"] = out["close_date"].notna() & (out["close_date"] < end)

    recent = quality[(quality["date"] >= origin - pd.Timedelta(days=28)) & (quality["date"] < origin)].copy()
    recent["column"] = recent["issue_type"].map(
        {
            "missing": "missing_days_28",
            "duplicate_load": "duplicate_rows_removed_28",
            "outlier_spike": "outliers_excluded_28",
            "outlier_dropout": "outliers_excluded_28",
        }
    )
    counts = recent.groupby(["store_id", "driver", "column"]).size().unstack("column", fill_value=0)
    out = out.merge(counts.reset_index(), on=["store_id", "driver"], how="left")
    for column in ("missing_days_28", "duplicate_rows_removed_28", "outliers_excluded_28"):
        out[column] = (out[column] if column in out else pd.Series(0, index=out.index)).fillna(0).astype("int64")

    seen = features[features["reported_volume"].notna() & (features["date"] < origin)]
    latest = seen.groupby(["store_id", "driver"])["date"].max()
    out = out.merge(latest.rename("latest_actual_date").reset_index(), on=["store_id", "driver"], how="left")
    out["data_is_stale"] = out["latest_actual_date"] < origin - pd.Timedelta(days=1)
    return out.drop(columns=["open_date"]).assign(run_id=run_id)


def _accuracy(forecasts: pd.DataFrame, runs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Past runs scored against the reported (clean) actuals, per store x driver x lead week."""
    live_origin = runs.loc[runs["is_live"], "origin"].iloc[0]
    past = runs[~runs["is_live"]]
    scored = forecasts[forecasts["run_id"].isin(past["run_id"])].merge(
        features[KEYS + ["target"]], on=KEYS, how="left"
    )
    scored = scored[scored["is_trading_day"] & scored["target"].notna() & scored["baseline"].notna()]
    scored = scored.assign(
        abs_err_model=(scored["forecast"] - scored["target"]).abs(),
        abs_err_baseline=(scored["baseline"] - scored["target"]).abs(),
    )
    table = scored.groupby(["run_id", "store_id", "driver", "lead_week"], as_index=False).agg(
        days=("target", "size"),
        actual=("target", "sum"),
        abs_err_model=("abs_err_model", "sum"),
        abs_err_baseline=("abs_err_baseline", "sum"),
    )
    window_end = past.set_index("run_id")["origin"] + pd.Timedelta(days=int(past["horizon_days"].iloc[0]) - 1)
    table["window_end"] = table["run_id"].map(window_end)
    table = table[table["window_end"] < live_origin]  # only windows that have fully played out
    return table.rename(columns={"run_id": "scored_run_id"}).reset_index(drop=True)


# --------------------------------------------------------------------- contract


def _schemas() -> dict[str, pa.DataFrameSchema]:
    keys = {"store_id": pa.Column(str), "driver": pa.Column(str), "date": pa.Column("datetime64[ns]", coerce=True)}
    groups = set(FEATURE_GROUPS.values())
    return {
        "runs": pa.DataFrameSchema(
            {
                "run_id": pa.Column(str, unique=True),
                "origin": pa.Column("datetime64[ns]", coerce=True),
                "is_live": pa.Column(bool),
                "horizon_days": pa.Column(int, pa.Check.gt(0)),
                "training_rows": pa.Column(int, pa.Check.gt(0)),
            }
        ),
        "forecasts": pa.DataFrameSchema(
            {
                "run_id": pa.Column(str),
                **keys,
                "lead_days": pa.Column(int, pa.Check.ge(0)),
                "lead_week": pa.Column(int, pa.Check.ge(1)),
                "is_trading_day": pa.Column(bool),
                "method": pa.Column(str, pa.Check.isin(["model", "non_trading_rule"])),
                "forecast": pa.Column(float, pa.Check.ge(-TOLERANCE)),
                "baseline": pa.Column(float, nullable=True),
                "base_value": pa.Column(float, nullable=True),
            },
            unique=["run_id", "store_id", "driver", "date"],
        ),
        "contributions": pa.DataFrameSchema(
            {
                "run_id": pa.Column(str),
                **keys,
                "feature": pa.Column(str, pa.Check.isin(list(FEATURE_GROUPS))),
                "group": pa.Column(str, pa.Check.isin(sorted(groups))),
                "contribution": pa.Column(float),
            },
            unique=["run_id", "store_id", "driver", "date", "feature"],
        ),
        "weekly_hours": pa.DataFrameSchema(
            {
                "run_id": pa.Column(str),
                "store_id": pa.Column(str),
                "department": pa.Column(str),
                "week_start": pa.Column("datetime64[ns]", coerce=True),
                "lead_week": pa.Column(int, pa.Check.ge(1)),
                "fixed_hours": pa.Column(float, pa.Check.ge(0)),
                "variable_hours": pa.Column(float),
                "total_hours": pa.Column(float),
            },
            unique=["run_id", "store_id", "department", "week_start"],
        ),
        "weekly_hour_contributions": pa.DataFrameSchema(
            {
                "run_id": pa.Column(str),
                "group": pa.Column(str, pa.Check.isin(sorted(groups | {"base"}))),
                "hours": pa.Column(float),
            },
            unique=["run_id", "store_id", "department", "week_start", "group"],
        ),
        "eligibility": pa.DataFrameSchema(
            {"run_id": pa.Column(str), "store_id": pa.Column(str), "driver": pa.Column(str), "is_ramping": pa.Column(bool), "data_is_stale": pa.Column(bool)},
            unique=["run_id", "store_id", "driver"],
        ),
        "accuracy": pa.DataFrameSchema(
            {"scored_run_id": pa.Column(str), "days": pa.Column(int, pa.Check.gt(0)), "actual": pa.Column(float, pa.Check.ge(0))},
            unique=["scored_run_id", "store_id", "driver", "lead_week"],
        ),
        "feature_dictionary": pa.DataFrameSchema({"feature": pa.Column(str, unique=True), "group": pa.Column(str), "description": pa.Column(str)}),
    }


def check_invariants(extract: Extract) -> list[str]:
    """Return every invariant violation (empty list = the extract is sound)."""
    problems: list[str] = []
    f, c = extract.forecasts, extract.contributions
    run_keys = ["run_id", *KEYS]

    sums = c.groupby(run_keys)["contribution"].sum().rename("contribution_sum")
    trading = f[f["is_trading_day"]].merge(sums, on=run_keys, how="left")
    gap = (trading["base_value"] + trading["contribution_sum"] - trading["forecast"]).abs()
    if trading["contribution_sum"].isna().any() or (gap > TOLERANCE).any() or gap.isna().any():
        problems.append(f"forecast != base + contributions on {int((gap > TOLERANCE).sum() + gap.isna().sum())} rows")

    closed = f[~f["is_trading_day"]]
    if (closed["forecast"] != 0).any():
        problems.append("non-trading days with a non-zero forecast")
    if not closed.merge(c[run_keys].drop_duplicates(), on=run_keys).empty:
        problems.append("non-trading days have contributions")

    week_keys = ["run_id", "store_id", "department", "week_start"]
    hour_sums = extract.weekly_hour_contributions.groupby(week_keys)["hours"].sum().rename("hours_sum")
    hours = extract.weekly_hours.merge(hour_sums, on=week_keys, how="left")
    if ((hours["hours_sum"].fillna(0) - hours["variable_hours"]).abs() > TOLERANCE).any():
        problems.append("weekly variable hours != sum of hour contributions")
    if ((hours["fixed_hours"] + hours["variable_hours"] - hours["total_hours"]).abs() > TOLERANCE).any():
        problems.append("total hours != fixed + variable")

    live = extract.runs[extract.runs["is_live"]]
    if len(live) != 1:
        problems.append(f"expected exactly one live run, found {len(live)}")
    elif (extract.accuracy["window_end"] >= live["origin"].iloc[0]).any():
        problems.append("accuracy uses windows that end on or after the live origin")
    if set(f["run_id"]) - set(extract.runs["run_id"]):
        problems.append("forecasts reference unknown runs")
    return problems


def validate(extract: Extract) -> None:
    """Raise ``ExtractContractError`` listing every schema and invariant violation."""
    problems = []
    for name, schema in _schemas().items():
        try:
            schema.validate(getattr(extract, name), lazy=True)
        except pa.errors.SchemaErrors as err:
            problems.append(f"{name}: {len(err.failure_cases)} schema failures ({sorted(set(err.failure_cases['check'].astype(str)))[:3]})")
    problems.extend(check_invariants(extract))
    if problems:
        raise ExtractContractError(problems)
