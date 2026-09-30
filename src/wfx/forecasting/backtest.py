"""Walk-forward back-test: retrain at each origin, forecast the next horizon, score by lead time.

Each origin sees only data dated before it, exactly as a live run would. Every
forecast is paired with a seasonal-naive baseline (the same weekday one horizon
earlier), so accuracy is always reported against a simple alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wfx.data.synthetic import SyntheticDataset
from wfx.forecasting.features import DEFAULT_HORIZON_DAYS
from wfx.forecasting.model import KEYS, ModelParams, predict, train


@dataclass(frozen=True)
class BacktestConfig:
    horizon_days: int = DEFAULT_HORIZON_DAYS  # must match the features' horizon
    step_days: int = 28  # days between forecast origins
    warmup_days: int = 455  # a year (for last-year features) + 13 weeks of them, before the first origin
    max_origins: int | None = None  # cap for quick runs; latest origins are kept
    params: ModelParams = field(default_factory=ModelParams)


def forecast_origins(dates: pd.Series, config: BacktestConfig) -> list[pd.Timestamp]:
    """Mondays from the end of the warm-up, every ``step_days``, with a full horizon of data after."""
    first, last = dates.min(), dates.max()
    start = first + pd.Timedelta(days=config.warmup_days)
    start += pd.Timedelta(days=(7 - start.dayofweek) % 7)
    latest = last - pd.Timedelta(days=config.horizon_days - 1)
    origins = list(pd.date_range(start, latest, freq=f"{config.step_days}D"))
    return origins[-config.max_origins :] if config.max_origins else origins


def walk_forward(
    features: pd.DataFrame, ds: SyntheticDataset, config: BacktestConfig | None = None
) -> pd.DataFrame:
    """Forecast every horizon window in the history.

    Returns one row per origin x store x driver x day with ``forecast``, ``baseline``,
    ``reported`` (clean reported volume, may be NaN) and ``true_volume`` (ground truth).
    """
    config = config or BacktestConfig()
    truth = ds.true_volumes[KEYS + ["volume"]].rename(columns={"volume": "true_volume"})
    results = []
    for origin in forecast_origins(features["date"], config):
        window = features[
            (features["date"] >= origin) & (features["date"] < origin + pd.Timedelta(days=config.horizon_days))
        ].copy()
        models = train(features, before=origin, params=config.params)
        window["origin"] = origin
        window["lead_days"] = (window["date"] - origin).dt.days
        window["forecast"] = predict(models, window)
        window["baseline"] = seasonal_naive(window)
        results.append(
            window[["origin", *KEYS, "department", "lead_days", "is_trading_day", "forecast", "baseline", "target"]]
        )
    out = pd.concat(results, ignore_index=True).rename(columns={"target": "reported"})
    return out.merge(truth, on=KEYS, how="left")


def seasonal_naive(rows: pd.DataFrame) -> pd.Series:
    """Same weekday one horizon earlier; the 4-week same-weekday mean if that day is missing; 0 when not trading."""
    baseline = rows["lag_h"].fillna(rows["same_weekday_mean_4w"])
    return baseline.where(rows["is_trading_day"], 0.0)


def accuracy(forecasts: pd.DataFrame, by: tuple[str, ...] = ("driver", "lead_week"), against: str = "true_volume") -> pd.DataFrame:
    """WAPE of model and baseline, grouped; ``skill`` = 1 - model WAPE / baseline WAPE (higher is better)."""
    scored = forecasts[forecasts["is_trading_day"] & forecasts[against].notna() & forecasts["baseline"].notna()].copy()
    scored["lead_week"] = scored["lead_days"] // 7 + 1
    scored["abs_err_model"] = (scored["forecast"] - scored[against]).abs()
    scored["abs_err_baseline"] = (scored["baseline"] - scored[against]).abs()
    grouped = scored.groupby(list(by), as_index=False).agg(
        n=(against, "size"),
        actual=(against, "sum"),
        abs_err_model=("abs_err_model", "sum"),
        abs_err_baseline=("abs_err_baseline", "sum"),
    )
    grouped["wape_model"] = grouped["abs_err_model"] / grouped["actual"]
    grouped["wape_baseline"] = grouped["abs_err_baseline"] / grouped["actual"]
    grouped["skill"] = 1 - grouped["wape_model"] / grouped["wape_baseline"]
    return grouped[[*by, "n", "wape_model", "wape_baseline", "skill"]].replace([np.inf, -np.inf], np.nan)
