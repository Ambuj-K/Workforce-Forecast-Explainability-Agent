"""One gradient-boosted model per workload driver, with exact per-row contributions.

Forecasts are in volume units, so every contribution reads directly as
"this feature moved the forecast by N units". For every forecast row:

    base_value + sum(contributions) == forecast   (exact, up to float error)

Non-trading days are not modelled: they are forecast as 0 by rule and explained as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from wfx.forecasting.features import FEATURE_GROUPS, FEATURE_NAMES

# Categorical features (day of week, holiday, region) are deliberately passed as plain
# integers. With native categorical splits, the SHAP tree explainer silently returns
# contributions that do not add up to the forecast (measured error: 1,641 units).
# Numeric encoding keeps base + contributions == forecast exact.
KEYS = ["store_id", "driver", "date"]


@dataclass(frozen=True)
class ModelParams:
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    random_state: int = 0


@dataclass
class DriverModel:
    """A fitted model for one driver and what it was trained on."""

    driver: str
    estimator: HistGradientBoostingRegressor
    trained_until: pd.Timestamp  # last target date used for training
    n_training_rows: int
    _explainer: shap.TreeExplainer | None = None

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.estimator)
        return self._explainer


def training_rows(features: pd.DataFrame, driver: str, before: pd.Timestamp) -> pd.DataFrame:
    """Rows usable for training: this driver, known clean target, trading day, date < ``before``."""
    mask = (
        (features["driver"] == driver)
        & (features["date"] < before)
        & features["target"].notna()
        & features["is_trading_day"]
    )
    return features[mask]


def train(features: pd.DataFrame, before: pd.Timestamp, params: ModelParams | None = None) -> dict[str, DriverModel]:
    """Fit one model per driver on everything known before ``before``."""
    params = params or ModelParams()
    models = {}
    for driver in sorted(features["driver"].unique()):
        rows = training_rows(features, driver, before)
        if rows.empty:
            raise ValueError(f"no training rows for {driver} before {before.date()}")
        empty = [name for name in FEATURE_NAMES if rows[name].isna().all()]
        if empty:
            raise ValueError(f"features with no history before {before.date()} for {driver}: {empty}; start later")
        estimator = HistGradientBoostingRegressor(
            early_stopping=False,
            **params.__dict__,
        )
        estimator.fit(rows[list(FEATURE_NAMES)], rows["target"])
        models[driver] = DriverModel(
            driver=driver,
            estimator=estimator,
            trained_until=rows["date"].max(),
            n_training_rows=len(rows),
        )
    return models


def predict(models: dict[str, DriverModel], rows: pd.DataFrame) -> pd.Series:
    """Forecast for each row: model output on trading days, 0 on non-trading days."""
    forecast = pd.Series(0.0, index=rows.index)
    for driver, model in models.items():
        mask = (rows["driver"] == driver) & rows["is_trading_day"]
        if mask.any():
            forecast[mask] = model.estimator.predict(rows.loc[mask, list(FEATURE_NAMES)])
    return forecast


def explain(models: dict[str, DriverModel], rows: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature contributions (long format) for trading-day rows.

    Columns: store_id, driver, date, feature, group, contribution, base_value.
    """
    parts = []
    for driver, model in models.items():
        subset = rows[(rows["driver"] == driver) & rows["is_trading_day"]]
        if subset.empty:
            continue
        values = model.explainer.shap_values(subset[list(FEATURE_NAMES)])
        wide = pd.DataFrame(values, columns=list(FEATURE_NAMES), index=subset.index)
        wide[KEYS] = subset[KEYS]
        long = wide.melt(id_vars=KEYS, var_name="feature", value_name="contribution")
        long["group"] = long["feature"].map(FEATURE_GROUPS)
        long["base_value"] = float(np.ravel(model.explainer.expected_value)[0])
        parts.append(long)
    columns = [*KEYS, "feature", "group", "contribution", "base_value"]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)[columns]
