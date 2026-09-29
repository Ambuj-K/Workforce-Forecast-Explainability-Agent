"""Synthetic retail workload generator with injected, ground-truthed data-quality caveats.

Produces daily workload volumes per store and driver, the calendar, promotions and
prices that shaped them, labour standards to convert volume into hours, and a
caveats table recording every data-quality problem injected into the reported
volumes. The clean pre-injection volumes are kept as ``true_volumes`` so that
forecast accuracy and caveat disclosure can both be evaluated against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wfx.data.config import GeneratorConfig

REGIONS = ("north", "south", "east", "west")
CAVEAT_COLUMNS = (
    "caveat_id",
    "caveat_type",
    "store_id",
    "department",
    "driver",
    "start_date",
    "end_date",
    "affected_rows",
    "detail",
)


@dataclass(frozen=True)
class SyntheticDataset:
    """All tables produced by one generator run."""

    calendar: pd.DataFrame
    stores: pd.DataFrame
    promotions: pd.DataFrame
    prices: pd.DataFrame
    true_volumes: pd.DataFrame
    volumes: pd.DataFrame  # reported: true volumes with caveats injected
    labour_standards: pd.DataFrame
    caveats: pd.DataFrame

    def write(self, out_dir: Path) -> dict[str, Path]:
        """Write every table as parquet under ``out_dir``; return the paths by table name."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for table in fields(self):
            path = out_dir / f"{table.name}.parquet"
            getattr(self, table.name).to_parquet(path, index=False)
            paths[table.name] = path
        return paths


def generate(config: GeneratorConfig | None = None) -> SyntheticDataset:
    """Generate a complete synthetic dataset. Deterministic for a given config and seed."""
    config = config or GeneratorConfig()
    _validate(config)
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start, config.end, freq="D")
    departments = _departments(config)

    holiday_mult, holiday_names = _holiday_multipliers(dates)
    calendar = _build_calendar(dates, holiday_mult, holiday_names)
    stores = _build_stores(config, dates, rng)
    promo_on = _build_promotion_grid(config, dates, departments, rng)
    price_index = _build_price_grid(config, dates, departments, rng)
    true_volumes = _simulate_volumes(
        config, dates, stores, departments, holiday_mult, promo_on, price_index, rng
    )
    volumes, caveats = _inject_caveats(config, true_volumes, stores, rng)

    return SyntheticDataset(
        calendar=calendar,
        stores=stores,
        promotions=_to_long(dates, departments, promo_on, "on_promotion"),
        prices=_to_long(dates, departments, price_index.round(4), "price_index"),
        true_volumes=true_volumes,
        volumes=volumes,
        labour_standards=_build_labour_standards(config),
        caveats=caveats,
    )


# --------------------------------------------------------------------------- setup


def _validate(config: GeneratorConfig) -> None:
    if config.end <= config.start:
        raise ValueError("end must be after start")
    if config.n_new_stores + config.n_closed_stores >= config.n_stores:
        raise ValueError("need at least one store that is neither new nor closed")
    drivers = [spec.driver for spec in config.drivers]
    if len(drivers) != len(set(drivers)):
        raise ValueError("driver names must be unique")


def _departments(config: GeneratorConfig) -> list[str]:
    return list(dict.fromkeys(spec.department for spec in config.drivers))


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _nth_monday(year: int, month: int, *, last: bool) -> date:
    days = [date(year, month, d) for d in range(1, 32) if _valid(year, month, d)]
    mondays = [d for d in days if d.weekday() == 0]
    return mondays[-1] if last else mondays[0]


def _valid(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def _holiday_multipliers(dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[str | None]]:
    """Demand multiplier and holiday name per date. A multiplier of 0 means non-trading."""
    mult = np.ones(len(dates))
    names: list[str | None] = [None] * len(dates)
    position = {d.date(): i for i, d in enumerate(dates)}

    def mark(day: date, multiplier: float, name: str) -> None:
        i = position.get(day)
        if i is not None:
            mult[i] = multiplier
            names[i] = name

    for year in sorted({d.year for d in dates}):
        easter = _easter_sunday(year)
        mark(date(year, 1, 1), 0.6, "new_year")
        mark(easter - timedelta(days=2), 1.1, "good_friday")
        mark(easter - timedelta(days=1), 1.25, "easter_saturday")
        mark(easter, 0.0, "easter_sunday")
        mark(easter + timedelta(days=1), 1.1, "easter_monday")
        mark(_nth_monday(year, 5, last=False), 1.1, "early_may_bank_holiday")
        mark(_nth_monday(year, 5, last=True), 1.1, "spring_bank_holiday")
        mark(_nth_monday(year, 8, last=True), 1.1, "summer_bank_holiday")
        for offset, multiplier in enumerate((1.15, 1.2, 1.25, 1.3, 1.4, 1.45)):
            mark(date(year, 12, 18 + offset), multiplier, "christmas_build_up")
        mark(date(year, 12, 24), 1.6, "christmas_eve")
        mark(date(year, 12, 25), 0.0, "christmas_day")
        mark(date(year, 12, 26), 0.7, "boxing_day")
        for day in range(27, 31):
            mark(date(year, 12, day), 0.95, "christmas_week")
        mark(date(year, 12, 31), 1.2, "new_years_eve")
    return mult, names


def _build_calendar(
    dates: pd.DatetimeIndex, holiday_mult: np.ndarray, holiday_names: list[str | None]
) -> pd.DataFrame:
    """Public calendar. Holiday names are exposed; their demand effects are not."""
    iso = dates.isocalendar()
    return pd.DataFrame(
        {
            "date": dates,
            "iso_year": iso["year"].to_numpy(dtype=np.int64),
            "iso_week": iso["week"].to_numpy(dtype=np.int64),
            "week_start": dates - pd.to_timedelta(dates.dayofweek, unit="D"),
            "day_of_week": dates.dayofweek.to_numpy(dtype=np.int64),
            "holiday": pd.array(holiday_names, dtype="string"),
            "is_trading_day": holiday_mult > 0,
        }
    )


def _build_stores(
    config: GeneratorConfig, dates: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    n = config.n_stores
    start, end = dates[0], dates[-1]
    span = (end - start).days

    open_dates = [start - pd.Timedelta(days=int(rng.integers(400, 4000))) for _ in range(n)]
    close_dates: list[pd.Timestamp] = [pd.NaT] * n
    new_ids = range(n - config.n_new_stores, n)
    closed_ids = range(n - config.n_new_stores - config.n_closed_stores, n - config.n_new_stores)
    for i in new_ids:
        open_dates[i] = start + pd.Timedelta(days=int(rng.integers(int(span * 0.25), int(span * 0.6))))
    for i in closed_ids:
        close_dates[i] = start + pd.Timedelta(days=int(rng.integers(int(span * 0.5), int(span * 0.85))))

    return pd.DataFrame(
        {
            "store_id": [f"S{i + 1:03d}" for i in range(n)],
            "region": rng.choice(REGIONS, size=n),
            "size_factor": np.round(rng.lognormal(0.0, 0.3, size=n), 3),
            "open_date": pd.to_datetime(open_dates),
            "close_date": pd.to_datetime(close_dates),
        }
    )


def _build_promotion_grid(
    config: GeneratorConfig,
    dates: pd.DatetimeIndex,
    departments: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Chain-wide, week-long promotions per department. Shape (dates, departments)."""
    on = np.zeros((len(dates), len(departments)), dtype=bool)
    week_starts = np.flatnonzero(dates.dayofweek == 0)
    n_weeks = min(len(week_starts), round(config.promo_weeks_per_year * len(dates) / 365.25))
    for j in range(len(departments)):
        for start in rng.choice(week_starts, size=n_weeks, replace=False):
            on[start : start + 7, j] = True
    return on


def _build_price_grid(
    config: GeneratorConfig,
    dates: pd.DatetimeIndex,
    departments: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Step-changing relative price index per department (starts at 1.0)."""
    index = np.ones((len(dates), len(departments)))
    n_changes = round(config.price_changes_per_year * len(dates) / 365.25)
    for j in range(len(departments)):
        level = 1.0
        for point in np.sort(rng.choice(np.arange(30, len(dates)), size=n_changes, replace=False)):
            level *= 1 + rng.uniform(-0.06, 0.06)
            index[point:, j] = level
    return index


def _lead_multiplier(mult: np.ndarray, lead_days: int) -> np.ndarray:
    """Holiday uplift seen ``lead_days`` early (e.g. deliveries arrive before demand)."""
    if lead_days == 0:
        return mult.copy()
    out = np.empty_like(mult)
    for i in range(len(mult)):
        window = mult[i : i + lead_days + 1]
        trading = window[window > 0]
        out[i] = trading.max() if trading.size else 1.0
    return out


# --------------------------------------------------------------------- simulation


def _simulate_volumes(
    config: GeneratorConfig,
    dates: pd.DatetimeIndex,
    stores: pd.DataFrame,
    departments: list[str],
    holiday_mult: np.ndarray,
    promo_on: np.ndarray,
    price_index: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Clean daily volume per store and driver while the store is trading."""
    specs = config.drivers
    n_dates, n_drivers = len(dates), len(specs)
    years = np.arange(n_dates) / 365.25
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    trading = holiday_mult > 0
    dept_col = {dept: j for j, dept in enumerate(departments)}

    chain = np.empty((n_dates, n_drivers))
    for k, spec in enumerate(specs):
        j = dept_col[spec.department]
        chain[:, k] = (
            spec.base_volume
            * (1 + spec.annual_growth) ** years
            * np.asarray(spec.weekly_profile)[dow]
            * (1 + spec.annual_amplitude * np.cos(2 * np.pi * (doy - spec.annual_peak_day) / 365.25))
            * np.where(trading, _lead_multiplier(holiday_mult, spec.holiday_lead_days), 0.0)
            * np.where(promo_on[:, j], spec.promo_uplift, 1.0)
            * price_index[:, j] ** -spec.price_elasticity
        )

    days_open = (dates.to_numpy()[:, None] - stores["open_date"].to_numpy()[None, :]) / np.timedelta64(1, "D")
    closed = dates.to_numpy()[:, None] >= stores["close_date"].to_numpy()[None, :]  # NaT compares False
    active = (days_open >= 0) & ~closed
    midpoint, steepness = config.ramp_up_days / 2, config.ramp_up_days / 8
    ramp = 1 / (1 + np.exp(-(days_open - midpoint) / steepness))

    sigma = np.array([spec.noise_sigma for spec in specs])
    noise = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=(n_dates, len(stores), n_drivers))
    size = stores["size_factor"].to_numpy()
    volume = np.rint(chain[:, None, :] * size[None, :, None] * ramp[:, :, None] * noise).astype(np.int64)

    d_idx, s_idx = np.nonzero(active)
    rows_d = np.repeat(d_idx, n_drivers)
    rows_s = np.repeat(s_idx, n_drivers)
    rows_k = np.tile(np.arange(n_drivers), len(d_idx))
    frame = pd.DataFrame(
        {
            "date": dates[rows_d],
            "store_id": stores["store_id"].to_numpy()[rows_s],
            "department": np.array([spec.department for spec in specs])[rows_k],
            "driver": np.array([spec.driver for spec in specs])[rows_k],
            "volume": volume[rows_d, rows_s, rows_k],
        }
    )
    return frame.sort_values(["date", "store_id", "driver"], ignore_index=True)


# ------------------------------------------------------------------------ caveats


def _inject_caveats(
    config: GeneratorConfig,
    true_volumes: pd.DataFrame,
    stores: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrupt a copy of the true volumes and record every change in a caveats table.

    Injected problems never overlap each other on the same store-day, so each one can
    be checked independently against ground truth.
    """
    cfg = config.caveats
    reported = true_volumes.copy()
    records: list[dict[str, object]] = []
    touched: set[tuple[str, pd.Timestamp]] = set()
    first_day, last_day = true_volumes["date"].min(), true_volumes["date"].max()

    def record(**values: object) -> None:
        records.append({column: values.get(column) for column in CAVEAT_COLUMNS})

    # Structural caveats: facts about the data, not corruption.
    for store in stores.itertuples():
        if store.open_date >= first_day:
            record(
                caveat_type="ramp_up",
                store_id=store.store_id,
                start_date=store.open_date,
                end_date=store.open_date + pd.Timedelta(days=config.ramp_up_days - 1),
                affected_rows=int(
                    (
                        (true_volumes["store_id"] == store.store_id)
                        & (true_volumes["date"] < store.open_date + pd.Timedelta(days=config.ramp_up_days))
                    ).sum()
                ),
                detail="new store: volumes below steady state while trading ramps up",
            )
        if pd.notna(store.close_date):
            record(
                caveat_type="store_closed",
                store_id=store.store_id,
                start_date=store.close_date,
                affected_rows=0,
                detail="store closed: no volumes from this date",
            )

    # Late feed: one store trading on the last day is missing its most recent days.
    trading_at_end = sorted(reported.loc[reported["date"] == last_day, "store_id"].unique())
    late_store = str(rng.choice(trading_at_end))
    cutoff = last_day - pd.Timedelta(days=cfg.late_feed_days - 1)
    late_mask = (reported["store_id"] == late_store) & (reported["date"] >= cutoff)
    record(
        caveat_type="late_feed",
        store_id=late_store,
        start_date=cutoff,
        end_date=last_day,
        affected_rows=int(late_mask.sum()),
        detail=f"{cfg.late_feed_days} most recent days not yet delivered",
    )
    touched.update((late_store, d) for d in pd.date_range(cutoff, last_day))
    reported = reported[~late_mask]

    # Candidate rows for random corruption: away from the edges and the late window.
    safe_until = cutoff - pd.Timedelta(days=cfg.gap_days[1] + 1)

    def pick_row(extra: pd.Series | None = None) -> pd.Series:
        mask = (reported["date"] > first_day) & (reported["date"] < safe_until)
        if extra is not None:
            mask &= extra
        pool = reported[mask]
        for _ in range(1000):
            row = pool.iloc[int(rng.integers(len(pool)))]
            if (row["store_id"], row["date"]) not in touched:
                return row
        raise RuntimeError("could not find an untouched row to corrupt")

    for _ in range(cfg.n_gaps):
        for _attempt in range(1000):
            row = pick_row()
            length = int(rng.integers(cfg.gap_days[0], cfg.gap_days[1] + 1))
            window = pd.date_range(row["date"], periods=length)
            if not any((row["store_id"], d) in touched for d in window):
                break
        else:
            raise RuntimeError("could not place a gap without overlapping another caveat")
        gap_mask = (
            (reported["store_id"] == row["store_id"])
            & (reported["driver"] == row["driver"])
            & reported["date"].isin(window)
        )
        record(
            caveat_type="gap",
            store_id=row["store_id"],
            department=row["department"],
            driver=row["driver"],
            start_date=window[0],
            end_date=window[-1],
            affected_rows=int(gap_mask.sum()),
            detail=f"{length} consecutive days missing",
        )
        touched.update((row["store_id"], d) for d in window)
        reported = reported[~gap_mask]

    for n in range(cfg.n_outliers):
        row = pick_row(extra=reported["volume"] > 0)
        spike = n % 2 == 0
        new_value = int(round(row["volume"] * cfg.outlier_spike_factor)) if spike else 0
        reported.loc[row.name, "volume"] = new_value
        record(
            caveat_type="outlier",
            store_id=row["store_id"],
            department=row["department"],
            driver=row["driver"],
            start_date=row["date"],
            end_date=row["date"],
            affected_rows=1,
            detail=(
                f"{'spike' if spike else 'dropout'}: reported {new_value}, true {int(row['volume'])}"
            ),
        )
        touched.add((row["store_id"], row["date"]))

    duplicates = []
    for _ in range(cfg.n_duplicate_days):
        row = pick_row()
        store_day = reported[(reported["store_id"] == row["store_id"]) & (reported["date"] == row["date"])]
        duplicates.append(store_day)
        record(
            caveat_type="duplicate_load",
            store_id=row["store_id"],
            start_date=row["date"],
            end_date=row["date"],
            affected_rows=len(store_day),
            detail="store-day loaded twice (every driver duplicated)",
        )
        touched.add((row["store_id"], row["date"]))

    reported = pd.concat([reported, *duplicates]).sort_values(
        ["date", "store_id", "driver"], ignore_index=True, kind="stable"
    )
    caveats = pd.DataFrame(records, columns=list(CAVEAT_COLUMNS))
    caveats["caveat_id"] = [f"C{i + 1:03d}" for i in range(len(caveats))]
    caveats["affected_rows"] = caveats["affected_rows"].astype("int64")
    return reported, caveats


# ---------------------------------------------------------------------- outputs


def _to_long(dates: pd.DatetimeIndex, departments: list[str], grid: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": np.repeat(dates, len(departments)),
            "department": np.tile(departments, len(dates)),
            name: grid.reshape(-1),
        }
    )


def _build_labour_standards(config: GeneratorConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": [spec.department for spec in config.drivers],
            "driver": [spec.driver for spec in config.drivers],
            "minutes_per_unit": [spec.minutes_per_unit for spec in config.drivers],
            "fixed_hours_per_week": [spec.fixed_hours_per_week for spec in config.drivers],
            "standard_version": config.standard_version,
        }
    )
