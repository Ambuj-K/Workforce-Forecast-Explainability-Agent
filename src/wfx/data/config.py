"""Configuration for the synthetic retail workload generator."""

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class DriverSpec:
    """One workload driver: how its daily volume behaves and how it converts to hours."""

    department: str
    driver: str
    base_volume: float  # daily volume for a store with size_factor 1.0
    weekly_profile: tuple[float, float, float, float, float, float, float]  # Mon..Sun
    annual_amplitude: float  # relative swing of the yearly cycle
    annual_peak_day: int  # day of year where the yearly cycle peaks
    annual_growth: float  # compound growth per year
    promo_uplift: float  # multiplier on promotion days
    price_elasticity: float  # volume multiplier = price_index ** -elasticity
    noise_sigma: float  # lognormal noise on daily volume
    minutes_per_unit: float  # variable labour standard
    fixed_hours_per_week: float  # labour that does not scale with volume
    holiday_lead_days: int = 0  # >0 when the driver moves ahead of demand (e.g. deliveries)


DEFAULT_DRIVERS: tuple[DriverSpec, ...] = (
    DriverSpec(
        department="checkouts",
        driver="transactions",
        base_volume=2500,
        weekly_profile=(0.85, 0.85, 0.90, 0.95, 1.10, 1.35, 1.00),
        annual_amplitude=0.08,
        annual_peak_day=354,
        annual_growth=0.02,
        promo_uplift=1.05,
        price_elasticity=0.3,
        noise_sigma=0.06,
        minutes_per_unit=0.9,
        fixed_hours_per_week=20,
    ),
    DriverSpec(
        department="grocery",
        driver="cases_filled",
        base_volume=1200,
        weekly_profile=(1.10, 1.00, 1.00, 1.00, 1.05, 0.95, 0.90),
        annual_amplitude=0.06,
        annual_peak_day=350,
        annual_growth=0.01,
        promo_uplift=1.25,
        price_elasticity=0.8,
        noise_sigma=0.08,
        minutes_per_unit=1.4,
        fixed_hours_per_week=15,
    ),
    DriverSpec(
        department="fresh",
        driver="units_prepared",
        base_volume=800,
        weekly_profile=(0.90, 0.90, 0.95, 1.00, 1.15, 1.30, 0.80),
        annual_amplitude=0.10,
        annual_peak_day=196,
        annual_growth=0.02,
        promo_uplift=1.30,
        price_elasticity=1.0,
        noise_sigma=0.10,
        minutes_per_unit=2.0,
        fixed_hours_per_week=25,
    ),
    DriverSpec(
        department="receiving",
        driver="pallets_received",
        base_volume=60,
        weekly_profile=(1.30, 1.10, 1.00, 1.10, 1.20, 0.80, 0.50),
        annual_amplitude=0.05,
        annual_peak_day=350,
        annual_growth=0.02,
        promo_uplift=1.10,
        price_elasticity=0.2,
        noise_sigma=0.12,
        minutes_per_unit=12.0,
        fixed_hours_per_week=10,
        holiday_lead_days=2,
    ),
    DriverSpec(
        department="online",
        driver="orders_picked",
        base_volume=150,
        weekly_profile=(1.00, 1.00, 1.00, 1.05, 1.10, 1.10, 0.75),
        annual_amplitude=0.12,
        annual_peak_day=350,
        annual_growth=0.15,
        promo_uplift=1.10,
        price_elasticity=0.4,
        noise_sigma=0.10,
        minutes_per_unit=9.0,
        fixed_hours_per_week=8,
    ),
)


@dataclass(frozen=True)
class CaveatConfig:
    """How many data-quality problems to inject into the reported volumes."""

    n_gaps: int = 4  # runs of missing days for one store-driver
    gap_days: tuple[int, int] = (3, 14)
    n_duplicate_days: int = 3  # a store-day loaded twice (all drivers)
    n_outliers: int = 6  # single store-driver-day spikes or dropouts
    outlier_spike_factor: float = 6.0
    late_feed_days: int = 5  # one store's most recent days not yet delivered


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level generator settings. Same config + seed always gives the same data."""

    start: date = date(2023, 1, 2)  # a Monday
    end: date = date(2025, 12, 28)  # a Sunday
    n_stores: int = 12
    n_new_stores: int = 2  # open during the range and ramp up
    n_closed_stores: int = 1  # close before the end of the range
    ramp_up_days: int = 70  # days for a new store to reach steady trading
    promo_weeks_per_year: int = 8  # per department, chain-wide
    price_changes_per_year: int = 2  # per department
    drivers: tuple[DriverSpec, ...] = DEFAULT_DRIVERS
    caveats: CaveatConfig = field(default_factory=CaveatConfig)
    standard_version: str = "LS-2024.1"
    seed: int = 7
