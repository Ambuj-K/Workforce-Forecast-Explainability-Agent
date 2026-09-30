"""Run the full walk-forward back-test on the synthetic dataset and write accuracy tables.

Usage:
    uv run python scripts/run_backtest.py --out reports
"""

import argparse
from pathlib import Path

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.forecasting.backtest import BacktestConfig, accuracy, walk_forward
from wfx.forecasting.features import build_features
from wfx.forecasting.prepare import prepare_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--max-origins", type=int, default=None)
    args = parser.parse_args()

    ds = generate(GeneratorConfig(seed=args.seed))
    features = build_features(prepare_observations(ds).frame, ds)
    forecasts = walk_forward(features, ds, BacktestConfig(max_origins=args.max_origins))

    args.out.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(args.out / "backtest_forecasts.parquet", index=False)
    for by in (("driver",), ("lead_week",), ("driver", "lead_week")):
        table = accuracy(forecasts, by=by)
        name = "accuracy_by_" + "_".join(by)
        table.to_csv(args.out / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
