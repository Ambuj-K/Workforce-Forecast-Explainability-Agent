"""Generate the synthetic retail workload dataset as parquet files.

Usage:
    uv run python scripts/generate_data.py --out data/raw --seed 7 --stores 12
"""

import argparse
from pathlib import Path

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--stores", type=int, default=GeneratorConfig.n_stores)
    args = parser.parse_args()

    dataset = generate(GeneratorConfig(seed=args.seed, n_stores=args.stores))
    for name, path in dataset.write(args.out).items():
        print(f"{name:18s} {len(getattr(dataset, name)):>7,d} rows  {path}")


if __name__ == "__main__":
    main()
