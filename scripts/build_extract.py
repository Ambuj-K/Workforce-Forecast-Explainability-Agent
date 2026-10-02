"""Build the explainability extract (past runs + one live run) and the agent's database.

Usage:
    uv run python scripts/build_extract.py --runs 5 --out data/extract --db data/agent.duckdb
"""

import argparse
from pathlib import Path

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import build_extract
from wfx.explain.store import build_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5, help="past (back-test) runs before the live run")
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument("--out", type=Path, default=Path("data/extract"))
    parser.add_argument("--db", type=Path, default=Path("data/agent.duckdb"))
    args = parser.parse_args()

    extract = build_extract(generate(GeneratorConfig(seed=args.seed)), n_backtest_runs=args.runs)
    for name, path in extract.write(args.out).items():
        print(f"{name:28s} {len(getattr(extract, name)):>8,d} rows  {path}")
    print(f"agent database: {build_database(extract, args.db)}")
    print(extract.runs[["run_id", "origin", "is_live", "data_through"]].to_string(index=False))


if __name__ == "__main__":
    main()
