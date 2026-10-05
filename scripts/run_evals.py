"""Run the golden questions against the live agent (Gemini), score them, compare with the baseline.

Usage:
    uv run python scripts/run_evals.py                    # run + compare with evals/baseline.json
    uv run python scripts/run_evals.py --save-baseline    # also overwrite the baseline
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from wfx.agent.graph import ExplainabilityAgent
from wfx.agent.llm import GeminiLLM
from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.evals.goldens import load_goldens
from wfx.evals.runner import compare, load, run_goldens, save, summarize
from wfx.explain.store import connect_readonly
from wfx.explain.tools import EvidenceTools

BASELINE = Path("evals/baseline.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/agent.duckdb"))
    parser.add_argument("--only", nargs="*", help="golden ids to run")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between questions (free-tier rate limits)")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    goldens = load_goldens(generate(GeneratorConfig()))
    if args.only:
        goldens = [g for g in goldens if g.id in args.only]
    llm = GeminiLLM()
    agent = ExplainabilityAgent(EvidenceTools(connect_readonly(args.db)), llm)
    results = run_goldens(agent, None if args.no_judge else llm, goldens, pause_seconds=args.pause)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save(results, Path(f"reports/evals/results-{stamp}.json"))
    for r in results:
        failed = [k for k, v in r.checks.items() if v is False]
        judge = f" judge={r.judge_score:.2f}" if r.judge_score is not None else ""
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.id:30s} {r.outcome:13s}{judge}  {'failed: ' + ', '.join(failed) if failed else ''}")
    print(json.dumps(summarize(results), indent=2))

    if BASELINE.exists():
        regressions = compare(results, load(BASELINE))
        print("regressions vs baseline:", regressions or "none")
    if args.save_baseline:
        save(results, BASELINE)
        print(f"baseline saved to {BASELINE}")


if __name__ == "__main__":
    main()
