"""Ask the explainability agent a question (uses Gemini; needs GOOGLE_API_KEY in .env).

Usage:
    uv run python scripts/ask.py "Why does store 1 grocery need these hours next week?"
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from wfx.agent.graph import ExplainabilityAgent
from wfx.agent.llm import GeminiLLM
from wfx.explain.store import connect_readonly
from wfx.explain.tools import EvidenceTools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--db", type=Path, default=Path("data/agent.duckdb"))
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    agent = ExplainabilityAgent(EvidenceTools(connect_readonly(args.db)), GeminiLLM())
    result = agent.ask(args.question)
    print(result.answer)
    plan = result.plan.question_type if result.plan else "-"
    print(f"\n[{result.outcome} · {plan} · explain attempts: {result.attempts}]")
    if args.show_evidence:
        for item in result.evidence:
            print(f"- {item['tool']}({item['args']}) → {item['status']}")


if __name__ == "__main__":
    main()
