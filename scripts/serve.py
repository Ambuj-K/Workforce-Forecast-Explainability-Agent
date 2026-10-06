"""Serve the explainability agent API + UI (uses Gemini; needs GOOGLE_API_KEY in .env).

Usage:
    uv run python scripts/serve.py            # http://127.0.0.1:8000
"""

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from wfx.agent.llm import GeminiLLM
from wfx.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/agent.duckdb"))
    parser.add_argument("--log", type=Path, default=Path("data/interactions.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_dotenv()
    app = create_app(args.db, GeminiLLM(), args.log, eval_dir=Path("reports/evals"))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
