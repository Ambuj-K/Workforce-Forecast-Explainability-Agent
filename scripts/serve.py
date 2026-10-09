"""Serve the explainability agent API + UI.

Usage:
    uv run python scripts/serve.py                                   # local: http://127.0.0.1:8000, no sign-in
    WFX_AUTH=token WFX_USERS_FILE=users.yaml uv run python scripts/serve.py --host 0.0.0.0

Configuration (environment, so it is the same on a laptop and in a container):
    WFX_AUTH              none | token | header   (default none; none is refused off loopback)
    WFX_USERS_FILE        users YAML (ids, roles, stores, token hashes); see wfx/api/auth.py
    WFX_AUTH_HEADER       header set by an identity proxy, for WFX_AUTH=header
    WFX_RATE_PER_MINUTE   questions per user per minute (default 6), WFX_RATE_BURST (default 3)
    GOOGLE_API_KEY_FILE   mounted secret holding the LLM key (or GOOGLE_API_KEY in .env locally)
    PORT                  port to listen on (default 8000; most container platforms set it)
"""

import argparse
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from wfx.agent.llm import GeminiLLM
from wfx.api.app import create_app
from wfx.api.auth import Authenticator, RateLimiter, check_bind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("WFX_DB", "data/agent.duckdb")))
    parser.add_argument("--log", type=Path, default=Path(os.environ.get("WFX_LOG", "data/interactions.jsonl")))
    parser.add_argument("--host", default=os.environ.get("WFX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    load_dotenv()
    mode = os.environ.get("WFX_AUTH", "none")
    try:
        check_bind(mode, args.host)
    except ValueError as err:
        raise SystemExit(str(err)) from None
    users = os.environ.get("WFX_USERS_FILE")
    auth = Authenticator.from_file(mode, Path(users) if users else None, os.environ.get("WFX_AUTH_HEADER", "X-Authenticated-User"))
    limiter = RateLimiter(float(os.environ.get("WFX_RATE_PER_MINUTE", "6")), int(os.environ.get("WFX_RATE_BURST", "3")))
    app = create_app(args.db, GeminiLLM(), args.log, auth=auth, limiter=limiter, eval_dir=Path("reports/evals"))
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=True, server_header=False)


if __name__ == "__main__":
    main()
