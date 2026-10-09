# Workforce Forecast Explainability Agent

An agent that explains **retail labour-demand forecasts** to workforce planners: why a store's forecast looks the way it does, how far to trust it, and what changed since the last run. It **only states what the data supports**.

> **Status:** works end to end on synthetic data: forecasting, explainability extract, agent, evals, monitoring, API with sign-in and per-user store scoping, container image. Next: deployment on a managed AI platform. See [`docs/design.md`](docs/design.md), [`docs/decisions.md`](docs/decisions.md), [`docs/security.md`](docs/security.md).

## Why this exists
Forecasts drive staffing, but planners only see the numbers. The reasons (feature contributions, data filters, labour-standard conversion, historical accuracy) are buried in pipeline outputs. This project makes them answerable in plain language, with every number traceable to its source.

## What makes it different
- **Numbers are verified, not generated.** A deterministic faithfulness gate checks every figure in an answer against tool outputs and pipeline invariants before it's shown.
- **Explanations are exact.** Forecast contributions come from SHAP, and `base + Σ contributions = forecast` is checked as an invariant.
- **Honest limits.** Attribution isn't causation; accuracy is always stated with lead time; known data caveats are disclosed; staffing decisions stay with humans.
- **Pipeline ↔ agent contract.** A schema-validated extract is the only interface between forecasting and the agent.

## Architecture
```
synthetic retail data → walk-forward forecast (GBM + SHAP) → explainability extract (validated)
  → DuckDB (read-only) → LangGraph agent: classify → route → evidence tools
  → capability registry → structured explanation → faithfulness gate → answer
```

## Roadmap
- [x] Synthetic data generator with injected caveats
- [x] Forecast pipeline: data preparation, features, gradient boosting per driver with exact contributions, walk-forward back-test vs seasonal-naive baseline (34–46% lower error)
- [x] Hours conversion with exact hour contributions (weekly hours 38–61% more accurate than seasonal naive)
- [x] Explainability extract (past + live runs, as-of, contract enforced at write and load) + locked-down agent database
- [x] Evidence tools (11, read-only, parameterised, uniform result with guards)
- [x] Agent: LLM plans and writes; code routes, guards, declines and verifies every number (faithfulness gate with retry + safe fallback)
- [x] Evals: 25 goldens (incl. staffing, false premise, injection), deterministic checks + judge, baseline regression: 25/25, 100% first-pass gate, judge 0.97–1.00 across runs
- [x] Monitoring over time (accuracy/data-quality trends, drift relative to baseline, agent health, eval trend) + API + minimal UI
- [x] Security doc: threat model, 17 controls each tied to a test, mapped to OWASP LLM Top 10, NIST AI RMF, ISO/IEC 42001 and the EU AI Act; controls as code ([docs/security.md](docs/security.md))
- [x] Deployment-ready, platform-agnostic: sign-in (tokens or identity proxy), admin-only monitoring, per-user store scoping enforced in the tools, per-user rate limits, mounted secrets, LLM timeout + clean failures, concurrent requests, non-root container image
- [ ] Deployment on a managed AI platform (identity proxy/SSO, secret manager, gateway limits, log service, paid LLM endpoint)
- [ ] Graph-vs-flat retrieval experiment over the store hierarchy

## Quick start
```bash
uv sync
uv run python scripts/generate_data.py --out data/raw   # synthetic dataset (parquet)
uv run python scripts/run_backtest.py --out reports      # walk-forward back-test (~7 min)
uv run python scripts/build_extract.py                   # explainability extract + agent database (~2 min)
uv run pytest -q
cp .env.example .env   # add GOOGLE_API_KEY
uv run python scripts/ask.py "Why does store 1 grocery need these hours next week?"
uv run python scripts/run_evals.py      # golden questions vs the saved baseline
uv run python scripts/serve.py          # API + UI at http://127.0.0.1:8000 (local: no sign-in)
```

## Running with sign-in, or in a container
```bash
uv run python scripts/make_token.py     # prints a token (give to the user) and its hash
cp deploy/users.example.yaml users.yaml # put hashes, roles and stores in it (users.yaml is gitignored)
WFX_AUTH=token WFX_USERS_FILE=users.yaml uv run python scripts/serve.py

docker build -t wfx .                   # builds the synthetic agent database inside the image
docker run -p 8000:8000 --read-only --tmpfs /tmp \
  -e WFX_USERS_FILE=/run/config/users.yaml -v $PWD/users.yaml:/run/config/users.yaml:ro \
  -e GOOGLE_API_KEY_FILE=/run/secrets/google_api_key -v $PWD/google_api_key:/run/secrets/google_api_key:ro wfx
```
Behind a platform identity proxy, use `WFX_AUTH=header` with `WFX_AUTH_HEADER` set to the header the proxy writes. See `scripts/serve.py` for every setting.

## Licence
MIT (see `LICENSE`)
