# Workforce Forecast Explainability Agent

An agent that explains **retail labour-demand forecasts** to workforce planners: why a store's forecast looks the way it does, how far to trust it, and what changed since the last run. It **only states what the data supports**.

> **Status:** design phase. See [`docs/design.md`](docs/design.md) and [`docs/decisions.md`](docs/decisions.md).

## Why this exists
Forecasts drive staffing, but planners only see the numbers. The reasons (feature contributions, data filters, labour-standard conversion, historical accuracy) are buried in pipeline outputs. This project makes them answerable in plain language, with every number traceable to its source.

## What makes it different
- **Numbers are verified, not generated.** A deterministic faithfulness gate checks every figure in an answer against tool outputs and pipeline invariants before it's shown.
- **Explanations are exact.** Forecast contributions come from SHAP, and `base + Σ contributions = forecast` is checked as an invariant.
- **Honest limits.** Attribution isn't causation; accuracy is always stated with lead time; known data caveats are disclosed; staffing decisions stay with humans.
- **Pipeline ↔ agent contract.** A schema-validated extract is the only interface between forecasting and the agent.

## Planned architecture
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
- [ ] Security doc (framework-mapped, controls as code) and deployment on a managed AI platform
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
uv run python scripts/serve.py          # API + UI at http://127.0.0.1:8000
```

## Licence
MIT (see `LICENSE`)
