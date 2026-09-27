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
- [ ] Synthetic data generator with injected caveats
- [ ] Forecast pipeline + walk-forward back-test + seasonal-naive baseline
- [ ] Explainability extract + invariants
- [ ] Evidence tools + agent graph + faithfulness gate
- [ ] Evals: deterministic, structural, judge; golden cases incl. misuse and injection
- [ ] Monitoring over time, security doc, deployment
- [ ] Graph-vs-flat retrieval experiment over the store hierarchy

## Licence
MIT (see `LICENSE`)
