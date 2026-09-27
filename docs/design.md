# Design — Workforce Forecast Explainability Agent

**Status:** design phase · **Last updated:** 2026-09-27

## 1. Problem

Retailers forecast how much work each store will have (units to shelve, transactions to serve, deliveries to receive), then convert that workload into **labour hours** using labour standards (minutes of work per unit of each task). Workforce planners build rotas from those hours.

Planners see the numbers but not the reasons. Three questions come up constantly:

1. **Why** does this store/department/week look like this?
2. **How much should I trust it?**
3. **What changed** since the last forecast?

The answers exist in the forecasting pipeline's outputs (feature contributions, filters applied, labour-standard conversions, historical accuracy), but they're only reachable by someone who can read the code and query the tables. This project builds an agent that answers those questions, and **only says what the data can support**.

## 2. System overview

```
 synthetic / public retail data
            │
   ┌────────▼─────────┐     ┌──────────────────────┐
   │ Forecast pipeline │────▶│ Explainability extract│  (parquet; schema-validated contract)
   │ (walk-forward)    │     └──────────┬───────────┘
   └───────────────────┘                │ ingest (columns checked against contract)
                                        ▼
                               DuckDB query layer (read-only)
                                        │
   ┌────────────────────────────────────▼────────────────────────────────┐
   │ Agent (LangGraph)                                                    │
   │ classify question → route → evidence tools → capability registry     │
   │ → structured explanation → numeric faithfulness gate → answer         │
   └──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Data
- **Default: synthetic generator** (fully reproducible, no licence questions). Stores × departments × workload drivers × days, with trend, weekly/annual seasonality, holidays, promotions, price changes, store openings/closures and injected data-quality issues (gaps, duplicates, outliers), so the agent has real caveats to disclose.
- **Optional:** a loader for a public hierarchical retail sales dataset (check the licence; never commit raw data).
- **Labour standards:** synthetic minutes-per-unit per task, with fixed vs variable elements.

### 2.2 Forecast model (deliberately simple and inspectable)
- Gradient-boosted trees on calendar, price, promotion, event and lag features, per workload driver
- **Walk-forward back-test** for honest accuracy
- **Per-prediction SHAP values** as the decomposition: `prediction = base_value + Σ SHAP_i` (an exact additive identity)
- Daily → weekly aggregation; hours = Σ(driver volume × labour standard), split fixed/variable
- **Naive baseline** (seasonal naive) produced alongside, so "is the model better than simple?" is always answerable

### 2.3 Explainability extract (the pipeline↔agent contract)
One parquet table per run, schema-validated with pandera at write and at ingest.

| Column group | Contents |
|---|---|
| Keys | run_id, run_date, model_version, store_id, department, driver, week |
| Provenance | training window, feature set version, back-test fold |
| Eligibility | is_new_store (ramp-up excluded), is_closed, had_gaps, used_fallback |
| Values | forecast, actual (when known), baseline |
| Contributions (long) | feature_group, feature, shap_value |
| Hours | fixed_hours, variable_hours, total_hours, standard_version |
| Accuracy | error_pct, abs_error, lead_weeks |

**Invariants** (checked in the pipeline, reused as agent eval checks):
- `base_value + Σ shap == forecast` (tol 1e-6)
- `Σ daily == weekly` (tol 0.5%)
- `total_hours == Σ volume × standard` (tol 0.01)

## 3. Agent

### 3.1 Graph
```
classify_question  (why_value | trust | change | missing | hours | baseline | glossary | out_of_scope)
  → route (deterministic, in code)
  → evidence tools (read-only, parameterised SQL; no free-form SQL in v1)
  → build_capability_registry   (which evidence layers returned data + flags)
  → explain (structured output; only layers the registry marks available)
  → faithfulness_check          (every number traces to a tool output; invariants hold)
      pass → answer | fail → degraded answer stating what couldn't be verified
  out_of_scope → decline + offer the in-scope alternative
```

### 3.2 Evidence layers → tools
| Layer | Question | Tool |
|---|---|---|
| Discovery | What exists? | `list_runs`, `find_stores/departments/drivers`, `resolve_name` |
| Provenance | Which run/model/standards produced this? | `get_provenance` |
| Eligibility | Why is this present, missing or using a fallback? | `get_eligibility` |
| Contributions | What drives the number? | `get_contributions(granularity=group|feature)` + waterfall plot |
| Hours | How does volume become hours? | `get_hours_breakdown` |
| Baseline | Is the model better than naive? | `compare_to_baseline` |
| Accuracy | How good has it been, at this lead time? | `get_accuracy` + forecast-vs-actual plot |
| Change | Why did it move since the last run? | `compare_runs` |
| Caveats | What's known to be wrong in this data? | `get_caveats` |

Every tool has **result guards**: `not_found` + closest matches + a next `action`; ambiguous → ask to disambiguate; partial → say what's missing. Plot tools return the plotted data alongside the figure, so the model never describes a chart it can't see.

### 3.3 Output schema
`Explanation{question_type, scope, provenance, headline, equations[], contributions[], hours?, vs_baseline?, accuracy?, caveats[], limitations[], confidence}`. **Numbers come only from tool outputs.**

### 3.4 Behavioural rules
1. SHAP contributions are **model attributions, not causes**: phrase them as "the model attributes…"
2. Always state **lead time** with accuracy (4-weeks-ahead ≠ 1-week-ahead)
3. Always state **units**: volume vs hours
4. Error sign convention stated explicitly: `error = (forecast − actual) / actual`
5. Aggregates (store, region, total) aren't leaves; resolve the level before querying
6. Staffing decisions are out of scope: decline, offer the evidence instead (a partial answer, not a flat refusal)
7. Correct false premises instead of answering them
8. Disclose active caveats for the scope asked about

## 4. Evaluation
1. **Deterministic numeric faithfulness:** every number in the answer matches a tool output; the quoted invariants hold
2. **Structural:** question type, tool choice and order, registry respected, caveats disclosed
3. **LLM-as-judge:** correctness, clarity, attribution-not-causation framing, misuse handling (pass ≥ 0.7; regression flag on a > 0.1 drop vs the saved baseline)
4. **Goldens:** single-turn and multi-turn, including misuse and premise-correction cases; red-team/injection cases

## 5. Production concerns (built, not just listed)
- **Monitoring over time:** forecast drift (error by lead time per run), faithfulness-failure rate, judge-score trend
- **Security:** read-only DB connection, fail-closed auth, prompt-injection evals, PII-free traces, threat model in `docs/security.md`
- **Deployment:** containerised API + minimal UI, live demo
- **Experiment:** graph-based retrieval over the store → department → driver hierarchy vs a flat lookup; keep or kill with evidence

## 6. Decisions
See `docs/decisions.md`.
