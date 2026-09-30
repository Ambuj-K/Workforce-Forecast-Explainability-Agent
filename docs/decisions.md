# Architecture Decisions

Each decision records the context, the options, the choice and what would make us revisit it.

---

## ADR-001 — Synthetic data by default
**Context:** The agent needs realistic hierarchy, seasonality, events and data-quality problems, and the repo must run anywhere without licence restrictions.
**Options:** public retail dataset only · synthetic only · synthetic default + optional public loader.
**Decision:** Synthetic default + optional public loader.
**Why:** Reproducible, runnable offline, and lets us *inject* known caveats (gaps, duplicates, new stores) so caveat disclosure is testable against ground truth.
**Revisit if:** synthetic patterns prove too easy (evals saturate) → switch evals to the public dataset.

## ADR-002 — Gradient boosting + SHAP as the forecast model
**Context:** Explanations must be exact and inspectable.
**Options:** classical additive time-series model · gradient boosting + SHAP · deep learning.
**Decision:** Gradient boosting + SHAP.
**Why:** Strong tabular baseline; SHAP gives an exact additive identity (`base + Σ contributions = prediction`) that can be checked as an invariant.
**Revisit if:** accuracy vs the seasonal-naive baseline is poor at longer lead times.

## ADR-003 — Precomputed extract + DuckDB, not in-memory DataFrames
**Context:** Row count is stores × departments × drivers × weeks × runs, so it grows fast.
**Decision:** The pipeline writes a schema-validated parquet extract; the agent queries it read-only through DuckDB.
**Why:** Scales past memory, keeps tools simple SQL, and the schema is an explicit contract between pipeline and agent (validated at write *and* ingest).
**Revisit if:** interactive latency exceeds ~2s per tool call.

## ADR-004 — Curated parameterised tools, no free-form SQL in v1
**Decision:** Every evidence layer is a fixed, parameterised query.
**Why:** Predictable, testable and injection-resistant. Free-form SQL is a later fallback only if the question logs show gaps.

## ADR-005 — Deterministic numeric faithfulness gate before answering
**Decision:** A non-LLM check that every number in the answer traces to a tool output and that quoted invariants hold. On failure, give a degraded answer that says what couldn't be verified.
**Why:** LLM judges miss numeric fabrication; a deterministic check doesn't.

## ADR-006 — LangGraph with routing in code
**Decision:** Question classification → deterministic routing → evidence tools → registry → explanation → gate.
**Why:** Workflows written in prompts drift; routing in code is testable and traceable.

## ADR-007 — Keep the truth next to the corrupted data
**Context:** Evals need to know which rows are wrong and what the right values were.
**Options:** corrupt in place and discard the truth · keep a list of corrupted keys only · keep full clean volumes + a caveats registry.
**Decision:** Keep full `true_volumes` plus a `caveats` table (type, store, driver, dates, affected rows, detail). Injected problems never overlap on a store-day.
**Why:** Forecast error can be measured against clean truth, and the agent's caveat disclosure can be scored deterministically (did it mention caveat C00n for this scope?). Non-overlap keeps each check independent.
**Revisit if:** realistic overlapping failures (e.g. a gap inside a late feed) become a test goal.

## ADR-008 — One model per driver, past volume lagged by the full horizon
**Context:** Forecasts are needed for every day up to 28 days ahead, and every feature must be explainable.
**Options:** recursive forecasting (feed predictions back) · one model per horizon step · one model with all past-volume features lagged ≥ horizon.
**Decision:** One model per driver; past-volume features end at least 28 days before the target date; planned inputs (calendar, promotions, prices, store facts) are known in advance. Horizon must be a multiple of 7 to keep weekday alignment.
**Why:** No error compounding, one model to explain, and contributions stay tied to real observed inputs. The cost: short-term momentum inside the horizon is not used.
**Revisit if:** accuracy at 1–7 days ahead lags a seasonal-naive baseline.

## ADR-009 — Outlier detection with two baselines, and no judging during ramp-up
**Decision:** A spike must exceed 4× the same-weekday median **and** 2× the recent 7-day median; a dropout is a zero on a trading day with a same-weekday baseline ≥ 10. Stores in their first 91 days are not judged.
**Why:** Measured against the answer key (see evolution log 2.2): 59/60 recall, 0 false positives across 10 seeds. The single miss falls inside a ramp-up window that is already disclosed.
**Revisit if:** new caveat types (e.g. level shifts) are added to the generator.

