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

## ADR-010 — scikit-learn gradient boosting + SHAP, raw units, numeric encoding
**Context:** ADR-002 chose gradient boosting + SHAP. The implementation needs exact per-row contributions, native missing-value handling and no system dependencies.
**Options:** LightGBM/XGBoost (native exact contributions, but need the OpenMP system library on macOS) · scikit-learn `HistGradientBoostingRegressor` + the `shap` tree explainer.
**Decision:** scikit-learn + `shap`. The target is **raw volume, not log volume**. Day of week, holiday and region are passed as **plain integers, not native categoricals**.
**Why:**
- Pure Python install (simpler Docker deploy); native NaN handling; measured additivity error 0.0.
- Raw units make each contribution read as "N units" to a planner (log space would make them multiplicative and unreadable).
- **Native categorical splits silently broke SHAP:** contributions stopped adding up to the forecast (max error 1,641 units) and inflated one group about 9×. With integer encoding the error is 0.0. A test now asserts `base + Σ contributions == forecast` for every forecast row.
**Revisit if:** accuracy needs native categoricals, in which case switch to a library with native exact contributions and keep the same invariant test.

## ADR-011 — Hours are a linear function of volume
**Decision:** hours = Σ volume × minutes per unit ÷ 60 + fixed weekly hours prorated by trading days. Hours accuracy is scored on variable hours only.
**Why:** Linearity makes every volume contribution convert exactly into hours, so "the promotion adds 35.9 hours" is verifiable arithmetic, not an estimate. Fixed hours are identical in forecast, baseline and truth, so including them would only dilute the error that planners feel.
**Revisit if:** labour standards become non-linear (e.g. minimum crew sizes, step functions).

## ADR-012 — Live run + as-of extract from reported data only
**Decision:** Each extract has past runs (scorable) and one live run made the day after the last reported data, using planned future inputs. Every table is built as of its run's origin, from reported data; ground-truth tables are never read (tested by blanking them).
**Why:** Planners ask about the next weeks, not the past; data-quality caveats (stale feeds, recent gaps) only appear as of "today". Answer keys stay usable for evaluation without leaking into what the agent sees.

## ADR-013 — The agent's database is locked down at connection time
**Decision:** DuckDB file built from the validated extract; the agent connects with `read_only=True`, `enable_external_access=false`, `lock_configuration=true`.
**Why:** LLM-written SQL is untrusted input. Read-only stops writes; disabling external access stops file/network reads (secrets, other databases); locking stops the session from undoing either. Eight escape attempts are covered by tests.

## ADR-014 — Plain-Python evidence tools with a uniform result
**Decision:** Tools are framework-free methods on `EvidenceTools` returning `ToolResult{status, data, notes, action, source}`; the agent framework wraps them later.
**Why:** Testable without an LLM; one result shape keeps the agent's handling and the faithfulness gate simple; `source` makes every number traceable to a table. Parameterised SQL + strict name resolution make user text inert.
**Revisit if:** a question type can't be answered by the fixed tools (then add a tool, not free-form SQL).

## ADR-015 — The LLM plans and writes; code routes, guards and verifies
**Decision:** Two LLM calls per answer at most (+1 retry): a structured plan and the explanation. Tool selection, clarification, guard handling, declines and number verification are code.
**Why:** Each LLM decision is a place answers can silently go wrong; keeping tool choice and verification deterministic makes behaviour testable with a scripted LLM and keeps every number traceable.
**Revisit if:** questions routinely span several types (then allow a plan with multiple question types, still routed in code).

## ADR-016 — Gemini behind a provider interface
**Decision:** Gemini (free tier, temperature 0, JSON-schema output for plans) via a small `LLM` protocol; a `ScriptedLLM` implements the same protocol for tests.
**Why:** Free for development; the interface lets deployment switch to a managed platform with guardrails (planned) without touching the agent.

## ADR-017 — Deterministic checks first, judge second, artifacts always
**Decision:** Every golden is scored by deterministic checks (routing, gate, evidence values, required/forbidden content, underperformance, leaks); an LLM judge scores only clarity and framing. Results store the answer, evidence and rejected drafts; regressions vs a saved baseline are pass→fail or a judge drop > 0.1.
**Why:** Most failure modes here are checkable in code, and code doesn't drift. The judge is the same model family as the agent and can be noisy; its verdicts are kept with reasoning and treated as a quality signal, not truth. Data-dependent expectations are resolved from the answer key, so goldens survive regeneration.

## ADR-018 — Tools return display-ready numbers
**Decision:** Where an answer needs a derived figure (percentages, differences, totals), the tool returns it, already rounded for display.
**Why:** The LLM truncated 5.77% to 5.7% and summed three contributions itself; the gate caught both, but the cheaper fix is to never ask the model to compute.

## ADR-019 — Accuracy drift is judged relative to the simple method
**Decision:** Alert on drift only when the model's error rises above 1.25× its typical level **and** its edge over the same-weekday method falls below its typical level.
**Why:** Hard periods (Christmas) raise everyone's error; the baseline experiences the same conditions, so the edge isolates model degradation from a hard week.

## ADR-020 — Log interactions for monitoring, not answers
**Decision:** The API logs masked questions, outcomes, gate attempts, rejected values, tools and latency; not the answer text.
**Why:** That's what diagnoses agent health (first-pass rate, fallbacks, slow paths) while keeping personal data out of logs; answers can be regenerated from the extract if needed.

## ADR-021 — Security controls are cited by test, and the citations are tested
**Decision:** `docs/security.md` lists each control with the test or golden that proves it; `tests/test_security_controls.py` fails if a cited test or golden disappears, if a key-shaped secret appears in a tracked file, or if `.env` stops being ignored. Framework mappings (OWASP LLM Top 10, NIST AI RMF, ISO/IEC 42001 themes, EU AI Act) reference control IDs rather than restating them; known gaps are listed with where they get fixed.
**Why:** A security document that isn't checked drifts from the code within weeks. Tying every claim to a passing test makes the document auditable and turns "do we still do X?" into a CI result.
**Revisit if:** controls appear that can't be tested in-repo (e.g. platform identity, gateway rate limits); then cite the deployment config or its policy check instead.

## ADR-022 — Access control lives in the evidence tools, identity at the edge
**Decision:** The API authenticates every data route (bearer tokens stored as hashes, or an identity proxy's header) and maps the user to a role and a set of stores. Each request builds its evidence tools with `allowed_stores`, so other stores don't resolve, aren't listed, and store-free questions aggregate only the user's stores (and say so). Monitoring, which aggregates across stores, is admin-only. There is no implicit "no sign-in" default; it's refused off loopback.
**Why:** Scoping in a prompt can be argued with; scoping in the only code path to the data can't. Putting identity at the edge keeps the same code working with a token file today and the platform's SSO proxy later.
**Revisit if:** users need row-level rules beyond stores (e.g. departments); extend the tools' scope, not the prompt.

## ADR-023 — Platform-agnostic deployment: env config, mounted secrets, one image
**Decision:** All settings come from environment variables; secrets are read from a mounted file (`NAME_FILE`) before the environment; the container is non-root, read-only friendly, signs in by default and builds the synthetic database in a builder stage. LLM calls have a timeout; failures return a plain 503, are logged and alerted on. Requests use a cursor each instead of one global lock.
**Why:** Every major platform can mount secrets as files and run a container with environment config, so the cloud choice changes only the deploy step. Latency was measured before tuning anything: the model reports no thinking tokens and the 4–24 s spread is provider-side queueing on the free tier, so the fix is a paid/provisioned endpoint, not prompt or graph changes.
**Revisit if:** the platform offers a native agent runtime worth adopting; the agent and tools are framework-light enough to move.
