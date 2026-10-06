# Evolution Log

Every meaningful change: the problem observed, the fix applied, the lesson learned.

---

## Phase 0 — Design (2026-09-27)
- Problem framed: explain retail labour-demand forecasts (why / trust / what changed) with verified numbers.
- Decisions ADR-001 to ADR-006 recorded in `docs/decisions.md`.

---

## Phase 1 — Synthetic Data With Ground-Truthed Caveats (2026-09-29)

### Iteration 1.1 — Generator
**Problem:** The agent's core promise is disclosing data problems and staying faithful to numbers. That can't be tested on real data, where nobody knows which rows are wrong.
**Fix:** A deterministic generator (`src/wfx/data/`) producing stores × 5 workload drivers × days, with trend, weekly and annual seasonality, holidays (including non-trading days), week-long promotions, step price changes, new stores that ramp up and a store that closes. Labour standards (minutes per unit + fixed weekly hours) convert volume to hours.
**Ground truth:** clean `true_volumes` are kept alongside the reported `volumes`. Every injected problem (gaps, spikes/dropouts, duplicate loads, a late feed) plus structural facts (ramp-up, closure) is recorded in a `caveats` table with exact dates and affected rows.
**Result:** 3 years × 12 stores ≈ 59k rows in under a second; 18 tests, including one that proves every difference between reported and true data is explained by a recorded caveat.
**Lesson:** Generate the answer key together with the data. Caveat disclosure and forecast accuracy become measurable, not judged.

### Iteration 1.2 — `.gitignore` swallowed a source package
**Problem:** The pattern `data/` (meant for generated datasets) matches *every* directory named `data`, so `src/wfx/data/` was silently excluded from the first commit.
**Fix:** Anchor it to the repo root: `/data/`.
**Lesson:** Anchor ignore patterns for generated artefacts, and run `git status` after adding a package directory.

---

## Phase 2 — Data Preparation and Features (2026-09-30)

### Iteration 2.1 — Clean grid + quality report
**Built:** `prepare_observations()` builds the full trading-life grid per store × driver, removes duplicate loads, marks missing days, and detects spikes/dropouts. Every finding goes into a quality report (issue type, keys, reported value, baseline, detail) for the agent to disclose. It never reads the ground-truth tables (a test proves it by blanking them).

### Iteration 2.2 — Outlier detector: measured, not eyeballed
Because the generator has an answer key, the detector was tuned against it across 10 seeds.
| Version | Rule | Recall | False positives |
|---|---|---|---|
| v1 | value > 4× same-weekday median (8 weeks) | 6/6 | **247** (one seed) |
| v2 | + value > 4× median of the last 7 days | 5/6 | 0 |
| v3 | recent-level check loosened to 2× | 60/60 | 62 |
| **v4** | + stores not judged in their first 91 days | **59/60** | **0** |
**What each failure taught:** v1 flagged a new store's normal growth (ramp-up days 23–31) as spikes; v2 missed Sunday pallet spikes because the 7-day median is inflated by busy Mondays (a Sunday runs at half a normal day); v3's residual false positives all sat in the steepest part of a ramp-up. v4's single miss is on a new store's day 90, inside the not-judged window: a designed trade-off, since that period is disclosed as a ramp-up anyway.
**Lesson:** A detector without labelled data gets tuned to look right; with an answer key it gets tuned to *be* right, and every rule change leaves a number behind.

### Iteration 2.3 — Leakage-safe features with explainable groups
**Built:** 18 features in 7 groups (calendar, holiday, promotion, price, store, recent level, seasonal history), each with a plain-English description. Past-volume features all end ≥ 28 days (the horizon) before the row's date; planned inputs are used as known in advance.
**Proof:** a test tampers with every volume after a forecast origin and asserts that no history feature within the horizon changes, and that the tampering does reach rows beyond it (so the test can fail).

---

## Phase 3 — Models, Exact Contributions, Walk-Forward Back-Test (2026-09-30)

### Iteration 3.1 — One model per driver + seasonal-naive baseline
**Built:** `forecasting/model.py` (scikit-learn gradient boosting per driver; non-trading days forecast as 0 by rule; `explain()` returns per-row, per-feature contributions in volume units) and `forecasting/backtest.py` (22 Monday origins, 28 days apart, retrain at each; every forecast paired with a seasonal-naive baseline: the same weekday one horizon earlier).
**Result (3 years, 12 stores, scored against ground truth, trading days):**
| Driver | WAPE model | WAPE baseline | Skill |
|---|---|---|---|
| grocery cases | 7.1% | 13.1% | +46% |
| online orders | 9.3% | 14.0% | +34% |
| pallets received | 10.4% | 15.6% | +33% |
| transactions | 5.8% | 9.5% | +39% |
| fresh units | 9.5% | 17.4% | +45% |
By lead week: 6.8% → 6.8% → 7.2% → 7.3% (baseline 11.1% → 12.9%). The model beats the baseline at every driver × lead-week combination.

### Iteration 3.2 — The first origin crashed on a year-ago feature
**Problem:** With a 364-day warm-up, `lag_364` had no values at all in the first training set, and the library failed with an unrelated-looking binning error.
**Fix:** Warm-up raised to 455 days (a year + 13 weeks of last-year history); `train()` now names any feature with no history instead of surfacing a library error.

### Iteration 3.3 — SHAP silently broke on categorical splits
**Problem:** With day of week, holiday and region as native categorical features, the SHAP tree explainer returned contributions that did **not** add up to the forecast: max error **1,641 units**, and promotions looked ~9× more important than they were. No warning, no exception.
**Fix:** Encode them as plain integers. Additivity error: **0.0**. Accuracy cost: ≤ 0.2 WAPE points on any driver.
**Guard:** a test asserts `base + Σ contributions == forecast` (tolerance 1e-6) for every forecast row.
**Lesson:** Explanations need their own invariant tests. A popular library produced confident, wrong attributions; only the arithmetic check caught it. This is exactly the failure the agent's faithfulness gate is designed for.

### Iteration 3.4 — Hours conversion, explained exactly (2026-10-01)
**Built:** `forecasting/hours.py`. Weekly hours per store × department = Σ(volume × minutes per unit ÷ 60) + fixed hours per week (prorated by trading days, so openings and closures get partial weeks). Because hours are linear in volume, per-feature volume contributions convert into **exact per-feature hour contributions** (additivity error ~1e-13).
**Example (store S001, grocery, week of 10 Nov 2025):** 270.9 h = 15.0 fixed + 255.9 variable; variable = 171.1 base + 35.9 promotion + 22.4 store + 18.5 recent level + 8.5 calendar + 0.4 seasonal history − 0.8 holiday − 0.1 price.
**Weekly hours accuracy (variable hours, 22 origins):**
| Department | WAPE model | WAPE baseline | Skill |
|---|---|---|---|
| checkouts | 2.4% | 4.9% | +50% |
| fresh | 4.7% | 12.2% | +61% |
| grocery | 3.3% | 8.0% | +55% |
| online | 4.4% | 7.2% | +38% |
| receiving | 4.6% | 7.8% | +40% |
**Lesson:** Choose the output unit the user acts on (hours per week) and keep the explanation exact all the way into it. Weekly errors are about half the daily ones because day-to-day noise cancels.

---

## Phase 4 — Explainability Extract and the Agent's Database (2026-10-02)

### Iteration 4.1 — A live run needs planned future inputs
**Problem:** Back-test runs alone can't show the caveats a planner meets *today*. The late-feed store only looks stale when the forecast is made the day after the data ends.
**Fix:** The generator now extends the calendar, promotions and prices 28 days past the last data day (as planned inputs are known in reality) and keeps what then happens in a separate answer-key table. Future inputs use an independent random stream: **historical data is byte-identical** (checksums before/after match; a test asserts it).

### Iteration 4.2 — The extract: one contract between pipeline and agent
**Built:** `explain/extract.py`. An extract holds 5 past runs + 1 live run, each with forecasts, exact per-feature contributions (volume and weekly hours), eligibility flags (ramping, stale, missing / outlier / duplicate counts in the last 28 days, closures), quality issues and accuracy history, all **as of the run's origin, from reported data only**. Schemas (pandera) + invariants (forecast = base + contributions; closed days 0 with no contributions; hours add up; one live run; accuracy windows end before the live origin) are enforced at build, write **and** load.
**Result:** the live run flags exactly one stale store (the injected late feed) and the injected grocery gap, with no access to the answer key (tests prove both).

### Iteration 4.3 — Lock the SQL engine, not just the data
**Built:** `explain/store.py`. The agent's DuckDB file is opened read-only with external access disabled and the configuration locked.
**Tested attacks (all blocked, for the right reason):** CREATE/DELETE (read-only database) · `read_csv('…/secret.env')`, `FROM '…/secret.env'`, `COPY … TO`, `ATTACH`, `INSTALL` (permission errors: no external access) · `SET enable_external_access = true` (configuration locked).
**Lesson:** If an LLM writes SQL, a read-only *data source* isn't enough: the engine itself can read files. Lock the engine at connection time and test the escapes.

---

## Phase 5 — Evidence Tools (2026-10-02)

### Iteration 5.1 — Runs that overlap
**Problem:** "What changed since the last run?" is the question planners ask most after a refresh, but runs spaced one horizon apart never cover the same week.
**Fix:** The extract now holds the live run, **3 weekly runs before it** (overlapping its window, for change analysis) and **4 older runs** one horizon apart whose windows have fully played out (for accuracy).

### Iteration 5.2 — Eleven read-only tools with one result shape
**Built:** `explain/tools.py`: `list_runs`, `find_entities`, `get_provenance`, `explain_week_hours`, `explain_day_volume`, `get_hours_breakdown`, `compare_to_baseline`, `get_accuracy`, `get_caveats`, `compare_runs`, `describe_feature`. Every query is parameterised; every result is `{status, data, notes, action, source}`. Notes are generated from the data ("S002 cases_filled: data is stale; latest actuals 2025-12-23, 6 days before the run"), so every sentence the agent repeats traces to a number.
**Guards:** user names resolved with a disclosed note ("Interpreted store 'store 1' as 'S001'"); ambiguous names → ask, never guess; unknown names, runs or weeks → `not_found` with the valid options and a next action; non-Monday dates → normalised with a note; non-trading days → explained as a rule, not by the model.
**Exactness carried through:** tests assert day and week explanations add up to the forecast, hour breakdowns match weekly totals, and run-to-run group changes add up to the total change.

### Iteration 5.3 — A junk name resolved to a real store
**Problem:** The injection test passed `"S001' OR 1=1 --"`. The SQL was safe (parameterised), but the digit-based matcher **resolved the junk to S001**, silently answering a question nobody asked.
**Fix:** Store names must be store-shaped (`S001`, `s1`, `store 1`, `1`); anything else is `not_found`.
**Lesson:** Injection safety isn't only about SQL. A lenient resolver turns malformed input into a confident wrong answer.

### Iteration 5.4 — An honest number the agent must not hide
For store S002 transactions, the model is **worse than the seasonal-naive baseline at 1 week ahead** (6.0% vs 4.9% WAPE) and better at weeks 2–4. The accuracy tool reports it per lead week; the agent's prompt and evals must require it to be stated, not averaged away.

---

## Phase 6 — The Agent (2026-10-03)

### Iteration 6.1 — The LLM does two narrow jobs; code does the rest
**Built:** `agent/` (LangGraph). The LLM (Gemini, behind a provider interface) **plans** (question type + the names/dates the user stated, as a structured object) and **writes** the explanation. Code decides everything else:
- **Routing:** each question type maps to a fixed list of evidence tools (`agent/routing.py`); the LLM never picks tools.
- **Missing facts:** a question without a needed store/department/week gets a clarifying question, not a guess.
- **Guards:** a tool returning not_found/ambiguous ends in a guided reply with the valid options; the explainer isn't called.
- **Declines:** staffing decisions are declined with an offer of the evidence; off-topic questions are declined.

### Iteration 6.2 — The numeric faithfulness gate
**Built:** `agent/faithfulness.py`. Every number and date in a draft answer must exist in the evidence (rounding and fraction→percentage allowed; numbers from the question allowed). Computed numbers are rejected: if an answer needs a total or difference, a tool provides it.
**Loop:** fail → the explainer gets the exact unsupported values and rewrites once → fail again → a templated answer built only from evidence (which passes the gate by construction).
**Tested:** invented values, a sum the LLM computed itself, and wrong dates (ISO and "12 January 2026") are all caught; grounded rounding, percentages and worded dates pass.

### Iteration 6.3 — Deterministic agent tests
A scripted LLM replays prepared plans and drafts, so every route (answer, retry, fallback, clarify, guard, decline) is tested without network calls or cost. 107 tests in total.

### Iteration 6.4 — First live run on Gemini (2026-10-04)
Nine real questions, every route exercised; every written explanation passed the faithfulness gate on the first attempt.
| Question | Outcome |
|---|---|
| Why does store 1 grocery need these hours next week? | Answered: "next week" resolved from the run date; every figure traced (217.8 h total; base 171.9, store +19.5, recent trend +13.3, calendar +7.1, promotion −6.7); "the model attributes" framing |
| Can I trust the transactions forecast for store 2? | Answered: **stated the model was worse than the simple baseline one week ahead (6.0% vs 4.9%)**, better at weeks 2–4; disclosed stale data and 5 missing days |
| Anything to be careful about this week? | Answered: the stale store across all five drivers + a 3-day grocery gap (both injected problems, found from reported data) |
| What changed since last week's run? | Answered: −1.1 h total, by group, with the "newer run saw more recent data" note |
| Should I cut two staff? / London weather? | Declined (staffing with an offer of evidence; off-topic) |
| Store 1 *bakery*? / week of 2 February? | Guided replies listing valid departments / covered weeks |
| Why are store 4 fresh hours up? | Asked which week |
**Defect found:** the bakery reply leaked an internal tool instruction to the user ("Ask the user to pick one of the listed values").
**Fix:** tool results now carry a separate user-facing `message`; `action` stays internal. A test asserts internal instructions never reach the user.
**Lesson:** Results written for the agent and messages written for the user are different contracts; keep them in different fields.

---

## Phase 7 — Evals (2026-10-05)

### Iteration 7.1 — Goldens, deterministic checks, judge, baseline
**Built:** `evals/`: 25 golden questions (why hours/volume, trust, caveats, change, baseline, breakdown, definitions, runs, staffing, off-topic, guards, clarification, false premise, injection). Facts that depend on the data (which store is stale, which had a gap, which closed) are filled in from the answer key at eval time.
**Deterministic checks:** outcome, question type, tools, first-pass gate, required mentions, required evidence values, forbidden strings, attribution framing, underperformance stated when the evidence shows it, no staffing advice, no internal leaks. **Judge** (Gemini, structured verdict per criterion) only for clarity and framing. Every result stores the answer, evidence and rejected drafts; runs are compared with a saved baseline (pass→fail or judge drop > 0.1).

### Iteration 7.2 — What the first runs found
| Run | Passed | Finding → fix |
|---|---|---|
| 1 | 23/25 | "recent level" didn't match the group `recent_level` → tool normalises + fuzzy-matches terms |
| 1 | | False premise ("why did the promotion add 80 hours?"): gate rejected the correct *correction* ("the data does not show 80 hours"), fell back → a large question figure is now allowed **only in a sentence that negates it**; agreement still fails |
| 2 | 24/25 | Trust answer rejected once: model wrote **5.7%** for **0.0577** (truncated, not rounded) → the accuracy tool now returns display-ready percentages and a generated note when the model is worse than the baseline. **Don't make the LLM do arithmetic, not even ×100** |
| 3 | 25/25 | Judge drops flagged by the regression check: one real (first sentence gave the total, not the *why*) → prompt rule; one judge error (complained a +0.2 h factor was omitted from "biggest drivers") → recorded as judge variance, not chased |
| 4–5 | 24/25 | The new opening-sentence rule made the model summarise minor factors and **add them up itself** ("holidays, pricing and seasonal history combined: −2.3 hours"), twice in a row → the gate rejected it each time. Banning summaries would make answers worse, so the explain tools now return **top 3 drivers + the other groups combined, pre-computed** → 9/9 first-pass on why-questions |

**Gate gap closed on the way:** numbers from the question had been allowed unconditionally, so a false premise ("…add 80 hours?") could be repeated as fact. Now only small identifiers (≤ 52) pass freely.
**Lesson:** Each fix landed where the problem was, not where it showed up: matching in the tool, arithmetic in the tool, wording in the prompt, a premise rule in the gate, and an expectation fix in the golden when the agent was right and the test was wrong.

---

## Phase 8 — Monitoring, API and UI (2026-10-06)

### Iteration 8.1 — Forecast monitoring from the agent's own database
**Built:** `monitoring/forecast.py`: accuracy per scored run (overall and by lead week, vs the same-weekday method), data quality per run (stale feeds, missing days, excluded outliers, duplicates, ramping stores), and alert rules: accuracy drift, worse than baseline, a store-driver worse than baseline in consecutive runs, stale data in the live run.
**On the real extract:** error 6.5% → 6.5% → 6.7% → 7.8% with the model ahead of the baseline every run; the stale feed raises the only alert.

### Iteration 8.2 — Drift must be relative, or every December is an incident
**Problem:** The latest scored window covers Christmas: the model's error rose to 7.8%, but the simple method's rose from 10.7% to 16.9%, so the model's edge actually *grew* (+54%). A raw-error drift rule would fire every December.
**Fix:** Drift = error above 1.25× its typical level **and** the edge over the simple method below its typical level. Tests cover both a Christmas-like run (no alert) and real drift (alert).
**Lesson:** Monitor the model against a baseline that faces the same conditions, not against its own history alone.

### Iteration 8.3 — Agent monitoring
**Built:** `monitoring/agent.py`: every question through the API is logged (masked question, outcome, gate attempts, rejected values, tools, latency; answers are not stored). Health: first-pass gate rate, fallback rate, latency p50/p95, most-rejected values; alerts on low first-pass, high fallback, slow p95 (only above a minimum sample). Eval history comes from saved eval results: pass rate 92% → 96% → 100% → 96% → 100% over the C6 runs.

### Iteration 8.4 — API + minimal UI
**Built:** `api/app.py` (FastAPI: `/`, `/health`, `/ask`, `/runs`, `/monitoring`) and a single-page UI (ask, example questions, outcome badge, evidence used, alerts, accuracy trend, agent health). Questions are length-limited; the shared read-only DuckDB connection is used under a lock (one question at a time, fine for a demo, noted for deployment).
**Live check (Gemini):** answers, decline and guided reply all correct over HTTP; monitoring showed the stale-data alert and 100% first-pass. **Explained answers take 12–15 s** (two free-tier calls), right at the 15 s p95 threshold: the deployment step should address it.

