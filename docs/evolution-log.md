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

