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

