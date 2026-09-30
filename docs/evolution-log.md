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

