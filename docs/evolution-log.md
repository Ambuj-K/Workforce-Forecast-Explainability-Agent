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

