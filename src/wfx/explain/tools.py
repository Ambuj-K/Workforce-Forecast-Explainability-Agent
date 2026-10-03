"""Evidence tools: the agent's only way to read the extract.

Every tool runs fixed, parameterised SQL on the locked-down agent database and returns
a ``ToolResult`` with the same shape:

* ``status``: ok | not_found | ambiguous | partial
* ``data``: rows straight from the database (numbers are never computed by the LLM)
* ``notes``: facts the answer must disclose, generated from the data itself
* ``action``: what to do next when the status is not ok
* ``source``: the tables the evidence came from

Names typed by a user ("store 2", "pallets", "Grocery") are resolved before querying;
uncertain matches are reported, never silently guessed.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

import duckdb
import pandas as pd

Status = Literal["ok", "not_found", "ambiguous", "partial"]


@dataclass
class ToolResult:
    status: Status
    data: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    action: str | None = None
    source: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()[:10]
    if pd.isna(value):
        return None
    return str(value)


class EvidenceTools:
    """Read-only evidence tools over a ``connect_readonly`` connection."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con
        self._stores = self._column("SELECT DISTINCT store_id FROM forecasts ORDER BY 1")
        self._departments = self._column("SELECT DISTINCT department FROM labour_standards ORDER BY 1")
        self._drivers = self._column("SELECT DISTINCT driver FROM labour_standards ORDER BY 1")

    # ------------------------------------------------------------------ plumbing

    def _rows(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        frame = self._con.execute(sql, params or []).fetch_df()
        return frame.astype(object).where(frame.notna(), None).to_dict("records")

    def _column(self, sql: str) -> list[str]:
        return [row[0] for row in self._con.execute(sql).fetchall()]

    def _run(self, run_id: str | None) -> tuple[dict[str, Any] | None, ToolResult | None]:
        if run_id is None:
            rows = self._rows("SELECT * FROM runs WHERE is_live")
            return rows[0], None
        rows = self._rows("SELECT * FROM runs WHERE run_id = ?", [run_id])
        if rows:
            return rows[0], None
        return None, ToolResult(
            "not_found",
            data=self._rows("SELECT run_id, origin, is_live FROM runs ORDER BY origin"),
            action="Use one of the listed run_id values, or omit run_id for the live run.",
            source=["runs"],
        )

    def _resolve(self, kind: str, text: str, options: list[str]) -> tuple[str | None, list[str], ToolResult | None]:
        """Resolve user text to one known value. Returns (value, notes, error)."""
        raw = str(text).strip()
        lowered = {o.lower(): o for o in options}
        if raw.lower() in lowered:
            return lowered[raw.lower()], [], None
        candidates: list[str] = []
        if kind == "store":
            shaped = re.fullmatch(r"(?:store\s*|s\s*)?0*(\d+)", raw, flags=re.IGNORECASE)  # "S001", "s1", "store 1", "1"
            if shaped:
                candidates = [o for o in options if int(re.sub(r"\D", "", o)) == int(shaped.group(1))]
        else:
            norm = raw.lower().replace(" ", "_")
            candidates = [o for o in options if norm in o or o in norm]
            if not candidates:
                candidates = difflib.get_close_matches(norm, options, n=3, cutoff=0.6)
        if len(candidates) == 1:
            return candidates[0], [f"Interpreted {kind} '{raw}' as '{candidates[0]}'."], None
        if len(candidates) > 1:
            return None, [], ToolResult(
                "ambiguous",
                data=[{kind: c} for c in candidates],
                action=f"Ask the user which {kind} they mean.",
            )
        return None, [], ToolResult(
            "not_found",
            data=[{kind: o} for o in options],
            action=f"No {kind} matches '{raw}'. Ask the user to pick one of the listed values.",
        )

    def _week(self, run: dict[str, Any], department: str, store: str, week: str | date) -> tuple[pd.Timestamp | None, list[str], ToolResult | None]:
        day = pd.Timestamp(week)
        start = day - pd.Timedelta(days=day.dayofweek)
        notes = [] if day == start else [f"{day.date()} is not a Monday; using the week starting {start.date()}."]
        weeks = [
            r["week_start"]
            for r in self._rows(
                "SELECT DISTINCT week_start FROM weekly_hours WHERE run_id = ? AND store_id = ? AND department = ? ORDER BY 1",
                [run["run_id"], store, department],
            )
        ]
        if start in weeks:
            return start, notes, None
        return None, notes, ToolResult(
            "not_found",
            data=[{"week_start": w} for w in weeks],
            action=f"Run {run['run_id']} has no week starting {start.date()} for this store and department; use a listed week or another run.",
            source=["weekly_hours"],
        )

    # --------------------------------------------------------------- discovery

    def list_runs(self) -> ToolResult:
        """All forecast runs, oldest first, with which one is live."""
        return ToolResult(
            "ok",
            data=self._rows("SELECT run_id, origin, is_live, data_through, horizon_days FROM runs ORDER BY origin"),
            source=["runs"],
        )

    def find_entities(self) -> ToolResult:
        """Known stores, departments and drivers (with their department)."""
        return ToolResult(
            "ok",
            data=[
                {"stores": self._stores},
                {"departments": self._departments},
                {"drivers": self._rows("SELECT driver, department FROM labour_standards ORDER BY department")},
            ],
            source=["forecasts", "labour_standards"],
        )

    def get_provenance(self, run_id: str | None = None) -> ToolResult:
        """Which run, made when, trained on data up to when, with which model/feature/labour-standard versions."""
        run, error = self._run(run_id)
        if error:
            return error
        return ToolResult("ok", data=[run], source=["runs"])

    # ------------------------------------------------------------- explanation

    def explain_week_hours(self, store: str, department: str, week_start: str | date, run_id: str | None = None) -> ToolResult:
        """Why a store-department-week has its hours: fixed + base + hours from each feature group."""
        run, error = self._run(run_id)
        if error:
            return error
        store_id, n1, error = self._resolve("store", store, self._stores)
        if error:
            return error
        dept, n2, error = self._resolve("department", department, self._departments)
        if error:
            return error
        week, n3, error = self._week(run, dept, store_id, week_start)
        if error:
            return error
        totals = self._rows(
            "SELECT store_id, department, week_start, lead_week, days, fixed_hours, variable_hours, total_hours, "
            "baseline_variable_hours FROM weekly_hours WHERE run_id = ? AND store_id = ? AND department = ? AND week_start = ?",
            [run["run_id"], store_id, dept, week],
        )
        parts = self._rows(
            "SELECT \"group\", hours FROM weekly_hour_contributions "
            "WHERE run_id = ? AND store_id = ? AND department = ? AND week_start = ? ORDER BY abs(hours) DESC",
            [run["run_id"], store_id, dept, week],
        )
        notes = [*n1, *n2, *n3, f"Run {run['run_id']} (made {pd.Timestamp(run['origin']).date()})."]
        notes.append("Contributions are the model's attributions, not proven causes; groups add up exactly to the variable hours.")
        return ToolResult("ok", data=[{"totals": totals[0], "contributions": parts}], notes=notes, source=["weekly_hours", "weekly_hour_contributions"])

    def explain_day_volume(self, store: str, driver: str, day: str | date, run_id: str | None = None, by: Literal["group", "feature"] = "group") -> ToolResult:
        """Why a store-driver-day forecast is what it is: base value + contributions by group or feature."""
        run, error = self._run(run_id)
        if error:
            return error
        store_id, n1, error = self._resolve("store", store, self._stores)
        if error:
            return error
        drv, n2, error = self._resolve("driver", driver, self._drivers)
        if error:
            return error
        day_ts = pd.Timestamp(day)
        forecast = self._rows(
            "SELECT store_id, driver, date, lead_days, is_trading_day, method, forecast, baseline, base_value "
            "FROM forecasts WHERE run_id = ? AND store_id = ? AND driver = ? AND date = ?",
            [run["run_id"], store_id, drv, day_ts],
        )
        if not forecast:
            return ToolResult(
                "not_found",
                action=f"Run {run['run_id']} has no forecast for {store_id}/{drv} on {day_ts.date()}; its window starts {pd.Timestamp(run['origin']).date()} and lasts {run['horizon_days']} days.",
                source=["forecasts"],
            )
        row = forecast[0]
        if not row["is_trading_day"]:
            return ToolResult("ok", data=[{"forecast": row, "contributions": []}], notes=[*n1, *n2, "Stores do not trade that day; the forecast is 0 by rule, not by the model."], source=["forecasts"])
        column = '"group"' if by == "group" else "feature"
        parts = self._rows(
            f"SELECT {column} AS name, sum(contribution) AS contribution FROM contributions "  # column is one of two literals
            "WHERE run_id = ? AND store_id = ? AND driver = ? AND date = ? GROUP BY 1 ORDER BY abs(sum(contribution)) DESC",
            [run["run_id"], store_id, drv, day_ts],
        )
        notes = [*n1, *n2, "Contributions are the model's attributions, not proven causes; base + contributions = forecast exactly."]
        return ToolResult("ok", data=[{"forecast": row, "contributions": parts}], notes=notes, source=["forecasts", "contributions"])

    def get_hours_breakdown(self, store: str, department: str, week_start: str | date, run_id: str | None = None) -> ToolResult:
        """How volume becomes hours for a store-department-week: forecast volume per driver x labour standard + fixed hours."""
        run, error = self._run(run_id)
        if error:
            return error
        store_id, n1, error = self._resolve("store", store, self._stores)
        if error:
            return error
        dept, n2, error = self._resolve("department", department, self._departments)
        if error:
            return error
        week, n3, error = self._week(run, dept, store_id, week_start)
        if error:
            return error
        drivers = self._rows(
            "SELECT f.driver, sum(f.forecast) AS forecast_volume, l.minutes_per_unit, "
            "sum(f.forecast) * l.minutes_per_unit / 60 AS variable_hours, l.fixed_hours_per_week, l.standard_version "
            "FROM forecasts f JOIN labour_standards l USING (driver) "
            "WHERE f.run_id = ? AND f.store_id = ? AND f.department = ? AND f.date >= ? AND f.date < ? + INTERVAL 7 DAY "
            "GROUP BY f.driver, l.minutes_per_unit, l.fixed_hours_per_week, l.standard_version",
            [run["run_id"], store_id, dept, week, week],
        )
        totals = self._rows(
            "SELECT days, fixed_hours, variable_hours, total_hours FROM weekly_hours "
            "WHERE run_id = ? AND store_id = ? AND department = ? AND week_start = ?",
            [run["run_id"], store_id, dept, week],
        )
        notes = [*n1, *n2, *n3, "Hours = volume x minutes per unit / 60 + fixed weekly hours (prorated by trading days)."]
        return ToolResult("ok", data=[{"totals": totals[0], "drivers": drivers}], notes=notes, source=["forecasts", "labour_standards", "weekly_hours"])

    def compare_to_baseline(self, store: str, department: str, week_start: str | date, run_id: str | None = None) -> ToolResult:
        """Model vs seasonal-naive baseline (same weekday one horizon earlier) for a week's variable hours."""
        result = self.explain_week_hours(store, department, week_start, run_id)
        if result.status != "ok":
            return result
        totals = result.data[0]["totals"]
        model, baseline = totals["variable_hours"], totals["baseline_variable_hours"]
        if baseline is None:
            return ToolResult(
                "partial",
                data=[{"model_variable_hours": model, "baseline_variable_hours": None}],
                notes=[*result.notes, "No baseline for this week: some days lack history one horizon earlier (e.g. a new store)."],
                source=["weekly_hours"],
            )
        return ToolResult(
            "ok",
            data=[{"model_variable_hours": model, "baseline_variable_hours": baseline, "difference_hours": model - baseline}],
            notes=result.notes,
            source=["weekly_hours"],
        )

    # ------------------------------------------------------------------- trust

    def get_accuracy(self, store: str | None = None, driver: str | None = None, run_id: str | None = None) -> ToolResult:
        """Past accuracy (WAPE) of model and baseline by lead week, from windows that ended before the run."""
        run, error = self._run(run_id)
        if error:
            return error
        notes: list[str] = []
        filters, params = ["window_end < ?"], [run["origin"]]
        if store is not None:
            store_id, n, error = self._resolve("store", store, self._stores)
            if error:
                return error
            filters.append("store_id = ?")
            params.append(store_id)
            notes += n
        if driver is not None:
            drv, n, error = self._resolve("driver", driver, self._drivers)
            if error:
                return error
            filters.append("driver = ?")
            params.append(drv)
            notes += n
        rows = self._rows(
            "SELECT lead_week, count(DISTINCT scored_run_id) AS runs_scored, sum(days) AS days, "
            "sum(abs_err_model) / sum(actual) AS wape_model, sum(abs_err_baseline) / sum(actual) AS wape_baseline "
            f"FROM accuracy WHERE {' AND '.join(filters)} GROUP BY lead_week ORDER BY lead_week",  # filters are fixed literals
            params,
        )
        if not rows:
            return ToolResult("not_found", notes=notes, action="No completed past runs to score yet for this selection.", source=["accuracy"])
        notes.append("WAPE = total absolute error / total actual volume; lower is better. Actuals are reported (cleaned) volumes.")
        notes.append(f"Scored windows all ended before {pd.Timestamp(run['origin']).date()}.")
        return ToolResult("ok", data=rows, notes=notes, source=["accuracy"])

    def get_caveats(self, store: str | None = None, run_id: str | None = None) -> ToolResult:
        """What to be careful about as of the run: stale data, recent gaps/outliers/duplicates, ramping or closing stores."""
        run, error = self._run(run_id)
        if error:
            return error
        notes: list[str] = []
        store_filter, params = "", [run["run_id"]]
        if store is not None:
            store_id, n, error = self._resolve("store", store, self._stores)
            if error:
                return error
            store_filter, params = " AND store_id = ?", [run["run_id"], store_id]
            notes += n
        flags = self._rows(
            "SELECT store_id, driver, latest_actual_date, data_is_stale, is_ramping, days_since_open, closes_in_horizon, close_date, "
            "missing_days_28, outliers_excluded_28, duplicate_rows_removed_28 FROM eligibility WHERE run_id = ?" + store_filter
            + " AND (data_is_stale OR is_ramping OR closes_in_horizon OR missing_days_28 > 0 OR outliers_excluded_28 > 0 "
            "OR duplicate_rows_removed_28 > 0) ORDER BY store_id, driver",
            params,
        )
        origin = pd.Timestamp(run["origin"])
        for f in flags:
            where = f"{f['store_id']} {f['driver']}"
            if f["data_is_stale"]:
                lag = (origin - pd.Timestamp(f["latest_actual_date"])).days
                notes.append(f"{where}: data is stale; latest actuals {pd.Timestamp(f['latest_actual_date']).date()}, {lag} days before the run.")
            if f["missing_days_28"]:
                notes.append(f"{where}: {f['missing_days_28']} missing day(s) in the 28 days before the run.")
            if f["outliers_excluded_28"]:
                notes.append(f"{where}: {f['outliers_excluded_28']} suspicious value(s) excluded from training in the last 28 days.")
            if f["duplicate_rows_removed_28"]:
                notes.append(f"{where}: {f['duplicate_rows_removed_28']} duplicate row(s) removed in the last 28 days.")
            if f["is_ramping"]:
                notes.append(f"{where}: new store ({f['days_since_open']} days trading); forecasts are less certain while it ramps up.")
            if f["closes_in_horizon"]:
                notes.append(f"{where}: store closes on {pd.Timestamp(f['close_date']).date()}, inside this forecast window.")
        if not flags:
            notes.append("No data-quality or store-status caveats for this selection as of the run.")
        return ToolResult("ok", data=flags, notes=notes, source=["eligibility"])

    def compare_runs(self, store: str, department: str, week_start: str | date, earlier_run_id: str | None = None, later_run_id: str | None = None) -> ToolResult:
        """What changed between two runs for the same store-department-week, by feature group."""
        later, error = self._run(later_run_id)
        if error:
            return error
        if earlier_run_id is None:
            previous = self._rows(
                "SELECT run_id FROM runs WHERE origin < ? AND origin + horizon_days * INTERVAL 1 DAY > ? ORDER BY origin DESC LIMIT 1",
                [later["origin"], pd.Timestamp(week_start)],
            )
            if not previous:
                return ToolResult("not_found", action="No earlier run covers this week; nothing to compare.", source=["runs"])
            earlier_run_id = previous[0]["run_id"]
        a = self.explain_week_hours(store, department, week_start, earlier_run_id)
        b = self.explain_week_hours(store, department, week_start, later["run_id"])
        for result in (a, b):
            if result.status != "ok":
                return result
        before = {p["group"]: p["hours"] for p in a.data[0]["contributions"]}
        after = {p["group"]: p["hours"] for p in b.data[0]["contributions"]}
        changes = sorted(
            ({"group": g, "earlier_hours": before.get(g, 0.0), "later_hours": after.get(g, 0.0), "change_hours": after.get(g, 0.0) - before.get(g, 0.0)} for g in before.keys() | after.keys()),
            key=lambda r: -abs(r["change_hours"]),
        )
        summary = {
            "earlier_run_id": earlier_run_id,
            "later_run_id": later["run_id"],
            "earlier_total_hours": a.data[0]["totals"]["total_hours"],
            "later_total_hours": b.data[0]["totals"]["total_hours"],
            "change_total_hours": b.data[0]["totals"]["total_hours"] - a.data[0]["totals"]["total_hours"],
        }
        notes = [*b.notes, "A run made later has seen more recent data; changes in 'recent_level' usually reflect that."]
        return ToolResult("ok", data=[{"summary": summary, "changes": changes}], notes=notes, source=["weekly_hours", "weekly_hour_contributions", "runs"])

    def describe_feature(self, name: str) -> ToolResult:
        """Plain-English meaning of a feature or feature group."""
        rows = self._rows('SELECT feature, "group", description, known_in_advance FROM feature_dictionary WHERE feature = ? OR "group" = ?', [name, name])
        if rows:
            return ToolResult("ok", data=rows, source=["feature_dictionary"])
        return ToolResult(
            "not_found",
            data=self._rows('SELECT DISTINCT "group" FROM feature_dictionary ORDER BY 1'),
            action="Use a feature name or one of the listed groups.",
            source=["feature_dictionary"],
        )


TOOL_NAMES = (
    "list_runs",
    "find_entities",
    "get_provenance",
    "explain_week_hours",
    "explain_day_volume",
    "get_hours_breakdown",
    "compare_to_baseline",
    "get_accuracy",
    "get_caveats",
    "compare_runs",
    "describe_feature",
)
