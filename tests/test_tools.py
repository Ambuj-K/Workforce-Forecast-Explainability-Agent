"""Tests for the evidence tools: correct numbers, exact explanations, and every result guard."""

import json
from datetime import date

import pandas as pd
import pytest

from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import build_extract
from wfx.explain.store import build_database, connect_readonly
from wfx.explain.tools import TOOL_NAMES, EvidenceTools
from wfx.forecasting.model import ModelParams


@pytest.fixture(scope="module")
def ds():
    # Ends on a Sunday just after Christmas, so recent runs cover a non-trading day.
    return generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 12, 29), n_stores=4, seed=5))


@pytest.fixture(scope="module")
def tools(ds, tmp_path_factory):
    extract = build_extract(ds, recent_runs=4, scored_runs=2, params=ModelParams(max_iter=60))
    db = build_database(extract, tmp_path_factory.mktemp("agent") / "agent.duckdb")
    return EvidenceTools(connect_readonly(db))


@pytest.fixture(scope="module")
def live_week(tools):
    origin = pd.Timestamp(next(r for r in tools.list_runs().data if r["is_live"])["origin"])
    return origin + pd.Timedelta(days=7)  # week 2 of the live run, also covered by recent runs


def test_every_tool_exists_and_returns_json(tools):
    for name in TOOL_NAMES:
        assert callable(getattr(tools, name))
    json.loads(tools.list_runs().to_json())


# ------------------------------------------------------------------ resolution


@pytest.mark.parametrize(("text", "expected"), [("S001", "S001"), ("s001", "S001"), ("store 1", "S001"), ("1", "S001")])
def test_store_names_resolve(tools, live_week, text, expected):
    result = tools.explain_week_hours(text, "grocery", live_week)
    assert result.status == "ok"
    assert result.data[0]["totals"]["store_id"] == expected


def test_fuzzy_match_is_disclosed(tools, live_week):
    result = tools.explain_week_hours("store 1", "Grocery", live_week)
    assert any("Interpreted store 'store 1' as 'S001'" in n for n in result.notes)


def test_ambiguous_name_asks_instead_of_guessing(tools, live_week):
    result = tools.explain_day_volume("S001", "s", live_week)
    assert result.status == "ambiguous" and len(result.data) > 1
    assert "Ask the user" in result.action


def test_unknown_name_lists_valid_options(tools, live_week):
    result = tools.explain_week_hours("S001", "bakery", live_week)
    assert result.status == "not_found"
    assert {"department": "grocery"} in result.data


def test_injection_attempt_in_a_name_is_just_not_found(tools, live_week):
    for junk in ("S001' OR 1=1 --", "S001; DROP TABLE runs", "store 1 or anything"):
        result = tools.explain_week_hours(junk, "grocery", live_week)
        assert result.status == "not_found"  # parameterised SQL + strict store-name shape


def test_unknown_run_lists_runs(tools):
    result = tools.get_provenance("run-1999-01-01")
    assert result.status == "not_found" and len(result.data) > 1


def test_week_outside_the_run_lists_available_weeks(tools):
    result = tools.explain_week_hours("S001", "grocery", "2023-03-06")
    assert result.status == "not_found"
    assert len(result.data) == 4


def test_non_monday_is_normalised_with_a_note(tools, live_week):
    result = tools.explain_week_hours("S001", "grocery", live_week + pd.Timedelta(days=2))
    assert result.status == "ok"
    assert any("not a Monday" in n for n in result.notes)


# ------------------------------------------------------------------- numbers


def test_week_explanation_adds_up_exactly(tools, live_week):
    result = tools.explain_week_hours("S001", "fresh", live_week)
    assert result.status == "ok"
    data = result.data[0]
    total = sum(part["hours"] for part in data["contributions"])
    assert total == pytest.approx(data["totals"]["variable_hours"], abs=1e-6)
    assert data["totals"]["fixed_hours"] + data["totals"]["variable_hours"] == pytest.approx(data["totals"]["total_hours"])


def test_day_explanation_adds_up_exactly(tools, live_week):
    data = tools.explain_day_volume("S003", "transactions", live_week).data[0]
    total = data["forecast"]["base_value"] + sum(p["contribution"] for p in data["contributions"])
    assert total == pytest.approx(data["forecast"]["forecast"], abs=1e-6)


def test_non_trading_day_is_explained_as_a_rule(tools):
    runs = [r for r in tools.list_runs().data if pd.Timestamp(r["origin"]) <= pd.Timestamp("2024-12-25") < pd.Timestamp(r["origin"]) + pd.Timedelta(days=28)]
    result = tools.explain_day_volume("S001", "transactions", "2024-12-25", run_id=runs[-1]["run_id"])
    assert result.data[0]["forecast"]["forecast"] == 0
    assert result.data[0]["contributions"] == []
    assert any("0 by rule" in n for n in result.notes)


def test_hours_breakdown_matches_the_weekly_total(tools, live_week):
    data = tools.get_hours_breakdown("S001", "checkouts", live_week).data[0]
    assert sum(d["variable_hours"] for d in data["drivers"]) == pytest.approx(data["totals"]["variable_hours"], abs=1e-6)


def test_compare_runs_changes_add_up_to_the_total_change(tools, live_week):
    data = tools.compare_runs("S001", "grocery", live_week).data[0]
    assert data["summary"]["earlier_run_id"] != data["summary"]["later_run_id"]
    assert sum(c["change_hours"] for c in data["changes"]) == pytest.approx(data["summary"]["change_total_hours"], abs=1e-6)


def test_baseline_comparison_is_consistent(tools, live_week):
    data = tools.compare_to_baseline("S001", "grocery", live_week).data[0]
    assert data["difference_hours"] == pytest.approx(data["model_variable_hours"] - data["baseline_variable_hours"])


# --------------------------------------------------------------------- trust


def test_accuracy_only_uses_windows_before_the_run(tools):
    live = tools.get_accuracy()
    assert live.status == "ok" and {r["lead_week"] for r in live.data} == {1, 2, 3, 4}
    oldest = tools.list_runs().data[0]["run_id"]
    assert tools.get_accuracy(run_id=oldest).status == "not_found"  # nothing had played out yet


def test_caveats_report_the_stale_store_with_its_dates(tools, ds):
    late = ds.caveats[ds.caveats["caveat_type"] == "late_feed"].iloc[0]  # answer key, test only
    result = tools.get_caveats()
    stale = [n for n in result.notes if "data is stale" in n]
    assert stale and all(late["store_id"] in n for n in stale)
    assert any(str((late["start_date"] - pd.Timedelta(days=1)).date()) in n for n in stale)


def test_describe_feature_by_name_or_group(tools):
    assert tools.describe_feature("lag_364").data[0]["group"] == "seasonal_history"
    assert len(tools.describe_feature("promotion").data) == 1
    assert tools.describe_feature("weather").status == "not_found"
