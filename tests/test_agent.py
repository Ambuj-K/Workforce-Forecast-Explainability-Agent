"""Tests for the agent graph (scripted LLM) and the numeric faithfulness gate."""

from datetime import date

import pandas as pd
import pytest

from wfx.agent.faithfulness import check
from wfx.agent.graph import ExplainabilityAgent
from wfx.agent.llm import ScriptedLLM
from wfx.agent.prompts import OUT_OF_SCOPE, STAFFING_DECLINE
from wfx.agent.schema import QuestionPlan, QuestionType
from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import build_extract
from wfx.explain.store import build_database, connect_readonly
from wfx.explain.tools import EvidenceTools
from wfx.forecasting.model import ModelParams


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    ds = generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 12, 29), n_stores=4, seed=5))
    extract = build_extract(ds, recent_runs=2, scored_runs=2, params=ModelParams(max_iter=60))
    return EvidenceTools(connect_readonly(build_database(extract, tmp_path_factory.mktemp("a") / "agent.duckdb")))


@pytest.fixture(scope="module")
def week(tools):
    live = next(r for r in tools.list_runs().data if r["is_live"])
    return str((pd.Timestamp(live["origin"]) + pd.Timedelta(days=7)).date())


def why_hours(week: str, **overrides: str) -> QuestionPlan:
    fields = {"question_type": QuestionType.WHY_HOURS, "store": "S001", "department": "grocery", "week": week, "reason": "test"}
    return QuestionPlan(**{**fields, **overrides})


def faithful_answer(tools: EvidenceTools, week: str) -> str:
    data = tools.explain_week_hours("S001", "grocery", week).data[0]
    top = data["contributions"][1]
    return (
        f"The week of {week} needs {data['totals']['total_hours']:.1f} hours "
        f"({data['totals']['fixed_hours']:.1f} fixed + {data['totals']['variable_hours']:.1f} variable).\n"
        f"- The model attributes {top['hours']:.1f} hours to {top['group']}."
    )


# --------------------------------------------------------------------- routes


def test_answers_a_why_hours_question_first_time(tools, week):
    llm = ScriptedLLM(structured=[why_hours(week)], texts=[faithful_answer(tools, week)])
    result = ExplainabilityAgent(tools, llm).ask("Why does S001 grocery need these hours?")
    assert result.outcome == "answered" and result.attempts == 1
    assert [e["tool"] for e in result.evidence] == ["explain_week_hours", "get_caveats"]


def test_invented_number_is_rejected_then_corrected(tools, week):
    llm = ScriptedLLM(structured=[why_hours(week)], texts=["Promotions add 987.6 hours.", faithful_answer(tools, week)])
    result = ExplainabilityAgent(tools, llm).ask("Why?")
    assert result.outcome == "answered" and result.attempts == 2
    retry_prompt = llm.prompts[-1][1]
    assert "987.6" in retry_prompt and "rejected" in retry_prompt


def test_repeated_invention_falls_back_to_verified_facts(tools, week):
    llm = ScriptedLLM(structured=[why_hours(week)], texts=["It is 987.6 hours.", "Still 987.6 hours."])
    result = ExplainabilityAgent(tools, llm).ask("Why?")
    assert result.outcome == "fallback"
    assert "987.6" not in result.answer
    assert check(result.answer, result.evidence).passed  # the fallback itself is grounded


def test_missing_week_asks_instead_of_guessing(tools):
    llm = ScriptedLLM(structured=[why_hours(None)])
    result = ExplainabilityAgent(tools, llm).ask("Why does S001 grocery need these hours?")
    assert result.outcome == "clarification" and "which week" in result.answer
    assert result.evidence == []


def test_unknown_department_is_guided_without_calling_the_explainer(tools, week):
    llm = ScriptedLLM(structured=[why_hours(week, department="bakery")], texts=[])  # explainer must not run
    result = ExplainabilityAgent(tools, llm).ask("Why does S001 bakery need these hours?")
    assert result.outcome == "guarded"
    assert "grocery" in result.answer and "bakery" in result.answer
    assert "Ask the user" not in result.answer  # internal instructions never reach the user


def test_staffing_decision_is_declined_with_an_offer(tools):
    plan = QuestionPlan(question_type=QuestionType.STAFFING_DECISION, store="S001", reason="test")
    result = ExplainabilityAgent(tools, ScriptedLLM(structured=[plan])).ask("Should I cut two people from S001?")
    assert result.outcome == "declined" and result.answer == STAFFING_DECLINE and result.evidence == []


def test_off_topic_is_declined(tools):
    plan = QuestionPlan(question_type=QuestionType.OUT_OF_SCOPE, reason="test")
    result = ExplainabilityAgent(tools, ScriptedLLM(structured=[plan])).ask("What's the capital of France?")
    assert result.outcome == "declined" and result.answer == OUT_OF_SCOPE


def test_trust_question_routes_to_accuracy_and_caveats(tools):
    plan = QuestionPlan(question_type=QuestionType.TRUST, reason="test")
    acc = tools.get_accuracy().data[0]
    answer = f"One week ahead the error was {acc['wape_model'] * 100:.1f}% (baseline {acc['wape_baseline'] * 100:.1f}%)."
    result = ExplainabilityAgent(tools, ScriptedLLM(structured=[plan], texts=[answer])).ask("Can I trust it?")
    assert [e["tool"] for e in result.evidence] == ["get_accuracy", "get_caveats"]
    assert result.outcome == "answered"


def test_planner_gets_the_live_run_context(tools, week):
    llm = ScriptedLLM(structured=[why_hours(week)], texts=[faithful_answer(tools, week)])
    ExplainabilityAgent(tools, llm).ask("Why next week?")
    live = next(r for r in tools.list_runs().data if r["is_live"])
    assert str(pd.Timestamp(live["origin"]).date()) in llm.prompts[0][1]


# ---------------------------------------------------------------------- gate

EVIDENCE = {
    "totals": {"variable_hours": 202.7897, "fixed_hours": 15.0, "week_start": "2026-01-05"},
    "contributions": [{"group": "promotion", "hours": -6.7288}],
    "accuracy": {"wape_model": 0.06005},
    "notes": ["S002: latest actuals 2025-12-23, 6 days before the run."],
}


@pytest.mark.parametrize(
    "answer",
    [
        "Week of 2026-01-05: 202.8 variable hours plus 15 fixed.",
        "About 203 hours; promotions take away 6.7 hours.",
        "Error one week ahead was 6.0%.",
        "For the week of 5 January 2026, see above.",
        "Latest actuals 2025-12-23, 6 days before the run.",
    ],
)
def test_gate_accepts_grounded_answers(answer):
    assert check(answer, EVIDENCE).passed


@pytest.mark.parametrize(
    ("answer", "bad"),
    [
        ("Promotions add 12.4 hours.", "12.4"),
        ("In total 217.8 hours.", "217.8"),  # computed sum, not in this evidence
        ("For the week of 12 January 2026.", "12 January 2026"),
        ("Data through 2025-12-24.", "2025-12-24"),
    ],
)
def test_gate_rejects_invented_computed_or_wrong_values(answer, bad):
    result = check(answer, EVIDENCE)
    assert not result.passed and bad in result.unsupported


def test_gate_allows_numbers_from_the_question():
    assert check("For store 7 there is no data.", EVIDENCE, question="what about store 7?").passed
