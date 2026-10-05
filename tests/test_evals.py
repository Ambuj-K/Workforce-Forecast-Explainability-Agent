"""Tests for the eval harness: goldens load and resolve, metrics score correctly, regressions are caught."""

from datetime import date

import pytest

from wfx.agent.graph import AgentAnswer
from wfx.agent.llm import ScriptedLLM
from wfx.agent.schema import QuestionPlan, QuestionType
from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.evals.goldens import Golden, load_goldens
from wfx.evals.judge import JudgeVerdict, judge
from wfx.evals.metrics import passed, score
from wfx.evals.runner import GoldenResult, compare


def answer(text: str, outcome: str = "answered", qtype: QuestionType | None = QuestionType.WHY_HOURS, evidence=None, attempts=1):
    plan = QuestionPlan(question_type=qtype, reason="t") if qtype else None
    return AgentAnswer(answer=text, outcome=outcome, plan=plan, evidence=evidence or [], attempts=attempts)


WEEK_EVIDENCE = [
    {"tool": "explain_week_hours", "args": {}, "primary": True, "status": "ok",
     "data": [{"totals": {"total_hours": 217.79}, "contributions": []}], "notes": []},
    {"tool": "get_caveats", "args": {}, "primary": False, "status": "ok", "data": [], "notes": []},
]


def test_goldens_load_and_placeholders_resolve():
    goldens = load_goldens(generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 12, 29), seed=5)))
    assert len(goldens) >= 20
    assert all("{" not in g.question for g in goldens)
    assert {g.category for g in goldens} >= {"why_hours", "trust", "caveats", "staffing", "guard", "premise", "injection"}


def test_grounded_answer_passes_every_check():
    golden = Golden("g", "q", "why_hours", expect={
        "outcome": "answered", "question_type": "why_hours", "tools": ["explain_week_hours", "get_caveats"],
        "values": [{"tool": "explain_week_hours", "path": "totals.total_hours"}], "attribution": True,
    })
    result = score(golden, answer("The week needs 217.8 hours; the model attributes most to the base level.", evidence=WEEK_EVIDENCE))
    assert passed(result)


def test_missing_value_and_framing_fail():
    golden = Golden("g", "q", "why_hours", expect={
        "outcome": "answered", "values": [{"tool": "explain_week_hours", "path": "totals.total_hours"}], "attribution": True,
    })
    result = score(golden, answer("Promotions drive the hours.", evidence=WEEK_EVIDENCE))
    assert result["values"] is False and result["attribution"] is False


def test_staffing_advice_and_internal_leaks_are_caught():
    golden = Golden("g", "q", "why_hours", expect={"outcome": "answered"})
    assert score(golden, answer("You should cut two people."))["no_staffing_advice"] is False
    assert score(golden, answer("Ask the user which store."))["no_internal_leak"] is False


def test_underperformance_must_be_stated_when_evidence_shows_it():
    evidence = [{"tool": "get_accuracy", "args": {}, "primary": True, "status": "ok", "notes": [],
                 "data": [{"lead_week": 1, "wape_model": 0.06, "wape_baseline": 0.049}]}]
    golden = Golden("g", "q", "trust", expect={"outcome": "answered"})
    assert score(golden, answer("Error was 6.0%.", evidence=evidence))["underperformance_stated"] is False
    assert score(golden, answer("One week ahead it did worse than the simple method.", evidence=evidence))["underperformance_stated"] is True


def test_forbidden_strings_catch_injection():
    golden = Golden("g", "q", "injection", expect={"forbid": ["999"]})
    assert score(golden, answer("It needs 999 hours.", outcome="answered"))["forbidden_absent"] is False


def test_judge_scores_fraction_of_criteria_met():
    llm = ScriptedLLM(structured=[JudgeVerdict(criteria_met=[True, False, True], reasoning="criterion 2 failed")])
    value, verdict = judge(llm, "q", "a", [], ["clear"])
    assert value == pytest.approx(2 / 3) and "criterion 2" in verdict.reasoning


def test_regression_detection():
    def r(passed_: bool, judge_score: float | None) -> GoldenResult:
        return GoldenResult("g", "c", "q", "a", "answered", "why_hours", [], 1, [], {}, judge_score, None, passed_)
    baseline = [{"id": "g", "passed": True, "judge_score": 1.0}]
    assert compare([r(False, 1.0)], baseline) == ["g: passed -> failed"]
    assert compare([r(True, 0.8)], baseline) == ["g: judge 1.00 -> 0.80"]
    assert compare([r(True, 0.95)], baseline) == []


def test_value_path_can_select_a_list_item():
    evidence = [{"tool": "explain_week_hours", "args": {}, "primary": True, "status": "ok", "notes": [],
                 "data": [{"contributions": [{"group": "base", "hours": 171.9}, {"group": "promotion", "hours": -6.73}]}]}]
    golden = Golden("g", "q", "premise", expect={"values": [{"tool": "explain_week_hours", "path": "contributions[group=promotion].hours"}]})
    assert score(golden, answer("The model attributes -6.7 hours to the promotion.", evidence=evidence))["values"] is True
    assert score(golden, answer("Promotions add 80 hours.", evidence=evidence))["values"] is False

