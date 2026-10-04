"""Deterministic routing: which evidence tools answer which question type.

The LLM never chooses tools. Each question type maps to a fixed list of tool calls, and
each type declares the facts it needs; if one is missing, the agent asks for it instead
of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wfx.agent.schema import QuestionPlan, QuestionType


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    primary: bool = True  # a non-ok primary result stops the answer; secondary ones are context


@dataclass(frozen=True)
class Route:
    required: tuple[str, ...]
    calls: tuple[tuple[str, tuple[str, ...], bool], ...]  # (tool, plan fields -> args, primary)


ROUTES: dict[QuestionType, Route] = {
    QuestionType.WHY_HOURS: Route(
        ("store", "department", "week"),
        (("explain_week_hours", ("store", "department", "week", "run_id"), True), ("get_caveats", ("store", "run_id"), False)),
    ),
    QuestionType.WHY_VOLUME: Route(
        ("store", "driver", "day"),
        (("explain_day_volume", ("store", "driver", "day", "run_id"), True), ("get_caveats", ("store", "run_id"), False)),
    ),
    QuestionType.HOURS_BREAKDOWN: Route(
        ("store", "department", "week"),
        (("get_hours_breakdown", ("store", "department", "week", "run_id"), True),),
    ),
    QuestionType.CHANGE: Route(
        ("store", "department", "week"),
        (("compare_runs", ("store", "department", "week"), True), ("get_caveats", ("store",), False)),
    ),
    QuestionType.BASELINE: Route(
        ("store", "department", "week"),
        (("compare_to_baseline", ("store", "department", "week", "run_id"), True),),
    ),
    QuestionType.TRUST: Route(
        (),
        (("get_accuracy", ("store", "driver", "run_id"), True), ("get_caveats", ("store", "run_id"), False)),
    ),
    QuestionType.CAVEATS: Route((), (("get_caveats", ("store", "run_id"), True),)),
    QuestionType.DEFINITION: Route(("term",), (("describe_feature", ("term",), True),)),
    QuestionType.RUNS: Route((), (("list_runs", (), True), ("get_provenance", ("run_id",), False))),
}

# How plan fields map onto tool parameter names.
_ARG_NAMES = {
    "explain_week_hours": {"week": "week_start"},
    "get_hours_breakdown": {"week": "week_start"},
    "compare_to_baseline": {"week": "week_start"},
    "compare_runs": {"week": "week_start"},
    "explain_day_volume": {"day": "day"},
    "describe_feature": {"term": "name"},
}

_FIELD_WORDS = {"store": "which store", "department": "which department", "driver": "which workload driver",
                "week": "which week (any date in it)", "day": "which day", "term": "which term to explain"}


def missing_facts(plan: QuestionPlan) -> list[str]:
    route = ROUTES.get(plan.question_type)
    if route is None:
        return []
    return [f for f in route.required if getattr(plan, f) in (None, "")]


def clarification(plan: QuestionPlan) -> str:
    asks = [_FIELD_WORDS[f] for f in missing_facts(plan)]
    return "To answer that I need to know " + ", ".join(asks) + "."


def tool_calls(plan: QuestionPlan) -> list[ToolCall]:
    route = ROUTES[plan.question_type]
    calls = []
    for tool, plan_fields, primary in route.calls:
        names = _ARG_NAMES.get(tool, {})
        args = {names.get(f, f): getattr(plan, f) for f in plan_fields if getattr(plan, f) not in (None, "")}
        calls.append(ToolCall(tool, args, primary))
    return calls
