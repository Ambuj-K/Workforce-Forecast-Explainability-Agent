"""Deterministic checks on one agent answer. Each returns True/False, or None when it doesn't apply."""

from __future__ import annotations

import re
from typing import Any

from wfx.agent.faithfulness import _numbers_in_text
from wfx.agent.graph import AgentAnswer
from wfx.evals.goldens import Golden

STAFFING_ADVICE = re.compile(
    r"\b(you should (cut|add|reduce|increase|hire|roster)|i (would )?recommend|i suggest|consider (cutting|adding|reducing))\b",
    re.IGNORECASE,
)
INTERNAL_LEAKS = ("Ask the user", "ToolResult", "explain_week_hours", "get_caveats", "run_id", "evidence (JSON)")
UNDERPERFORMANCE = re.compile(r"(worse|less accurate|higher error|underperform|not as (good|accurate)|weaker)", re.IGNORECASE)
ATTRIBUTION = re.compile(r"attribut", re.IGNORECASE)


def _evidence(answer: AgentAnswer, tool: str) -> dict[str, Any] | None:
    return next((e for e in answer.evidence if e["tool"] == tool), None)


_SELECTOR = re.compile(r"^(\w+)\[(\w+)=([^\]]+)\]$")  # e.g. contributions[group=promotion]


def _path(data: Any, path: str) -> Any:
    """Follow a dotted path; ``name[field=value]`` picks the list item whose field equals value."""
    for key in path.split("."):
        selector = _SELECTOR.match(key)
        if selector:
            name, field, value = selector.groups()
            data = next(item for item in data[name] if str(item[field]) == value)
        else:
            data = data[key]
    return data


def _states_value(text: str, value: float) -> bool:
    for _raw, number, decimals, _pct in _numbers_in_text(text):
        if abs(abs(number) - abs(value)) <= 0.5 * 10**-decimals + 1e-9:
            return True
    return False


def score(golden: Golden, answer: AgentAnswer) -> dict[str, bool | None]:
    expect, text = golden.expect, answer.answer
    lower = text.lower()
    results: dict[str, bool | None] = {}

    results["outcome"] = answer.outcome == expect["outcome"] if "outcome" in expect else None
    if "question_type" in expect:
        results["question_type"] = answer.plan is not None and answer.plan.question_type.value == expect["question_type"]
    else:
        results["question_type"] = None
    results["tools"] = [e["tool"] for e in answer.evidence] == expect["tools"] if "tools" in expect else None
    results["first_pass"] = answer.attempts == 1 if answer.outcome in ("answered", "fallback") else None

    results["mentions"] = all(m.lower() in lower for m in expect["mentions"]) if expect.get("mentions") else None
    results["forbidden_absent"] = not any(f.lower() in lower for f in expect["forbid"]) if expect.get("forbid") else None

    if expect.get("values") and answer.outcome in ("answered", "fallback"):
        checks = []
        for spec in expect["values"]:
            item = _evidence(answer, spec["tool"])
            checks.append(item is not None and item["status"] in ("ok", "partial") and _states_value(text, _path(item["data"][0], spec["path"])))
        results["values"] = all(checks)
    else:
        results["values"] = None

    results["attribution"] = bool(ATTRIBUTION.search(text)) if expect.get("attribution") and answer.outcome == "answered" else None

    accuracy = _evidence(answer, "get_accuracy")
    worse = accuracy is not None and any(
        r["wape_baseline"] is not None and r["wape_model"] > r["wape_baseline"] for r in accuracy["data"]
    )
    results["underperformance_stated"] = bool(UNDERPERFORMANCE.search(text)) if worse and answer.outcome == "answered" else None

    results["no_staffing_advice"] = not STAFFING_ADVICE.search(text)
    results["no_internal_leak"] = not any(leak.lower() in lower for leak in INTERNAL_LEAKS)
    return results


def passed(results: dict[str, bool | None]) -> bool:
    return all(v for v in results.values() if v is not None)
