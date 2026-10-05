"""Run the golden questions through the agent, score them, and compare with a saved baseline.

Every result keeps the artifact under test (the answer and its evidence), not just scores,
so a failing check can be inspected without re-running anything.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wfx.agent.graph import ExplainabilityAgent
from wfx.agent.llm import LLM
from wfx.evals.goldens import Golden
from wfx.evals.judge import judge
from wfx.evals.metrics import passed, score
from wfx.explain.tools import json_default

JUDGE_PASS = 0.7
REGRESSION_DROP = 0.1


@dataclass
class GoldenResult:
    id: str
    category: str
    question: str
    answer: str
    outcome: str
    question_type: str | None
    tools: list[str]
    attempts: int
    unsupported: list[str]
    checks: dict[str, bool | None]
    judge_score: float | None
    judge_reasoning: str | None
    passed: bool
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_drafts: list[dict[str, Any]] = field(default_factory=list)


def run_goldens(agent: ExplainabilityAgent, judge_llm: LLM | None, goldens: list[Golden], pause_seconds: float = 0.0) -> list[GoldenResult]:
    results = []
    for golden in goldens:
        answer = agent.ask(golden.question)
        checks = score(golden, answer)
        judge_score, reasoning = None, None
        if judge_llm is not None and golden.judge and answer.outcome == "answered":
            judge_score, verdict = judge(judge_llm, golden.question, answer.answer, answer.evidence, golden.judge)
            reasoning = verdict.reasoning
        ok = passed(checks) and (judge_score is None or judge_score >= JUDGE_PASS)
        results.append(
            GoldenResult(
                id=golden.id,
                category=golden.category,
                question=golden.question,
                answer=answer.answer,
                outcome=answer.outcome,
                question_type=answer.plan.question_type.value if answer.plan else None,
                tools=[e["tool"] for e in answer.evidence],
                attempts=answer.attempts,
                unsupported=answer.unsupported,
                checks=checks,
                judge_score=judge_score,
                judge_reasoning=reasoning,
                passed=ok,
                evidence=[{k: e[k] for k in ("tool", "args", "status", "notes")} for e in answer.evidence],
                rejected_drafts=answer.rejected,
            )
        )
        if pause_seconds:
            time.sleep(pause_seconds)
    return results


def summarize(results: list[GoldenResult]) -> dict[str, Any]:
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_check: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r.passed)
        for name, value in r.checks.items():
            if value is not None:
                by_check[name].append(value)
    judged = [r.judge_score for r in results if r.judge_score is not None]
    return {
        "goldens": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": sum(r.passed for r in results) / len(results),
        "first_pass_rate": _rate(by_check.get("first_pass", [])),
        "avg_judge": sum(judged) / len(judged) if judged else None,
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_category.items())},
        "by_check": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_check.items())},
        "failed": [r.id for r in results if not r.passed],
    }


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def compare(results: list[GoldenResult], baseline: list[dict[str, Any]]) -> list[str]:
    """Regressions vs a saved baseline: pass -> fail, or judge score dropping by more than 0.1."""
    previous = {b["id"]: b for b in baseline}
    findings = []
    for r in results:
        b = previous.get(r.id)
        if b is None:
            continue
        if b["passed"] and not r.passed:
            findings.append(f"{r.id}: passed -> failed")
        if b.get("judge_score") is not None and r.judge_score is not None and b["judge_score"] - r.judge_score > REGRESSION_DROP:
            findings.append(f"{r.id}: judge {b['judge_score']:.2f} -> {r.judge_score:.2f}")
    return findings


def save(results: list[GoldenResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=json_default))


def load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())
