"""Agent monitoring: every question is logged; health metrics and alerts come from the log.

The log keeps what is needed to diagnose behaviour (outcome, gate attempts, rejected values,
tools, latency). Questions are stored with emails, phone numbers and long digit strings
masked; answers are not stored, only their length.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wfx.agent.graph import AgentAnswer
from wfx.monitoring.forecast import Alert

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def mask(text: str) -> str:
    """Mask emails and phone-like numbers before anything is written to disk."""
    return _PHONE.sub("[number]", _EMAIL.sub("[email]", text))


@dataclass(frozen=True)
class Interaction:
    timestamp: str
    question: str
    outcome: str
    question_type: str | None
    attempts: int
    rejected_values: list[str]
    tools: list[str]
    latency_ms: int
    answer_chars: int

    @classmethod
    def from_answer(cls, question: str, answer: AgentAnswer, latency_ms: int) -> Interaction:
        return cls(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            question=mask(question),
            outcome=answer.outcome,
            question_type=answer.plan.question_type.value if answer.plan else None,
            attempts=answer.attempts,
            rejected_values=[v for r in answer.rejected for v in r["unsupported"]],
            tools=[e["tool"] for e in answer.evidence],
            latency_ms=latency_ms,
            answer_chars=len(answer.answer),
        )


class InteractionLog:
    """Append-only JSON-lines log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, interaction: Interaction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(asdict(interaction)) + "\n")

    def read(self, last: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()
        return [json.loads(line) for line in (lines[-last:] if last else lines)]


@dataclass(frozen=True)
class AgentRules:
    window: int = 200  # most recent interactions considered
    min_first_pass_rate: float = 0.9  # of explained answers
    max_fallback_rate: float = 0.05  # of explained answers
    max_latency_p95_ms: int = 15_000
    min_sample: int = 10  # don't alert on tiny samples


@dataclass
class AgentHealth:
    interactions: int
    by_outcome: dict[str, int]
    explained: int
    first_pass_rate: float | None
    fallback_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    top_rejected_values: list[tuple[str, int]] = field(default_factory=list)


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))])


def agent_health(records: list[dict[str, Any]]) -> AgentHealth:
    by_outcome: dict[str, int] = {}
    for r in records:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    explained = [r for r in records if r["outcome"] in ("answered", "fallback")]
    rejected: dict[str, int] = {}
    for r in records:
        for value in r["rejected_values"]:
            rejected[value] = rejected.get(value, 0) + 1
    latencies = [r["latency_ms"] for r in records]
    return AgentHealth(
        interactions=len(records),
        by_outcome=by_outcome,
        explained=len(explained),
        first_pass_rate=sum(r["outcome"] == "answered" and r["attempts"] == 1 for r in explained) / len(explained) if explained else None,
        fallback_rate=sum(r["outcome"] == "fallback" for r in explained) / len(explained) if explained else None,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        top_rejected_values=sorted(rejected.items(), key=lambda kv: -kv[1])[:5],
    )


def agent_alerts(health: AgentHealth, rules: AgentRules | None = None) -> list[Alert]:
    rules = rules or AgentRules()
    if health.explained < rules.min_sample:
        return []
    alerts = []
    if health.first_pass_rate is not None and health.first_pass_rate < rules.min_first_pass_rate:
        alerts.append(Alert(
            "gate_first_pass_low", "warning",
            f"Only {health.first_pass_rate:.0%} of explanations passed the faithfulness gate first time "
            f"(target {rules.min_first_pass_rate:.0%}). Most rejected values: {health.top_rejected_values}.",
        ))
    if health.fallback_rate is not None and health.fallback_rate > rules.max_fallback_rate:
        alerts.append(Alert(
            "fallback_rate_high", "critical",
            f"{health.fallback_rate:.0%} of explanations fell back to templated facts (limit {rules.max_fallback_rate:.0%}).",
        ))
    if health.latency_p95_ms is not None and health.latency_p95_ms > rules.max_latency_p95_ms:
        alerts.append(Alert(
            "latency_high", "warning",
            f"95th percentile latency {health.latency_p95_ms / 1000:.1f}s (limit {rules.max_latency_p95_ms / 1000:.0f}s).",
        ))
    return alerts


def eval_trend(results_dir: Path) -> list[dict[str, Any]]:
    """One row per saved eval run: pass rate, first-pass rate, judge average (oldest first)."""
    rows = []
    for path in sorted(results_dir.glob("results-*.json")):
        results = json.loads(path.read_text())
        if len(results) < 10:  # skip targeted re-runs of a few goldens
            continue
        explained = [r for r in results if r["outcome"] in ("answered", "fallback")]
        judged = [r["judge_score"] for r in results if r.get("judge_score") is not None]
        rows.append({
            "run": path.stem.removeprefix("results-"),
            "goldens": len(results),
            "pass_rate": sum(r["passed"] for r in results) / len(results),
            "first_pass_rate": sum(r["attempts"] == 1 for r in explained) / len(explained) if explained else None,
            "avg_judge": sum(judged) / len(judged) if judged else None,
        })
    return rows
