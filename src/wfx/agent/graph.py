"""The explainability agent as a LangGraph state machine.

    plan (LLM, structured) ─┬─ staffing decision / off-topic ──────────────▶ decline
                            ├─ missing facts ──────────────────────────────▶ clarify
                            └─ gather (tools, chosen in code) ─┬─ guard hit ─▶ guided reply
                                                               └─ explain (LLM) ─▶ gate
    gate: pass ─▶ answer · fail ─▶ explain again with the violations · fail twice ─▶ safe fallback

The LLM never picks tools and never produces a number the gate can't trace to evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from wfx.agent import prompts
from wfx.agent.faithfulness import check
from wfx.agent.llm import LLM
from wfx.agent.routing import clarification, missing_facts, tool_calls
from wfx.agent.schema import QuestionPlan, QuestionType
from wfx.explain.tools import EvidenceTools, json_default

MAX_EXPLAIN_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    plan: QuestionPlan
    evidence: list[dict[str, Any]]
    answer: str
    outcome: str  # answered | clarification | guarded | declined | fallback
    attempts: int
    unsupported: list[str]


@dataclass
class AgentAnswer:
    answer: str
    outcome: str
    plan: QuestionPlan | None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    unsupported: list[str] = field(default_factory=list)


class ExplainabilityAgent:
    def __init__(self, tools: EvidenceTools, llm: LLM) -> None:
        self._tools = tools
        self._llm = llm
        self._graph = self._build()

    def ask(self, question: str) -> AgentAnswer:
        state = self._graph.invoke({"question": question, "attempts": 0, "evidence": []})
        return AgentAnswer(
            answer=state["answer"],
            outcome=state["outcome"],
            plan=state.get("plan"),
            evidence=state.get("evidence", []),
            attempts=state.get("attempts", 0),
            unsupported=state.get("unsupported", []),
        )

    # ---------------------------------------------------------------- graph

    def _build(self):  # noqa: ANN202 - compiled LangGraph type
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("decline", self._decline)
        graph.add_node("clarify", self._clarify)
        graph.add_node("gather", self._gather)
        graph.add_node("guarded", self._guarded)
        graph.add_node("explain", self._explain)
        graph.add_node("gate", self._gate)
        graph.add_node("fallback", self._fallback)

        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._after_plan, ["decline", "clarify", "gather"])
        graph.add_conditional_edges("gather", self._after_gather, ["guarded", "explain"])
        graph.add_edge("explain", "gate")
        graph.add_conditional_edges("gate", self._after_gate, ["explain", "fallback", END])
        for terminal in ("decline", "clarify", "guarded", "fallback"):
            graph.add_edge(terminal, END)
        return graph.compile()

    # ---------------------------------------------------------------- nodes

    def _plan(self, state: AgentState) -> AgentState:
        runs = self._tools.list_runs().data
        live = next(r for r in runs if r["is_live"])
        context = (
            f"Run context: the current (live) forecast run starts on {json_default(live['origin'])} (a Monday) "
            f"and covers {live['horizon_days']} days. Earlier runs: "
            + ", ".join(r["run_id"] for r in runs if not r["is_live"])
        )
        plan = self._llm.structured(prompts.PLANNER, f"{context}\n\nQuestion: {state['question']}", QuestionPlan)
        return {"plan": plan}

    @staticmethod
    def _after_plan(state: AgentState) -> str:
        plan = state["plan"]
        if plan.question_type in (QuestionType.STAFFING_DECISION, QuestionType.OUT_OF_SCOPE):
            return "decline"
        return "clarify" if missing_facts(plan) else "gather"

    @staticmethod
    def _decline(state: AgentState) -> AgentState:
        text = prompts.STAFFING_DECLINE if state["plan"].question_type == QuestionType.STAFFING_DECISION else prompts.OUT_OF_SCOPE
        return {"answer": text, "outcome": "declined"}

    @staticmethod
    def _clarify(state: AgentState) -> AgentState:
        return {"answer": clarification(state["plan"]), "outcome": "clarification"}

    def _gather(self, state: AgentState) -> AgentState:
        evidence = []
        for call in tool_calls(state["plan"]):
            result = getattr(self._tools, call.tool)(**call.args)
            evidence.append({"tool": call.tool, "args": call.args, "primary": call.primary, **asdict(result)})
        return {"evidence": evidence}

    @staticmethod
    def _after_gather(state: AgentState) -> str:
        blocked = [e for e in state["evidence"] if e["primary"] and e["status"] in ("not_found", "ambiguous")]
        return "guarded" if blocked else "explain"

    @staticmethod
    def _guarded(state: AgentState) -> AgentState:
        hit = next(e for e in state["evidence"] if e["primary"] and e["status"] in ("not_found", "ambiguous"))
        options = [", ".join(v if isinstance(v, str) else str(json_default(v)) for v in row.values()) for row in hit["data"][:12]]
        text = hit["action"] or "I couldn't find that."
        if options:
            text += " Options: " + "; ".join(options) + "."
        return {"answer": text, "outcome": "guarded"}

    def _explain(self, state: AgentState) -> AgentState:
        evidence = [{k: e[k] for k in ("tool", "status", "data", "notes")} for e in state["evidence"]]
        prompt = f"Question: {state['question']}\n\nEvidence (JSON):\n{json.dumps(evidence, default=json_default)}"
        if state.get("unsupported"):
            prompt += (
                "\n\nYour previous draft was rejected. These values are not in the evidence: "
                + ", ".join(state["unsupported"])
                + ". Rewrite it using only numbers that appear in the evidence; do not compute new ones."
            )
        answer = self._llm.text(prompts.EXPLAINER, prompt)
        return {"answer": answer, "attempts": state.get("attempts", 0) + 1}

    def _gate(self, state: AgentState) -> AgentState:
        result = check(state["answer"], state["evidence"], state["question"])
        return {"unsupported": result.unsupported, "outcome": "answered" if result.passed else "rejected"}

    @staticmethod
    def _after_gate(state: AgentState) -> str:
        if state["outcome"] == "answered":
            return END
        return "explain" if state["attempts"] < MAX_EXPLAIN_ATTEMPTS else "fallback"

    @staticmethod
    def _fallback(state: AgentState) -> AgentState:
        """Facts straight from the evidence, formatted without an LLM (always passes the gate)."""
        lines = ["I couldn't produce a verified written explanation, so here are the facts from the forecast data:"]
        for e in state["evidence"]:
            if e["status"] not in ("ok", "partial"):
                continue
            for note in e["notes"]:
                lines.append(f"- {note}")
            for row in e["data"]:
                lines.extend(f"- {line}" for line in _flatten(row))
        return {"answer": "\n".join(lines), "outcome": "fallback"}


def _flatten(row: Any, prefix: str = "") -> list[str]:
    if isinstance(row, dict):
        out: list[str] = []
        for key, value in row.items():
            out.extend(_flatten(value, f"{prefix}{key} "))
        return out
    if isinstance(row, list):
        out = []
        for item in row:
            out.extend(_flatten(item, prefix))
        return out
    if isinstance(row, float):
        return [f"{prefix.strip()}: {row:.2f}"]
    return [f"{prefix.strip()}: {json_default(row) if row is not None else 'n/a'}"]
