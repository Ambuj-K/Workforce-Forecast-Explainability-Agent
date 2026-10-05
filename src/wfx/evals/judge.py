"""LLM judge for what code can't check: clarity and framing. Deterministic checks come first.

Caveat: judging with the same model family that wrote the answer can be lenient; scores are a
quality signal, not proof. Every verdict is stored with its reasoning for inspection.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from wfx.agent.llm import LLM
from wfx.explain.tools import json_default

JUDGE_PROMPT = """You grade answers from an assistant that explains labour-demand forecasts to a
retail workforce planner (busy, commercially sharp, not a data scientist).

For each criterion, decide strictly whether the answer meets it. Judge only the answer text
against the criterion, using the evidence to check facts. Return one boolean per criterion,
in the same order, and a short reasoning that names any criterion that failed and why.
"""

DEFAULT_CRITERIA = [
    "Uses plain English a non-technical planner would understand (no unexplained jargon)",
    "Does not recommend staffing decisions",
]


class JudgeVerdict(BaseModel):
    criteria_met: list[bool] = Field(description="One boolean per criterion, in order.")
    reasoning: str = Field(description="Short; name any failed criterion and why.")


def judge(llm: LLM, question: str, answer: str, evidence: list[dict], criteria: list[str]) -> tuple[float, JudgeVerdict]:
    all_criteria = [*criteria, *DEFAULT_CRITERIA]
    compact = [{k: e[k] for k in ("tool", "status", "data", "notes")} for e in evidence]
    prompt = (
        f"Question: {question}\n\nAnswer:\n{answer}\n\nEvidence (JSON):\n{json.dumps(compact, default=json_default)}\n\n"
        "Criteria:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(all_criteria))
    )
    verdict = llm.structured(JUDGE_PROMPT, prompt, JudgeVerdict)
    met = verdict.criteria_met[: len(all_criteria)]
    if len(met) != len(all_criteria):
        return 0.0, JudgeVerdict(criteria_met=met, reasoning=f"judge returned {len(met)} verdicts for {len(all_criteria)} criteria")
    return sum(met) / len(met), verdict
