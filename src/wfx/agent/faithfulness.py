"""Numeric faithfulness gate: every number in an answer must come from the evidence.

The answer may round (171.89 -> 171.9 or 172) and show fractions as percentages
(0.0601 -> 6.0%), but it may not compute new numbers (sums, differences, ratios) or
invent any. Tools provide derived numbers when an answer needs them.

Dates are checked as whole dates; numbers that appear in the user's question are allowed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
NUMBER = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_MONTHS = {
    m: i
    for i, names in enumerate(
        [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
         ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
         ("dec", "december")],
        start=1,
    )
    for m in names
}
WORDED_DATE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b\.?(?:,?\s+(\d{4}))?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    unsupported: list[str]  # numbers or dates in the answer with no source in the evidence

    def feedback(self) -> str:
        return (
            "These values are not in the evidence: "
            + ", ".join(self.unsupported)
            + ". Use only numbers that appear in the evidence (rounding is fine); do not add, subtract or compute new ones."
        )


def _leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _leaves(v)
    else:
        yield value


def _numbers_in_text(text: str) -> list[tuple[str, float, int, bool]]:
    """(raw, value, decimals, is_percent) for each number in ``text``."""
    found = []
    for match in NUMBER.finditer(text):
        raw = match.group(0)
        clean = raw.replace(",", "").replace("−", "-").rstrip("%")
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        found.append((raw, float(clean), decimals, raw.endswith("%")))
    return found


def evidence_index(evidence: Any) -> tuple[set[float], set[str]]:
    """All numeric values and ISO dates in the evidence (including inside strings such as notes)."""
    numbers: set[float] = set()
    dates: set[str] = set()
    for leaf in _leaves(evidence):
        if isinstance(leaf, bool) or leaf is None:
            continue
        if isinstance(leaf, (int, float)):
            numbers.add(float(leaf))
            continue
        text = str(leaf)
        for y, m, d in ISO_DATE.findall(text):
            dates.add(f"{y}-{m}-{d}")
        for _raw, value, _dec, _pct in _numbers_in_text(ISO_DATE.sub(" ", text)):
            numbers.add(value)
    return numbers, dates


def _supported(value: float, decimals: int, is_percent: bool, numbers: set[float]) -> bool:
    tolerance = 0.5 * 10**-decimals + 1e-9
    for source in numbers:
        candidates = (source, source * 100) if is_percent or abs(source) <= 1 else (source,)
        for candidate in candidates:
            if abs(abs(candidate) - abs(value)) <= tolerance:
                return True
    return False


def check(answer: str, evidence: Any, question: str = "") -> GateResult:
    """Check that every number and date in ``answer`` is grounded in ``evidence`` (or the question)."""
    numbers, dates = evidence_index(evidence)
    q_numbers = {v for _r, v, _d, _p in _numbers_in_text(question)}
    evidence_days = {(int(d[:4]), int(d[5:7]), int(d[8:])) for d in dates}
    unsupported: list[str] = []

    for y, m, d in ISO_DATE.findall(answer):
        if f"{y}-{m}-{d}" not in dates and f"{y}-{m}-{d}" not in question:
            unsupported.append(f"{y}-{m}-{d}")
    text = ISO_DATE.sub(" ", answer)

    for match in WORDED_DATE.finditer(text):
        day, month = int(match.group(1)), _MONTHS[match.group(2).lower()]
        year = int(match.group(3)) if match.group(3) else None
        if not any(dd == day and mm == month and (year is None or yy == year) for yy, mm, dd in evidence_days):
            unsupported.append(match.group(0).strip())
    text = WORDED_DATE.sub(" ", text)

    for raw, value, decimals, is_percent in _numbers_in_text(text):
        if value in q_numbers or _supported(value, decimals, is_percent, numbers):
            continue
        unsupported.append(raw)
    return GateResult(passed=not unsupported, unsupported=unsupported)
