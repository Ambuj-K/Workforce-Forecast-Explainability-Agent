"""Load golden questions and resolve their placeholders from the answer key and the run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from wfx.data.synthetic import SyntheticDataset

GOLDENS_PATH = Path(__file__).with_name("goldens.yaml")


@dataclass(frozen=True)
class Golden:
    id: str
    question: str
    category: str
    expect: dict[str, Any] = field(default_factory=dict)
    judge: list[str] = field(default_factory=list)


def placeholders(ds: SyntheticDataset) -> dict[str, str]:
    """Facts the goldens refer to, taken from the answer key (evaluation only; the agent never sees it)."""
    caveats = ds.caveats
    live_origin = ds.volumes["date"].max() + pd.Timedelta(days=1)
    late = caveats.loc[caveats["caveat_type"] == "late_feed", "store_id"].iloc[0]
    recent_gaps = caveats[
        (caveats["caveat_type"] == "gap")
        & (caveats["start_date"] >= live_origin - pd.Timedelta(days=28))
        & (caveats["store_id"] != late)
    ]
    gap_store = recent_gaps["store_id"].iloc[0] if len(recent_gaps) else late
    closed = caveats.loc[caveats["caveat_type"] == "store_closed", "store_id"].iloc[0]
    number = lambda store: str(int(re.sub(r"\D", "", store)))  # noqa: E731
    return {
        "late_feed_store": late,
        "late_feed_store_number": number(late),
        "gap_store": gap_store,
        "gap_store_number": number(gap_store),
        "closed_store_number": number(closed),
    }


def _fill(value: Any, facts: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**facts)
    if isinstance(value, list):
        return [_fill(v, facts) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, facts) for k, v in value.items()}
    return value


def load_goldens(ds: SyntheticDataset, path: Path = GOLDENS_PATH) -> list[Golden]:
    facts = placeholders(ds)
    raw = yaml.safe_load(path.read_text())
    goldens = [Golden(**_fill(item, facts)) for item in raw]
    ids = [g.id for g in goldens]
    if len(ids) != len(set(ids)):
        raise ValueError("golden ids must be unique")
    return goldens
