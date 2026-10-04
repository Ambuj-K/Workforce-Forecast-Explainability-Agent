"""Structured types the agent passes between steps."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    WHY_HOURS = "why_hours"  # why does a store-department-week need these hours?
    WHY_VOLUME = "why_volume"  # why is a store-driver-day forecast what it is?
    HOURS_BREAKDOWN = "hours_breakdown"  # how does volume become hours?
    TRUST = "trust"  # how accurate has the forecast been?
    CHANGE = "change"  # what changed since the last run?
    BASELINE = "baseline"  # is the model better than the simple method here?
    CAVEATS = "caveats"  # what should I be careful about?
    DEFINITION = "definition"  # what does a feature or group mean?
    RUNS = "runs"  # which forecast runs exist / which is current?
    STAFFING_DECISION = "staffing_decision"  # asks the agent to decide staffing (declined)
    OUT_OF_SCOPE = "out_of_scope"  # unrelated to the forecasts


class QuestionPlan(BaseModel):
    """What the user is asking, extracted by the planner. Values are copied from the question; never invented."""

    question_type: QuestionType
    store: str | None = Field(default=None, description="Store as written by the user, e.g. 'S001' or 'store 1'.")
    department: str | None = Field(default=None, description="Department as written, e.g. 'grocery'.")
    driver: str | None = Field(default=None, description="Workload driver as written, e.g. 'transactions'.")
    week: str | None = Field(default=None, description="A date in the week asked about, YYYY-MM-DD.")
    day: str | None = Field(default=None, description="A specific day asked about, YYYY-MM-DD.")
    run_id: str | None = Field(default=None, description="A run id only if the user named one.")
    term: str | None = Field(default=None, description="The feature or group to define, for definition questions.")
    reason: str = Field(description="One short sentence on why this question type was chosen.")
