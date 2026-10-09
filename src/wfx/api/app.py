"""HTTP API and a minimal UI for the explainability agent.

Routes:
    GET  /            single-page UI (static; data comes from the routes below)
    GET  /health      liveness + the live run id (no auth: for platform probes)
    POST /ask         {"question": "..."} -> answer, outcome, evidence summary, latency   [user, rate-limited]
    GET  /runs        forecast runs                                                      [user]
    GET  /monitoring  accuracy/data-quality trends, agent health, eval trend, alerts      [admin]

Each request gets its own DuckDB cursor (same read-only, locked-down database) and its own
evidence tools scoped to the caller's stores, so requests run concurrently and a user can
only ever retrieve evidence for stores they're allowed to see. Monitoring aggregates across
all stores, so it is admin-only.

Every /ask is logged (user id, masked question, outcome, gate attempts, rejected values,
latency) for monitoring and audit.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from wfx.agent.graph import AgentAnswer, ExplainabilityAgent
from wfx.agent.llm import LLM
from wfx.api.auth import Authenticator, Principal, RateLimiter
from wfx.explain.store import connect_readonly
from wfx.explain.tools import EvidenceTools, json_default
from wfx.monitoring.agent import Interaction, InteractionLog, agent_alerts, agent_health, eval_trend
from wfx.monitoring.forecast import accuracy_trend, data_quality_trend, forecast_alerts

UI_PATH = Path(__file__).with_name("ui.html")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _records(frame: Any) -> list[dict[str, Any]]:
    return [{k: (json_default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v) for k, v in row.items()}
            for row in frame.astype(object).where(frame.notna(), None).to_dict("records")]


def create_app(
    db_path: Path,
    llm: LLM,
    log_path: Path,
    *,
    auth: Authenticator,
    limiter: RateLimiter | None = None,
    eval_dir: Path | None = None,
) -> FastAPI:
    con = connect_readonly(db_path)
    log = InteractionLog(log_path)
    limiter = limiter or RateLimiter()
    app = FastAPI(title="Workforce Forecast Explainability Agent")

    def principal(request: Request) -> Principal:
        who = auth.authenticate(request.headers.get("authorization"), dict(request.headers))
        if who is None:
            raise HTTPException(401, "Sign in to use the explainer.", headers={"WWW-Authenticate": "Bearer"})
        return who

    def admin(who: Principal = Depends(principal)) -> Principal:  # noqa: B008 - FastAPI idiom
        if not who.is_admin:
            raise HTTPException(403, "Monitoring is available to admins only.")
        return who

    def tools_for(who: Principal) -> EvidenceTools:
        return EvidenceTools(con.cursor(), allowed_stores=who.stores)

    @app.get("/", response_class=HTMLResponse)
    def ui() -> str:
        return UI_PATH.read_text()

    @app.get("/health")
    def health() -> dict[str, Any]:
        live = next(r for r in EvidenceTools(con.cursor()).list_runs().data if r["is_live"])
        return {"status": "ok", "live_run": live["run_id"]}

    @app.post("/ask")
    def ask(request: AskRequest, who: Principal = Depends(principal)) -> dict[str, Any]:  # noqa: B008
        wait = limiter.acquire(who.user_id)
        if wait:
            raise HTTPException(429, "Too many questions; try again shortly.", headers={"Retry-After": str(max(1, round(wait)))})
        started = time.perf_counter()
        try:
            result = ExplainabilityAgent(tools_for(who), llm).ask(request.question)
        except Exception as err:  # LLM timeout/outage: log it, tell the user plainly, never leak internals
            latency_ms = int((time.perf_counter() - started) * 1000)
            failed = AgentAnswer(answer="", outcome="error", plan=None)
            log.append(Interaction.from_answer(request.question, failed, latency_ms, user_id=who.user_id, error=type(err).__name__))
            raise HTTPException(503, "The assistant is unavailable right now; please try again shortly.") from None
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.append(Interaction.from_answer(request.question, result, latency_ms, user_id=who.user_id))
        return {
            "answer": result.answer,
            "outcome": result.outcome,
            "question_type": result.plan.question_type.value if result.plan else None,
            "attempts": result.attempts,
            "latency_ms": latency_ms,
            "evidence": [{"tool": e["tool"], "status": e["status"], "notes": e["notes"]} for e in result.evidence],
        }

    @app.get("/runs")
    def runs(who: Principal = Depends(principal)) -> list[dict[str, Any]]:  # noqa: B008
        return _records_from_rows(tools_for(who).list_runs().data)

    @app.get("/monitoring")
    def monitoring(who: Principal = Depends(admin)) -> dict[str, Any]:  # noqa: B008
        cur = con.cursor()
        f_alerts = forecast_alerts(cur)
        health_ = agent_health(log.read(last=200))
        alerts = [a.to_dict() for a in (*f_alerts, *agent_alerts(health_))]
        return {
            "alerts": alerts,
            "forecast": {"accuracy_trend": _records(accuracy_trend(cur)), "data_quality_trend": _records(data_quality_trend(cur))},
            "agent": asdict(health_),
            "evals": eval_trend(eval_dir) if eval_dir and eval_dir.exists() else [],
        }

    return app


def _records_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: (v if isinstance(v, (str, int, float, bool, type(None))) else json_default(v)) for k, v in r.items()} for r in rows]
