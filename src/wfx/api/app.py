"""HTTP API and a minimal UI for the explainability agent.

Routes:
    GET  /            single-page UI (ask questions, see evidence, see monitoring)
    GET  /health      liveness + the live run id
    POST /ask         {"question": "..."} -> answer, outcome, evidence summary, latency
    GET  /runs        forecast runs
    GET  /monitoring  forecast accuracy/data-quality trends, agent health, eval trend, alerts

Every /ask is logged (masked question, outcome, gate attempts, rejected values, latency) for
monitoring. The DuckDB connection is shared, so questions are answered one at a time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from wfx.agent.graph import ExplainabilityAgent
from wfx.agent.llm import LLM
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


def create_app(db_path: Path, llm: LLM, log_path: Path, eval_dir: Path | None = None) -> FastAPI:
    con = connect_readonly(db_path)
    tools = EvidenceTools(con)
    agent = ExplainabilityAgent(tools, llm)
    log = InteractionLog(log_path)
    lock = threading.Lock()
    app = FastAPI(title="Workforce Forecast Explainability Agent")

    @app.get("/", response_class=HTMLResponse)
    def ui() -> str:
        return UI_PATH.read_text()

    @app.get("/health")
    def health() -> dict[str, Any]:
        with lock:
            live = next(r for r in tools.list_runs().data if r["is_live"])
        return {"status": "ok", "live_run": live["run_id"]}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        started = time.perf_counter()
        with lock:
            result = agent.ask(request.question)
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.append(Interaction.from_answer(request.question, result, latency_ms))
        return {
            "answer": result.answer,
            "outcome": result.outcome,
            "question_type": result.plan.question_type.value if result.plan else None,
            "attempts": result.attempts,
            "latency_ms": latency_ms,
            "evidence": [{"tool": e["tool"], "status": e["status"], "notes": e["notes"]} for e in result.evidence],
        }

    @app.get("/runs")
    def runs() -> list[dict[str, Any]]:
        with lock:
            return _records_from_rows(tools.list_runs().data)

    @app.get("/monitoring")
    def monitoring() -> dict[str, Any]:
        with lock:
            trend = accuracy_trend(con)
            quality = data_quality_trend(con)
            f_alerts = forecast_alerts(con)
        health_ = agent_health(log.read(last=200))
        alerts = [a.to_dict() for a in (*f_alerts, *agent_alerts(health_))]
        return {
            "alerts": alerts,
            "forecast": {"accuracy_trend": _records(trend), "data_quality_trend": _records(quality)},
            "agent": asdict(health_),
            "evals": eval_trend(eval_dir) if eval_dir and eval_dir.exists() else [],
        }

    return app


def _records_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: (v if isinstance(v, (str, int, float, bool, type(None))) else json_default(v)) for k, v in r.items()} for r in rows]
