"""Tests for monitoring rules, the interaction log, and the HTTP API (scripted LLM, no network)."""

from datetime import date

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wfx.agent.llm import ScriptedLLM
from wfx.agent.schema import QuestionPlan, QuestionType
from wfx.api.app import create_app
from wfx.data.config import GeneratorConfig
from wfx.data.synthetic import generate
from wfx.explain.extract import build_extract
from wfx.explain.store import build_database
from wfx.forecasting.model import ModelParams
from wfx.monitoring.agent import AgentRules, InteractionLog, agent_alerts, agent_health, mask
from wfx.monitoring.forecast import forecast_alerts


# --------------------------------------------------------------- forecast rules


def accuracy_db(runs: list[tuple[float, float]]) -> duckdb.DuckDBPyConnection:
    """A database with one store-driver whose model/baseline error per run is given."""
    rows = [
        {"scored_run_id": f"r{i}", "store_id": "S001", "driver": "transactions", "lead_week": 1, "days": 7,
         "actual": 100.0, "abs_err_model": model, "abs_err_baseline": base, "window_end": pd.Timestamp("2025-01-05") + pd.Timedelta(days=28 * i)}
        for i, (model, base) in enumerate(runs)
    ]
    con = duckdb.connect()
    con.register("acc", pd.DataFrame(rows))
    con.execute("CREATE TABLE accuracy AS SELECT * FROM acc")
    con.execute("CREATE TABLE runs (run_id VARCHAR, origin TIMESTAMP, is_live BOOLEAN)")
    con.execute("CREATE TABLE eligibility (run_id VARCHAR, store_id VARCHAR, data_is_stale BOOLEAN, latest_actual_date TIMESTAMP)")
    return con


def rules_fired(con) -> set[str]:
    return {a.rule for a in forecast_alerts(con)}


def test_hard_period_is_not_drift():
    # error jumps (6% -> 9%) but the simple method got much worse too (skill rises): Christmas-like
    assert "accuracy_drift" not in rules_fired(accuracy_db([(6, 10), (6, 10), (6, 10), (9, 20)]))


def test_real_drift_alerts():
    # error jumps and the edge over the simple method shrinks
    assert "accuracy_drift" in rules_fired(accuracy_db([(6, 10), (6, 10), (6, 10), (9, 10)]))


def test_worse_than_baseline_alerts_and_persistence_is_tracked():
    fired = rules_fired(accuracy_db([(6, 10), (12, 10), (12, 10)]))
    assert {"worse_than_baseline", "persistently_worse_than_baseline"} <= fired


# ------------------------------------------------------------------ agent log


def record(outcome: str, attempts: int = 1, latency: int = 1000, rejected=()):
    return {"outcome": outcome, "attempts": attempts, "latency_ms": latency, "rejected_values": list(rejected)}


def test_agent_health_and_alerts():
    records = [record("answered")] * 8 + [record("answered", attempts=2, latency=20_000, rejected=["5.7%"])] * 3 + [record("fallback", 2, 20_000, ["80"])]
    health = agent_health(records)
    assert health.explained == 12
    assert health.first_pass_rate == pytest.approx(8 / 12)
    assert health.fallback_rate == pytest.approx(1 / 12)
    assert health.top_rejected_values[0] == ("5.7%", 3)
    assert {a.rule for a in agent_alerts(health, AgentRules(min_sample=10))} == {"gate_first_pass_low", "fallback_rate_high", "latency_high"}


def test_no_alerts_on_tiny_samples():
    assert agent_alerts(agent_health([record("fallback", 2, 99_000)])) == []


def test_questions_are_masked_before_logging():
    masked = mask("store 1? email me at jo.bloggs@example.com or +44 7700 900123")
    assert "example.com" not in masked and "7700" not in masked and "store 1" in masked


# ------------------------------------------------------------------------ API


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    ds = generate(GeneratorConfig(start=date(2023, 1, 2), end=date(2024, 12, 29), n_stores=4, seed=5))
    extract = build_extract(ds, recent_runs=2, scored_runs=3, params=ModelParams(max_iter=60))
    return build_database(extract, tmp_path_factory.mktemp("api") / "agent.duckdb")


def client(db, tmp_path, llm) -> tuple[TestClient, InteractionLog]:
    log_path = tmp_path / "interactions.jsonl"
    return TestClient(create_app(db, llm, log_path)), InteractionLog(log_path)


def test_health_ui_and_runs(db, tmp_path):
    c, _ = client(db, tmp_path, ScriptedLLM())
    assert c.get("/health").json()["status"] == "ok"
    assert "<title>Workforce Forecast Explainer</title>" in c.get("/").text
    assert any(r["is_live"] for r in c.get("/runs").json())


def test_ask_answers_and_logs_a_masked_interaction(db, tmp_path):
    plan = QuestionPlan(question_type=QuestionType.STAFFING_DECISION, reason="t")
    c, log = client(db, tmp_path, ScriptedLLM(structured=[plan]))
    body = c.post("/ask", json={"question": "Should I cut staff? call me on +44 7700 900123"}).json()
    assert body["outcome"] == "declined" and body["evidence"] == []
    [entry] = log.read()
    assert entry["outcome"] == "declined" and "7700" not in entry["question"]


def test_ask_rejects_oversized_questions(db, tmp_path):
    c, _ = client(db, tmp_path, ScriptedLLM())
    assert c.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_monitoring_reports_trends_alerts_and_agent_health(db, tmp_path):
    c, _ = client(db, tmp_path, ScriptedLLM())
    data = c.get("/monitoring").json()
    assert len(data["forecast"]["accuracy_trend"]) >= 2
    assert any(a["rule"] == "stale_data_live" for a in data["alerts"])
    assert data["agent"]["interactions"] == 0
