"""Access control: sign-in, roles, per-user store scoping, rate limits and secrets (scripted LLM, no network)."""

import pytest
from fastapi.testclient import TestClient

from wfx.agent.llm import ScriptedLLM
from wfx.agent.schema import QuestionPlan, QuestionType
from wfx.api.app import create_app
from wfx.api.auth import Authenticator, RateLimiter, check_bind, hash_token, load_users
from wfx.explain.store import connect_readonly
from wfx.explain.tools import EvidenceTools
from wfx.monitoring.agent import InteractionLog
from wfx.secrets import secret

USERS = """
users:
  - {id: planner-one, role: planner, stores: [S001], token_sha256: %s}
  - {id: boss, role: admin, stores: "*", token_sha256: %s}
  - {id: proxy-user, role: planner, stores: [S003]}
"""


@pytest.fixture
def users_file(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text(USERS % (hash_token("planner-token"), hash_token("admin-token")))
    return path


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------ scoping


@pytest.fixture
def scoped(api_db):
    con = connect_readonly(api_db)
    yield EvidenceTools(con, allowed_stores=frozenset({"S001"})), EvidenceTools(con.cursor())
    con.close()


def test_out_of_scope_store_is_not_found_and_not_listed(scoped):
    tools, everyone = scoped
    live = next(r for r in tools.list_runs().data if r["is_live"])
    assert everyone.explain_week_hours("S003", "grocery", live["origin"]).status == "ok"
    result = tools.explain_week_hours("store 3", "grocery", live["origin"])
    assert result.status == "not_found"
    assert result.data == [{"store": "S001"}]  # the user's own stores, never the others
    assert tools.find_entities().data[0] == {"stores": ["S001"]}


def test_store_free_questions_only_aggregate_the_users_stores(scoped):
    tools, everyone = scoped
    assert tools.get_accuracy().data == tools.get_accuracy(store="S001").data
    assert tools.get_accuracy().data != everyone.get_accuracy().data
    assert {f["store_id"] for f in tools.get_caveats().data} <= {"S001"}
    assert {f["store_id"] for f in everyone.get_caveats().data} - {"S001"}  # others do have caveats
    assert "only the stores you have access to: S001" in " ".join(tools.get_caveats().notes)  # so "no caveats" isn't read as global
    assert "access to" not in " ".join(everyone.get_caveats().notes)


def test_empty_scope_sees_nothing(api_db):
    tools = EvidenceTools(connect_readonly(api_db), allowed_stores=frozenset())
    assert tools.get_accuracy().status == "not_found"
    assert tools.get_caveats().data == []
    assert tools.explain_week_hours("S001", "grocery", "2024-12-02").status == "not_found"


# --------------------------------------------------------------------- auth


def test_tokens_resolve_to_users_and_bad_tokens_do_not(users_file):
    auth = Authenticator.from_file("token", users_file)
    assert auth.authenticate("Bearer planner-token", {}).user_id == "planner-one"
    assert auth.authenticate("bearer admin-token", {}).is_admin
    for header in (None, "", "Bearer", "Bearer wrong", "Basic planner-token", hash_token("planner-token")):
        assert auth.authenticate(header, {}) is None
    assert auth.authenticate(None, {"x-authenticated-user": "planner-one"}) is None  # header mode is off


def test_header_mode_trusts_only_listed_users(users_file):
    auth = Authenticator.from_file("header", users_file, header="X-Authenticated-User")
    assert auth.authenticate(None, {"x-authenticated-user": "proxy-user"}).stores == frozenset({"S003"})
    assert auth.authenticate(None, {"x-authenticated-user": "stranger"}) is None
    assert auth.authenticate("Bearer planner-token", {}) is None  # tokens are not accepted in header mode


def test_users_file_is_validated_and_auth_needs_users(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("users:\n  - {id: x, role: root, stores: '*'}\n")
    with pytest.raises(ValueError, match="role"):
        load_users(bad)
    with pytest.raises(ValueError, match="users file"):
        Authenticator("token")


def test_no_sign_in_is_refused_off_loopback():
    check_bind("none", "127.0.0.1")
    check_bind("token", "0.0.0.0")
    for host in ("0.0.0.0", "10.0.0.5", "example.internal"):
        with pytest.raises(ValueError, match="loopback"):
            check_bind("none", host)


def test_rate_limit_allows_a_burst_then_waits():
    now = [0.0]
    limiter = RateLimiter(per_minute=6, burst=2, clock=lambda: now[0])
    assert limiter.acquire("a") == 0 and limiter.acquire("a") == 0
    assert limiter.acquire("a") == pytest.approx(10)
    assert limiter.acquire("b") == 0  # per user
    now[0] = 10.0
    assert limiter.acquire("a") == 0


# ---------------------------------------------------------------------- API


def api(db, tmp_path, users_file, llm=None, limiter=None) -> tuple[TestClient, InteractionLog]:
    log_path = tmp_path / "interactions.jsonl"
    app = create_app(db, llm or ScriptedLLM(), log_path, auth=Authenticator.from_file("token", users_file), limiter=limiter)
    return TestClient(app), InteractionLog(log_path)


def test_routes_require_sign_in_and_monitoring_requires_admin(api_db, tmp_path, users_file):
    c, _ = api(api_db, tmp_path, users_file)
    assert c.get("/health").status_code == 200 and c.get("/").status_code == 200
    assert c.post("/ask", json={"question": "why?"}).status_code == 401
    assert c.get("/runs").status_code == 401
    assert c.get("/monitoring").status_code == 401
    assert c.get("/runs", headers=bearer("planner-token")).status_code == 200
    assert c.get("/monitoring", headers=bearer("planner-token")).status_code == 403
    assert c.get("/monitoring", headers=bearer("admin-token")).status_code == 200


def test_planner_cannot_get_evidence_for_another_store(api_db, tmp_path, users_file):
    week = next(r for r in EvidenceTools(connect_readonly(api_db)).list_runs().data if r["is_live"])["origin"].date().isoformat()
    plan = QuestionPlan(question_type=QuestionType.WHY_HOURS, store="S003", department="grocery", week=week, reason="t")
    c, log = api(api_db, tmp_path, users_file, llm=ScriptedLLM(structured=[plan]))
    body = c.post("/ask", json={"question": "Why does store 3 grocery need these hours?"}, headers=bearer("planner-token")).json()
    assert body["outcome"] == "guarded"
    assert "S003" not in body["answer"].replace("'S003'", "")  # only echoes what was asked
    assert "S004" not in body["answer"]
    assert "S001" in body["answer"]
    [entry] = log.read()
    assert entry["user_id"] == "planner-one"


def test_ask_is_rate_limited_per_user(api_db, tmp_path, users_file):
    plans = [QuestionPlan(question_type=QuestionType.OUT_OF_SCOPE, reason="t") for _ in range(3)]
    c, _ = api(api_db, tmp_path, users_file, llm=ScriptedLLM(structured=plans), limiter=RateLimiter(per_minute=1, burst=1))
    assert c.post("/ask", json={"question": "weather?"}, headers=bearer("planner-token")).status_code == 200
    limited = c.post("/ask", json={"question": "weather?"}, headers=bearer("planner-token"))
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
    assert c.post("/ask", json={"question": "weather?"}, headers=bearer("admin-token")).status_code == 200


# ------------------------------------------------------------------ secrets


def test_secret_prefers_a_mounted_file_and_never_echoes_values(tmp_path, monkeypatch):
    mounted = tmp_path / "key"
    mounted.write_text("from-file\n")
    monkeypatch.setenv("WFX_TEST_KEY", "from-env")
    assert secret("WFX_TEST_KEY") == "from-env"
    monkeypatch.setenv("WFX_TEST_KEY_FILE", str(mounted))
    assert secret("WFX_TEST_KEY") == "from-file"
    monkeypatch.setenv("WFX_TEST_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="doesn't exist"):
        secret("WFX_TEST_KEY")
    monkeypatch.delenv("WFX_TEST_KEY_FILE")
    monkeypatch.delenv("WFX_TEST_KEY")
    with pytest.raises(RuntimeError) as err:
        secret("WFX_TEST_KEY")
    assert "from-env" not in str(err.value)
    assert secret("WFX_TEST_KEY", required=False) is None


# --------------------------------------------------------------- failures


class BrokenLLM:
    def structured(self, system, user, schema):
        raise TimeoutError("provider took too long; internal detail")

    text = structured


def test_llm_failure_is_a_clean_503_and_is_logged(api_db, tmp_path, users_file):
    c, log = api(api_db, tmp_path, users_file, llm=BrokenLLM())
    response = c.post("/ask", json={"question": "Why does store 1 need these hours?"}, headers=bearer("planner-token"))
    assert response.status_code == 503 and "internal detail" not in response.text
    [entry] = log.read()
    assert entry["outcome"] == "error" and entry["error"] == "TimeoutError" and entry["user_id"] == "planner-one"


def test_error_rate_alerts():
    from wfx.monitoring.agent import AgentRules, agent_alerts, agent_health

    records = [{"outcome": "declined", "attempts": 0, "latency_ms": 100, "rejected_values": []}] * 8
    records += [{"outcome": "error", "attempts": 0, "latency_ms": 30_000, "rejected_values": []}] * 2
    health = agent_health(records)
    assert health.error_rate == pytest.approx(0.2)
    assert "agent_errors_high" in {a.rule for a in agent_alerts(health, AgentRules(min_sample=10, max_latency_p95_ms=10**9))}


def test_concurrent_requests_each_get_their_own_cursor(api_db, tmp_path, users_file):
    from concurrent.futures import ThreadPoolExecutor

    c, _ = api(api_db, tmp_path, users_file)
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda i: c.get("/runs" if i % 2 else "/monitoring", headers=bearer("admin-token")).status_code, range(32)))
    assert codes == [200] * 32
