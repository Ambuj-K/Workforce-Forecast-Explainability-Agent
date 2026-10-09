# Security and AI-Risk Controls

**Scope:** the explainability agent, its evidence tools, the agent database, the HTTP API, the container image and the monitoring log.
**Principle:** every control names the test or eval that proves it. `tests/test_security_controls.py` fails if a cited test or golden question disappears, so this document can't silently drift from the code.

---

## 1. System and trust boundaries

```
 user ──(1)──▶ API /ask ──▶ planner LLM ──▶ routing (code) ──▶ evidence tools ──(2)──▶ agent DB (read-only)
                                                                        │
                       answer ◀── faithfulness gate (code) ◀── explainer LLM ◀── evidence (JSON)
                                                                        │
                                                         interaction log (masked)
 pipeline ──(3)──▶ validated extract ──▶ agent DB          LLM provider ◀──(4)── prompts + evidence
```
| Boundary | What crosses it | Trust |
|---|---|---|
| (1) user → API | bearer token or proxy identity + free-text question | identity verified per request; question **untrusted**: may contain instructions, false premises, personal data |
| (2) tools → database | parameterised queries | user text never reaches SQL; the database can't be written or used to read files |
| (3) pipeline → extract | forecasts, contributions, flags | trusted only after the contract (schemas + invariants) passes at build, write and load |
| (4) agent → LLM provider | question + evidence | **data leaves the system**: what is sent, and the provider's data terms, matter |

**Assets:** forecast integrity (numbers planners act on), the truthfulness of explanations, the API key, data in prompts, availability of the API.
**Actors considered:** an unauthenticated caller, a signed-in user trying to see other stores, a curious or careless user, a user attempting prompt injection, a compromised data feed, the LLM itself (hallucination, arithmetic errors).

---

## 2. Controls

| ID | Control | Risk addressed | Verified by |
|---|---|---|---|
| C-01 | Agent database opened **read-only** with **external access disabled** and **configuration locked** | LLM-written or injected SQL reading files/secrets, writing data, re-enabling access | `tests/test_extract_store.py::test_locked_down_connection_blocks_escapes` (8 attacks) |
| C-02 | Tools use **fixed, parameterised SQL**; user text is only ever a bound parameter | SQL injection | `tests/test_tools.py::test_injection_attempt_in_a_name_is_just_not_found` |
| C-03 | **Strict entity resolution**: store names must be store-shaped; fuzzy matches are disclosed; ambiguity asks | junk or injection text silently resolving to a real entity | `tests/test_tools.py::test_injection_attempt_in_a_name_is_just_not_found`, `tests/test_tools.py::test_ambiguous_name_asks_instead_of_guessing`, `tests/test_tools.py::test_fuzzy_match_is_disclosed` |
| C-04 | **The LLM never chooses tools**: routing by question type in code | excessive agency; tool misuse via injection | `tests/test_agent.py::test_trust_question_routes_to_accuracy_and_caveats`, `tests/test_agent.py::test_answers_a_why_hours_question_first_time` |
| C-05 | **Numeric faithfulness gate**: every number/date in an answer must exist in the evidence; computed numbers rejected; retry once, then a templated answer from verified facts | hallucinated or miscalculated figures reaching planners | `tests/test_agent.py::test_gate_rejects_invented_computed_or_wrong_values`, `tests/test_agent.py::test_invented_number_is_rejected_then_corrected`, `tests/test_agent.py::test_repeated_invention_falls_back_to_verified_facts` |
| C-06 | **False-premise rule**: a figure from the question is allowed only in a sentence that negates it | the agent agreeing with a user's wrong number | `tests/test_agent.py::test_gate_does_not_let_a_false_premise_through`, `tests/test_agent.py::test_gate_rejects_agreeing_with_a_false_premise_even_with_other_negations`, `goldens.yaml#premise-promotion-80` |
| C-07 | **Staffing decisions declined** with an offer of evidence; no recommendations in answers | over-reliance; the agent acting as decision-maker | `tests/test_agent.py::test_staffing_decision_is_declined_with_an_offer`, `tests/test_evals.py::test_staffing_advice_and_internal_leaks_are_caught`, `goldens.yaml#staffing-cut`, `goldens.yaml#staffing-roster` |
| C-08 | **Off-topic and instruction-override requests declined** | scope creep; prompt injection; system-prompt disclosure | `tests/test_agent.py::test_off_topic_is_declined`, `goldens.yaml#off-topic`, `goldens.yaml#injection-number`, `goldens.yaml#injection-secrets` |
| C-09 | **Internal instructions never shown to users**: tool results separate internal `action` from user-facing `message` | leaking system internals | `tests/test_agent.py::test_unknown_department_is_guided_without_calling_the_explainer` |
| C-10 | **Explanations framed as model attributions**, not causes; underperformance stated when evidence shows it | misinformation; overconfidence | `tests/test_evals.py::test_missing_value_and_framing_fail`, `tests/test_evals.py::test_underperformance_must_be_stated_when_evidence_shows_it` |
| C-11 | **Extract contract**: schemas + invariants enforced at build, write and load; tampering is detected | corrupted or poisoned pipeline outputs | `tests/test_extract_store.py::test_tampered_contribution_breaks_the_contract`, `tests/test_extract_store.py::test_write_and_load_round_trip_with_validation` |
| C-12 | **As-of, reported-data-only extract**: answer keys and future data never reach the agent | leakage of information the agent must not have | `tests/test_extract_store.py::test_extract_never_reads_ground_truth`, `tests/test_model_backtest.py::test_backtest_cannot_see_the_future`, `tests/test_prepare_features.py::test_history_features_never_see_the_forecast_window` |
| C-13 | **Input bounded**: questions limited to 500 characters | prompt stuffing; cost/latency abuse | `tests/test_monitoring_api.py::test_ask_rejects_oversized_questions` |
| C-14 | **Logs hold masked questions and no answers** | personal data in logs | `tests/test_monitoring_api.py::test_questions_are_masked_before_logging`, `tests/test_monitoring_api.py::test_ask_answers_and_logs_a_masked_interaction` |
| C-15 | **No secrets in the repository**: the API key lives only in a gitignored `.env`; tracked files are scanned for key patterns | credential leakage | `tests/test_security_controls.py::test_no_secrets_in_tracked_files`, `tests/test_security_controls.py::test_env_file_is_ignored` |
| C-16 | **Continuous monitoring**: accuracy drift (baseline-relative), worse-than-baseline, stale data, agent first-pass/fallback/latency alerts | silent degradation | `tests/test_monitoring_api.py::test_real_drift_alerts`, `tests/test_monitoring_api.py::test_hard_period_is_not_drift`, `tests/test_monitoring_api.py::test_agent_health_and_alerts` |
| C-17 | **Golden-question evals with regression baseline** (25 cases incl. injection, premise, staffing) | behaviour regressing after prompt/model changes | `tests/test_evals.py::test_regression_detection`, `evals/baseline.json` |
| C-18 | **Sign-in on every data route**; monitoring (cross-store aggregates) is admin-only; tokens stored only as hashes and compared in constant time; proxy-identity mode accepts only listed users; running without sign-in is refused off loopback | unauthenticated access; privilege escalation to cross-store views | `tests/test_access.py::test_routes_require_sign_in_and_monitoring_requires_admin`, `tests/test_access.py::test_tokens_resolve_to_users_and_bad_tokens_do_not`, `tests/test_access.py::test_header_mode_trusts_only_listed_users`, `tests/test_access.py::test_no_sign_in_is_refused_off_loopback` |
| C-19 | **Per-user store scoping inside the evidence tools**: other stores resolve as not found, are never listed, and store-free questions aggregate only the user's stores (and say so) | a planner reading another region's forecasts, including by asking the agent to "ignore the restriction" | `tests/test_access.py::test_out_of_scope_store_is_not_found_and_not_listed`, `tests/test_access.py::test_store_free_questions_only_aggregate_the_users_stores`, `tests/test_access.py::test_empty_scope_sees_nothing`, `tests/test_access.py::test_planner_cannot_get_evidence_for_another_store` |
| C-20 | **Per-user rate limit** on questions (token bucket; 429 with Retry-After) | cost abuse; one user starving others | `tests/test_access.py::test_rate_limit_allows_a_burst_then_waits`, `tests/test_access.py::test_ask_is_rate_limited_per_user` |
| C-21 | **Secrets from mounted files** (`NAME_FILE`) ahead of environment variables; error messages name the variable, never the value | key exposure via environment dumps or logs | `tests/test_access.py::test_secret_prefers_a_mounted_file_and_never_echoes_values` |
| C-22 | **LLM failures contained**: per-call timeout; a failure returns a plain 503 without internals, is logged with its type, and an error-rate alert fires | hung requests; leaking stack traces; silent outages | `tests/test_access.py::test_llm_failure_is_a_clean_503_and_is_logged`, `tests/test_access.py::test_error_rate_alerts` |
| C-23 | **Concurrent requests without a shared lock**: each request gets its own cursor on the same locked-down database | one slow question blocking everyone; lock-down lost on new cursors | `tests/test_extract_store.py::test_locked_down_connection_blocks_escapes` (connection and cursor), `tests/test_access.py::test_concurrent_requests_each_get_their_own_cursor` |
| C-24 | **Container image**: non-root user, sign-in on by default, no secrets or local data in the build context, key and users file mounted at run time | secrets baked into images; running as root; an open-by-default deployment | `tests/test_security_controls.py::test_container_runs_non_root_with_sign_in_and_no_secrets` |

---

## 3. Framework mappings

### 3.1 OWASP Top 10 for LLM Applications (2025)
| Risk | Status | Controls |
|---|---|---|
| LLM01 Prompt injection | Mitigated (direct); residual risk noted for indirect | C-02, C-03, C-04, C-08; evidence is structured data and code-generated notes |
| LLM02 Sensitive information disclosure | Partially mitigated | C-01, C-09, C-14, C-15, C-18, C-19, C-21; **gap G-04** (provider data terms) |
| LLM03 Supply chain | Partially mitigated | dependencies locked in `uv.lock`; **gap G-06** (no vulnerability scanning) |
| LLM04 Data and model poisoning | Mitigated for the pipeline→agent boundary | C-11, C-12; outlier/duplicate detection in data preparation |
| LLM05 Improper output handling | Mitigated | C-05, C-06; answers are plain text; the UI renders them as text, never as HTML |
| LLM06 Excessive agency | Mitigated | C-04 (no LLM tool choice), C-01 (read-only), C-07 (no decisions), C-19 (tools can only reach the user's stores) |
| LLM07 System prompt leakage | Mitigated; low impact | C-08; prompts contain no secrets |
| LLM08 Vector and embedding weaknesses | Not applicable | no retrieval over embeddings |
| LLM09 Misinformation | Mitigated | C-05, C-06, C-10, C-17 |
| LLM10 Unbounded consumption | Mitigated in-app; platform limits pending | C-13, C-20, C-22; at most 3 LLM calls per question; **gap G-02** (limits are per instance) |

### 3.2 NIST AI Risk Management Framework
| Function | What exists |
|---|---|
| **Govern** | Decision records (`docs/decisions.md`), evolution log, this document; controls tied to tests |
| **Map** | Trust boundaries and actors (§1); intended use: explaining forecasts to planners, explicitly not making staffing decisions (C-07) |
| **Measure** | Walk-forward accuracy vs a baseline by lead time; golden evals with deterministic checks + judge; faithfulness first-pass rate; detector recall against an answer key |
| **Manage** | Alerts (C-16, C-22), regression baseline (C-17), guarded/templated fallbacks (C-05), access control (C-18–C-20) |

### 3.3 ISO/IEC 42001 (AI management system), by theme
| Theme | Evidence |
|---|---|
| AI risk assessment and treatment | §1 threat model, §2 controls, §4 gaps with owners |
| Data quality and provenance | Extract contract (C-11), as-of construction (C-12), quality report and caveats exposed to users |
| Transparency to users | Answers disclose interpretations, caveats, attribution framing and accuracy by lead time (C-10) |
| Human oversight | Decisions stay with planners (C-07); clarifying questions instead of guesses |
| Access control and accountability | Sign-in and roles (C-18), least-privilege data scoping (C-19), per-user interaction log for audit (C-14) |
| Monitoring and continual improvement | C-16, C-17, evolution log |

### 3.4 EU AI Act, risk classification (assessment, not legal advice)
- **Intended use:** explains store-level labour-demand forecasts (volumes and hours per store, department and week) to planners. It does not process data about individual workers and does not allocate tasks to, monitor or evaluate individuals.
- **Assessment:** most likely **not** a high-risk system under Annex III (employment and workers management), which targets systems that make or support decisions about individual workers. **Transparency obligations** for systems interacting with people apply: the UI states that it is an AI assistant whose numbers are checked against forecast data.
- **What would change this:** using it to allocate shifts to named people, or to evaluate individual or team performance, would likely move it into the high-risk category and require a conformity assessment, risk management system, logging and human oversight measures beyond this design.

---

## 4. Known gaps

| ID | Gap | Status | Plan |
|---|---|---|---|
| G-01 | Authentication | **Partly closed** (C-18): in-app tokens or proxy identity. Tokens don't expire, and revoking one means editing the users file and restarting | Deployment: the platform's identity proxy (SSO) in front, `WFX_AUTH=header`; TLS terminated by the platform |
| G-02 | Rate limiting | **Partly closed** (C-20): per user, per instance | Deployment: gateway limits across instances; a monthly spend cap on the LLM project |
| G-03 | Per-user data scoping | **Closed** (C-19) | Map users to stores from the identity provider's groups instead of a file |
| G-04 | **Free-tier LLM provider terms**: unpaid API tiers may use submitted content to improve services | Open | Development uses synthetic data only; a real deployment uses a paid tier or managed platform with no-training terms. Check the provider's current terms |
| G-05 | **Indirect prompt injection** via data fields (e.g. a store name crafted as an instruction) | Open | Evidence is mostly numeric with code-generated notes; sanitise text fields at the extract boundary before using real data |
| G-06 | **No dependency vulnerability scanning or SBOM** | Open | Add `pip-audit` (or equivalent) and an SBOM to CI; scan the image |
| G-07 | **Judge is the same model family as the agent** | Open | Deterministic checks carry the security-relevant assertions; consider a different judge model |
| G-08 | **Latency** 4–24 s per explained answer | **Diagnosed**: two LLM calls; graph build ~10 ms; the model reports no thinking tokens; the spread is provider-side queueing on the free tier. A timeout now bounds it (C-22) | Deployment: paid or provisioned endpoint. Answers can't be streamed because the gate checks the whole answer before anyone sees it; show progress instead |
| G-09 | Interaction log is a local file (ephemeral in a container) | Open | Deployment: write to stdout or the platform's log service, with retention set |
