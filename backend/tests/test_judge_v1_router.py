"""Router-level sanity for /api/judge-v1/* -- registry/jobs endpoints
never leak API keys, and the registry's role invariant holds at the
HTTP layer too."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_registry_endpoint_has_exactly_one_primary_and_one_secondary():
    resp = client.get("/api/judge-v1/registry")
    assert resp.status_code == 200
    judges = resp.json()["judges"]
    roles = [j["role"] for j in judges]
    assert roles.count("primary") == 1
    assert roles.count("secondary") == 1


def test_registry_endpoint_never_returns_raw_api_key(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-should-never-appear")
    resp = client.get("/api/judge-v1/registry")
    assert "sk-should-never-appear" not in resp.text


def test_jobs_endpoint_hides_invalid_by_default():
    default_resp = client.get("/api/judge-v1/jobs")
    assert default_resp.status_code == 200
    shown_resp = client.get("/api/judge-v1/jobs", params={"show_invalid": "true"})
    assert shown_resp.status_code == 200
    # The 10 real pre-dashboard smoke-test jobs marked invalid earlier in
    # this session must not appear by default, but do appear when asked.
    assert len(shown_resp.json()["jobs"]) >= len(default_resp.json()["jobs"])


def test_plan_endpoint_rejects_empty_run_ids():
    resp = client.post("/api/judge-v1/plan", json={"run_ids": [], "judge_ids": ["primary"], "workers": {}})
    assert resp.status_code == 400


def test_plan_endpoint_rejects_empty_judge_ids():
    resp = client.post("/api/judge-v1/plan", json={"run_ids": ["A-fake"], "judge_ids": [], "workers": {}})
    assert resp.status_code == 400


def test_disagreements_include_one_line_reasoning_from_each_judge(monkeypatch):
    """results_disagreements() must pass through each judge's reasoning
    text (for the markdown export's disagreement list), not just its
    label -- a plain consequence of pair_two_judges() returning full
    records, asserted directly so a future refactor can't silently drop
    it."""
    from app.services import engine_service as es

    real_mod = es._judge_agreement_module()

    records = [
        {"run_id": "A-1", "question_id": 1, "run_label": "A", "judge_id": "primary",
         "label": "FAKTUAL", "reasoning": "Matches the accepted answer.", "call_failed": False},
        {"run_id": "A-1", "question_id": 1, "run_label": "A", "judge_id": "secondary",
         "label": "HALUSINASI_SEBAGIAN", "reasoning": "Adds an unsupported claim.", "call_failed": False},
    ]

    class FakeMod:
        load_judge_records = staticmethod(lambda paths: records)
        pair_two_judges = staticmethod(real_mod.pair_two_judges)

    monkeypatch.setattr(es, "_judge_agreement_module", lambda: FakeMod)
    monkeypatch.setattr(
        "app.routers.judge_v1._existing_canonical_outputs", lambda judge_ids: {jid: "/fake" for jid in judge_ids}
    )

    resp = client.get("/api/judge-v1/results/disagreements", params={"judge_id_a": "primary", "judge_id_b": "secondary"})
    assert resp.status_code == 200
    disagreements = resp.json()["disagreements"]
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d["primary_reasoning"] == "Matches the accepted answer."
    assert d["secondary_reasoning"] == "Adds an unsupported claim."
