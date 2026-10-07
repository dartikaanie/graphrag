"""Router-level sanity for /api/context-relevance/* -- plan validation
and the results endpoints never leak secrets, never crash on an empty
dataset."""

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_plan_endpoint_rejects_empty_run_ids():
    resp = client.post("/api/context-relevance/plan", json={"run_ids": [], "judge_ids": ["primary"], "workers": {}})
    assert resp.status_code == 400


def test_plan_endpoint_rejects_empty_judge_ids():
    resp = client.post("/api/context-relevance/plan", json={"run_ids": ["A-fake"], "judge_ids": [], "workers": {}})
    assert resp.status_code == 400


def test_status_endpoint_404s_for_unknown_run():
    resp = client.get("/api/context-relevance/status/does-not-exist")
    assert resp.status_code == 404


def test_cancel_endpoint_404s_for_unknown_run():
    resp = client.post("/api/context-relevance/cancel/does-not-exist")
    assert resp.status_code == 404


def test_results_summary_404s_for_unknown_run():
    resp = client.get("/api/context-relevance/results/summary", params={"run_id": "does-not-exist"})
    assert resp.status_code == 404


def test_results_link_to_outcomes_404s_for_unknown_run():
    resp = client.get("/api/context-relevance/results/link-to-outcomes", params={"run_id": "does-not-exist"})
    assert resp.status_code == 404


def test_link_to_outcomes_returns_aggregated_tables(tmp_path, monkeypatch):
    from app.services import engine_service, history_service

    output_path = tmp_path / "condition_c_run.jsonl"
    with open(output_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "retrieved_context": [
            {"answer_id": "A1", "chunk_text": "relevant text"},
        ]}) + "\n")
        f.write(json.dumps({"question_id": 2, "retrieved_context": [
            {"answer_id": "A2", "chunk_text": "irrelevant text"},
        ]}) + "\n")

    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "C", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 2, "n_processed": 2, "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": "h1",
        "fusion_mode": "uniform", "require_grounding": True,
    }
    history_path.write_text(json.dumps(history_record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)
    run_id = history_service._history_id("C", history_record)

    ctxrel_mod = engine_service._context_relevance_v1_module()
    ctxrel_out = tmp_path / "ctxrel_v1_primary_ctxrel-v1.jsonl"
    with open(ctxrel_out, "w") as f:
        f.write(json.dumps({"question_id": 1, "answer_id": "A1", "text_hash": ctxrel_mod._text_hash("relevant text"), "label": "RELEVANT"}) + "\n")
        f.write(json.dumps({"question_id": 2, "answer_id": "A2", "text_hash": ctxrel_mod._text_hash("irrelevant text"), "label": "IRRELEVANT"}) + "\n")
    monkeypatch.setattr(engine_service, "ctxrel_v1_output_paths",
                         lambda jid: (ctxrel_out, tmp_path / "f.jsonl", tmp_path / "r.jsonl") if jid == "primary"
                         else (tmp_path / "nope.jsonl", tmp_path / "f2.jsonl", tmp_path / "r2.jsonl"))

    jv1_out = tmp_path / "jv1_primary.jsonl"
    with open(jv1_out, "w") as f:
        f.write(json.dumps({"run_id": run_id, "question_id": 1, "label": "FAKTUAL"}) + "\n")
        f.write(json.dumps({"run_id": run_id, "question_id": 2, "label": "ABSTAIN"}) + "\n")
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (jv1_out, tmp_path / "jf.jsonl", tmp_path / "jr.jsonl") if jid == "primary"
                         else (tmp_path / "jnope.jsonl", tmp_path / "jf2.jsonl", tmp_path / "jr2.jsonl"))

    resp = client.get("/api/context-relevance/results/link-to-outcomes", params={"run_id": run_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tables"]["label_x_relevant"]["FAKTUAL"] == {"has_relevant": 1, "no_relevant": 0}
    assert data["tables"]["label_x_relevant"]["ABSTAIN"] == {"has_relevant": 0, "no_relevant": 1}
    # run_label is grounded (require_grounding=True) -- abstain table present
    assert data["tables"]["abstain_x_relevant"] is not None
    assert data["tables"]["abstain_x_relevant"]["no_relevant"]["abstain"] == 1


def test_jobs_endpoint_returns_200():
    resp = client.get("/api/context-relevance/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()
