"""Router-level sanity for /api/judge-v2/* -- mirrors
test_judge_v1_router.py. Also covers version-comparison/reference-
conflicts, which have no v1 equivalent."""

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_registry_endpoint_same_as_v1():
    resp = client.get("/api/judge-v2/registry")
    assert resp.status_code == 200
    roles = [j["role"] for j in resp.json()["judges"]]
    assert roles.count("primary") == 1
    assert roles.count("secondary") == 1


def test_jobs_endpoint_returns_200():
    resp = client.get("/api/judge-v2/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()


def test_plan_endpoint_rejects_empty_run_ids():
    resp = client.post("/api/judge-v2/plan", json={"run_ids": [], "judge_ids": ["primary"], "workers": {}})
    assert resp.status_code == 400


def test_plan_endpoint_rejects_empty_judge_ids():
    resp = client.post("/api/judge-v2/plan", json={"run_ids": ["A-fake"], "judge_ids": [], "workers": {}})
    assert resp.status_code == 400


def test_status_endpoint_404s_for_unknown_run():
    resp = client.get("/api/judge-v2/status/does-not-exist")
    assert resp.status_code == 404


def test_cancel_endpoint_404s_for_unknown_run():
    resp = client.post("/api/judge-v2/cancel/does-not-exist")
    assert resp.status_code == 404


def test_version_comparison_rejects_empty_run_ids():
    resp = client.get("/api/judge-v2/results/version-comparison", params={"run_ids": ""})
    assert resp.status_code == 400


def test_reference_conflicts_rejects_empty_run_ids():
    resp = client.get("/api/judge-v2/results/reference-conflicts", params={"run_ids": ""})
    assert resp.status_code == 400


def test_version_comparison_transition_matrix_and_kappas(tmp_path, monkeypatch):
    from app.services import engine_service

    v1_out = tmp_path / "jv1_primary.jsonl"
    with open(v1_out, "w") as f:
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary", "run_label": "A",
                             "label": "HALUSINASI_PENUH", "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary", "run_label": "A",
                             "label": "ABSTAIN", "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "secondary", "run_label": "A",
                             "label": "FAKTUAL", "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "secondary", "run_label": "A",
                             "label": "ABSTAIN", "prompt_version": "judge-v1", "blinding_version": "blind-v2"}) + "\n")

    v2_out = tmp_path / "jv2_primary.jsonl"
    with open(v2_out, "w") as f:
        # Q1: v1 said HALUSINASI_PENUH (the non-answer-judged-as-hallucination
        # bug), v2 correctly says ABSTAIN -- the whole point of judge-v2.
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "primary", "run_label": "A",
                             "label": "ABSTAIN", "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "primary", "run_label": "A",
                             "label": "FAKTUAL", "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 1, "judge_id": "secondary", "run_label": "A",
                             "label": "ABSTAIN", "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")
        f.write(json.dumps({"run_id": "r1", "question_id": 2, "judge_id": "secondary", "run_label": "A",
                             "label": "FAKTUAL", "prompt_version": "judge-v2", "blinding_version": "blind-v2"}) + "\n")

    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (v1_out, tmp_path / "f1.jsonl", tmp_path / "r1.jsonl") if jid == "primary"
                         else (v1_out.with_name("jv1_secondary.jsonl"), tmp_path / "f2.jsonl", tmp_path / "r2.jsonl"))
    # secondary records are in the SAME v1_out file in this fixture -- point both at it
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (v1_out, tmp_path / "f1.jsonl", tmp_path / "r1.jsonl"))
    monkeypatch.setattr(engine_service, "judge_v2_output_paths",
                         lambda jid: (v2_out, tmp_path / "f3.jsonl", tmp_path / "r3.jsonl"))

    resp = client.get("/api/judge-v2/results/version-comparison",
                       params={"run_ids": "r1", "judge_id_a": "primary", "judge_id_b": "secondary"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["transition_matrix"]["A"]["HALUSINASI_PENUH"]["ABSTAIN"] == 1
    assert data["transition_matrix"]["A"]["ABSTAIN"]["FAKTUAL"] == 1
    # Q1 is ABSTAIN/ABSTAIN (excluded from kappa by design), Q2 is the only ordinal pair
    assert data["kappa_v2"]["n_pairs"] == 2
    assert data["kappa_v2"]["kappas"]["n_used"] == 1


def test_reference_conflicts_extracts_flagged_claims(tmp_path, monkeypatch):
    from app.services import engine_service

    v2_out = tmp_path / "jv2_primary.jsonl"
    with open(v2_out, "w") as f:
        f.write(json.dumps({
            "run_id": "r1", "question_id": 1, "run_label": "A",
            "claims": [
                {"claim": "XFINIUM.PDF supports UWP", "verdict": "SUPPORTED",
                 "reference_conflict": True, "evidence": "well-known library"},
                {"claim": "other claim", "verdict": "SUPPORTED", "reference_conflict": False, "evidence": "e"},
            ],
        }) + "\n")
    monkeypatch.setattr(engine_service, "judge_v2_output_paths",
                         lambda jid: (v2_out, tmp_path / "f.jsonl", tmp_path / "r.jsonl") if jid == "primary"
                         else (tmp_path / "nope.jsonl", tmp_path / "f2.jsonl", tmp_path / "r2.jsonl"))

    resp = client.get("/api/judge-v2/results/reference-conflicts", params={"run_ids": "r1", "judge_ids": "primary"})
    assert resp.status_code == 200
    conflicts = resp.json()["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["claim"] == "XFINIUM.PDF supports UWP"
    assert conflicts[0]["question_id"] == 1
