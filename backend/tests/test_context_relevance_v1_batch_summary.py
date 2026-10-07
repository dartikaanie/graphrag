"""Tests for the context-relevance batch-summary endpoint and its
underlying mapping logic -- the actual fix for "context relevance
table not shown on the batch results page": ctxrel-v1 output records
carry no run_id (they're deduplicated on question_id/answer_id/
text_hash), so a batch's per-run_label summary must be computed by
mapping each run's OWN retrieved_context back to those deduplicated
records, not by looking up run_id directly.
"""

import json

from fastapi.testclient import TestClient

from app.main import app
from app.services import engine_service, history_service

client = TestClient(app)


def _context_item(answer_id, chunk_text):
    return {"answer_id": answer_id, "chunk_text": chunk_text, "trust_weight": 0.9,
            "combined_score": 0.5, "hop": 1, "source_stage": "anchor", "is_accepted": True, "rel_type": "ANSWERS"}


def _write_run(tmp_path, monkeypatch, condition, folder, run_label, config_hash, output_name, retrieved_context_by_qid,
               batch_id=None, status="success"):
    history_path = tmp_path / folder / "logs" / "run_history.jsonl"
    output_path = tmp_path / output_name
    with open(output_path, "w") as f:
        for qid, ctx in retrieved_context_by_qid.items():
            record = {"question_id": qid, "llm_answer": f"a{qid}", "title": "t", "tags": "",
                      "ground_truth_answer": "ref", "config_hash": config_hash}
            if ctx is not None:
                record["retrieved_context"] = ctx
            f.write(json.dumps(record) + "\n")

    extra = {}
    if condition == "C":
        extra["fusion_mode"] = "trust_weighted" if "trust" in run_label else "uniform"
        extra["require_grounding"] = "grounded" in run_label
    elif condition in ("B", "D"):
        extra["require_grounding"] = "grounded" in run_label

    history_record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": condition, "status": status,
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": len(retrieved_context_by_qid),
        "n_processed": len(retrieved_context_by_qid), "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": config_hash,
        **extra,
    }
    if batch_id:
        history_record["batch_id"] = batch_id
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(history_record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)
    run_id = history_service._history_id(condition, history_record)
    return run_id, output_path


def test_batch_summary_404s_for_unknown_batch():
    resp = client.get("/api/context-relevance/results/batch-summary", params={"batch_id": "does-not-exist"})
    assert resp.status_code == 404


def test_batch_summary_maps_deduplicated_records_back_to_each_run(tmp_path, monkeypatch):
    """The core regression test: two runs (B-plain, B-grounded) sharing
    BYTE-IDENTICAL retrieved_context (as verified on the real pilot)
    must produce IDENTICAL metrics, even though the ctxrel-v1 output
    file has no run_id field at all -- the mapping must go through
    (question_id, answer_id, text_hash), not run_id."""
    shared_ctx = {1: [_context_item("A1", "relevant text"), _context_item("A2", "irrelevant text")]}
    run_id_plain, _ = _write_run(tmp_path, monkeypatch, "B", "b_rag", "B-plain", "hashB1", "condition_b_plain.jsonl", shared_ctx, batch_id="batch-1")
    run_id_grounded, _ = _write_run(tmp_path, monkeypatch, "B", "b_rag", "B-grounded", "hashB2", "condition_b_grounded.jsonl", shared_ctx, batch_id="batch-1")
    run_id_a, _ = _write_run(tmp_path, monkeypatch, "A", "a_pure_llm", "A", "hashA", "condition_a.jsonl", {1: None}, batch_id="batch-1")

    ctxrel_out = tmp_path / "ctxrel_v1_primary_ctxrel-v1.jsonl"
    with open(ctxrel_out, "w") as f:
        f.write(json.dumps({"question_id": 1, "answer_id": "A1", "text_hash": engine_service._context_relevance_v1_module()._text_hash("relevant text"),
                             "label": "RELEVANT", "reason": "ok"}) + "\n")
        f.write(json.dumps({"question_id": 1, "answer_id": "A2", "text_hash": engine_service._context_relevance_v1_module()._text_hash("irrelevant text"),
                             "label": "IRRELEVANT", "reason": "ok"}) + "\n")
    monkeypatch.setattr(engine_service, "ctxrel_v1_output_paths",
                         lambda jid: (ctxrel_out, tmp_path / "f.jsonl", tmp_path / "r.jsonl") if jid == "primary"
                         else (tmp_path / "nope.jsonl", tmp_path / "f2.jsonl", tmp_path / "r2.jsonl"))

    resp = client.get("/api/context-relevance/results/batch-summary", params={"batch_id": "batch-1", "judge_id": "primary"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_any_results"] is True

    rows_by_label = {r["run_label"]: r for r in data["rows"]}
    assert rows_by_label["A"]["no_context"] is True
    assert rows_by_label["A"]["metrics"] is None

    plain_metrics = rows_by_label["B-plain"]["metrics"]
    grounded_metrics = rows_by_label["B-grounded"]["metrics"]
    assert plain_metrics == grounded_metrics  # identical shared context -> identical metrics
    assert plain_metrics["n_items_judged"] == 2
    assert plain_metrics["pct_relevant"] == 50.0
    assert plain_metrics["pct_irrelevant"] == 50.0


def test_batch_summary_has_any_results_false_when_no_ctxrel_output(tmp_path, monkeypatch):
    run_id, _ = _write_run(tmp_path, monkeypatch, "B", "b_rag", "B-plain", "hashB1", "condition_b_plain.jsonl",
                            {1: [_context_item("A1", "text")]}, batch_id="batch-2")
    monkeypatch.setattr(engine_service, "ctxrel_v1_output_paths",
                         lambda jid: (tmp_path / "does_not_exist.jsonl", tmp_path / "f.jsonl", tmp_path / "r.jsonl"))

    resp = client.get("/api/context-relevance/results/batch-summary", params={"batch_id": "batch-2", "judge_id": "primary"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_any_results"] is False
    row = next(r for r in data["rows"] if r["run_label"] == "B-plain")
    assert row["metrics"]["n_items_judged"] == 0
