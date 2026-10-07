"""Tests for engine_service.plan_context_relevance_v1() -- condition A
exclusion, cross-run dedup, and the plan table's per-judge estimates.
No LLM calls; tiny fixture generation run_history + output files stand
in for real pilot runs.
"""

import json

import pytest

from app.services import engine_service, history_service

engine_service._context_relevance_v1_module()  # self-heals sys.path for _run_metadata below
import _run_metadata  # noqa: E402


def _context_item(answer_id, chunk_text):
    return {"answer_id": answer_id, "chunk_text": chunk_text, "trust_weight": 0.9,
            "combined_score": 0.5, "hop": 1, "source_stage": "anchor", "is_accepted": True, "rel_type": "ANSWERS"}


def _write_run(tmp_path, monkeypatch, condition, folder, config_hash, output_name, retrieved_context_by_qid):
    history_path = tmp_path / folder / "logs" / "run_history.jsonl"
    output_path = tmp_path / output_name
    with open(output_path, "w") as f:
        for qid, ctx in retrieved_context_by_qid.items():
            record = {"question_id": qid, "llm_answer": f"a{qid}", "title": "t", "tags": "",
                      "ground_truth_answer": "ref", "config_hash": config_hash}
            if ctx is not None:
                record["retrieved_context"] = ctx
            f.write(json.dumps(record) + "\n")
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": condition, "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": len(retrieved_context_by_qid),
        "n_processed": len(retrieved_context_by_qid), "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": config_hash,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, condition, history_path)
    run_id = history_service._history_id(condition, record)
    return run_id, output_path


def test_plan_refuses_condition_a_runs(tmp_path, monkeypatch):
    run_id, _ = _write_run(tmp_path, monkeypatch, "A", "a_pure_llm", "hashA", "condition_a_x.jsonl", {1: None})

    result = engine_service.plan_context_relevance_v1([run_id], ["primary"], {})
    assert result["rows"][0]["refused_reason"].startswith("condition A has no retrieved_context")


def test_plan_dedups_identical_context_across_two_runs(tmp_path, monkeypatch):
    """A plain/grounded pair sharing byte-identical retrieved_context
    must contribute the dedup count ONCE, not twice -- the whole point
    of Step 3's cross-run dedup."""
    shared_ctx = {1: [_context_item("A1", "x"), _context_item("A2", "y")]}
    run_id_1, _ = _write_run(tmp_path, monkeypatch, "C", "c_graphrag", "hashC1", "condition_c_x1.jsonl", shared_ctx)
    run_id_2, _ = _write_run(tmp_path, monkeypatch, "C", "c_graphrag", "hashC2", "condition_c_x2.jsonl", shared_ctx)

    result = engine_service.plan_context_relevance_v1([run_id_1, run_id_2], ["primary"], {})
    assert result["totals"]["unique_items"] == 2
    assert result["totals"]["per_judge"]["primary"]["items_to_judge"] == 2


def test_plan_reports_per_run_context_item_counts(tmp_path, monkeypatch):
    run_id, _ = _write_run(
        tmp_path, monkeypatch, "C", "c_graphrag", "hashC", "condition_c_x.jsonl",
        {1: [_context_item("A1", "x")], 2: [_context_item("A2", "y"), _context_item("A3", "z")]},
    )
    result = engine_service.plan_context_relevance_v1([run_id], ["primary"], {})
    row = next(r for r in result["rows"] if "refused_reason" not in r)
    assert row["n_context_items"] == 3


def test_plan_cost_unknown_for_unregistered_judge_id(tmp_path, monkeypatch):
    run_id, _ = _write_run(tmp_path, monkeypatch, "C", "c_graphrag", "hashC", "condition_c_x.jsonl",
                            {1: [_context_item("A1", "x")]})
    result = engine_service.plan_context_relevance_v1([run_id], ["not-a-real-judge"], {})
    pj = result["totals"]["per_judge"]["not-a-real-judge"]
    assert pj["cost_unknown"] is True
    assert pj["est_cost_usd"] is None
