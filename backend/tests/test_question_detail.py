"""Tests for engine_service.build_question_detail() -- the per-question
detail export's data assembly (Step 2.3). No LLM calls; tiny fixtures
for the generation run, judge-v1 output, and context-relevance output."""

import json

import pytest

from app.services import engine_service, history_service


def _write_generation_run(tmp_path, monkeypatch, condition="C", config_hash="hashC"):
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    output_path = tmp_path / "condition_c_x.jsonl"
    record = {
        "question_id": 1, "title": "How do I do X?", "tags": "<python>",
        "ground_truth_answer": "<p>Use the frobnicate() function.</p>",
        "llm_answer": "You should use frobnicate() [SO-1].",
        "has_citation": True, "has_valid_citation": True,
        "retrieved_context": [
            {"answer_id": "A1", "question_id": 99, "chunk_text": "frobnicate() docs here",
             "is_accepted": True, "trust_weight": 0.9, "hop": 1, "source_stage": "anchor", "combined_score": 0.5},
        ],
        "config_hash": config_hash,
    }
    with open(output_path, "w") as f:
        f.write(json.dumps(record) + "\n")
    history_record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": condition, "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 1, "n_processed": 1, "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": config_hash,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)
    run_id = history_service._history_id(condition, history_record)
    return run_id


def test_build_question_detail_basic_fields(tmp_path, monkeypatch):
    run_id = _write_generation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(engine_service, "_questions_parquet", lambda: "fake_parquet")

    mod = engine_service._context_relevance_v1_module()
    monkeypatch.setattr(mod, "load_question_bodies", lambda parquet, ids: {}, raising=False)
    import _judge_common
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {})

    result = engine_service.build_question_detail(run_id, [1])
    assert len(result) == 1
    q = result[0]
    assert q["question_id"] == 1
    assert q["title"] == "How do I do X?"
    assert q["tags"] == "<python>"
    assert "frobnicate" in q["reference_answer"]
    assert q["candidate_raw"] == "You should use frobnicate() [SO-1]."
    assert "[SO-1]" not in q["candidate_blinded"]  # citation marker stripped
    assert q["citation_outcome"] == "valid"
    assert len(q["context_items"]) == 1
    assert q["context_items"][0]["answer_id"] == "A1"
    assert q["context_items"][0]["rank"] == 1


def test_build_question_detail_truncates_long_text(tmp_path, monkeypatch):
    run_id = _write_generation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(engine_service, "_questions_parquet", lambda: "fake_parquet")
    import _judge_common
    long_body = "x" * 2000
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {1: f"<p>{long_body}</p>"})

    result = engine_service.build_question_detail(run_id, [1])
    q = result[0]
    assert len(q["body"]) <= 1500 + len(" … [truncated]")
    assert q["body"].endswith("[truncated]")


def test_build_question_detail_unknown_question_id_reports_error_not_crash(tmp_path, monkeypatch):
    run_id = _write_generation_run(tmp_path, monkeypatch)
    import _judge_common
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {})
    monkeypatch.setattr(engine_service, "_questions_parquet", lambda: "fake_parquet")

    result = engine_service.build_question_detail(run_id, [1, 99999])
    assert result[0]["question_id"] == 1
    assert "error" in result[1]
    assert result[1]["question_id"] == 99999


def test_build_question_detail_raises_for_unknown_run():
    with pytest.raises(ValueError):
        engine_service.build_question_detail("DOES-NOT-EXIST", [1])


def test_build_question_detail_never_leaks_api_keys_or_headers(tmp_path, monkeypatch):
    """Step 2.4: exports must never include API keys or request headers
    -- the assembled detail dict must not surface request_params/
    api_key/authorization fields even though the underlying judge-v1
    records carry a `request_params` field with the model/temperature
    actually sent."""
    run_id = _write_generation_run(tmp_path, monkeypatch)
    import _judge_common
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {})
    monkeypatch.setattr(engine_service, "_questions_parquet", lambda: "fake_parquet")

    jv1_out = tmp_path / "jv1_primary_fake.jsonl"
    jv1_out.parent.mkdir(parents=True, exist_ok=True)
    with open(jv1_out, "w") as f:
        f.write(json.dumps({
            "run_id": run_id, "question_id": 1, "judge_id": "primary", "label": "FAKTUAL",
            "derived_label": "FAKTUAL", "consistent": True, "claims": [], "reasoning": "ok",
            "response_model": "fake-model",
            "request_params": {"model": "fake-model", "temperature": 0.0},
            "attempts": [{"attempt": 1, "error": None, "status_code": None}],
        }) + "\n")
    monkeypatch.setattr(
        engine_service, "judge_v1_output_paths",
        lambda jid: (jv1_out, tmp_path / "f.jsonl", tmp_path / "r.jsonl") if jid == "primary"
        else (tmp_path / "nope.jsonl", tmp_path / "f2.jsonl", tmp_path / "r2.jsonl"),
    )

    result = engine_service.build_question_detail(run_id, [1])
    q = result[0]
    assert "request_params" not in q["judges"].get("primary", {})
    assert "api_key" not in json.dumps(q)
    assert "authorization" not in json.dumps(q).lower()
