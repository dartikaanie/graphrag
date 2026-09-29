"""Phase 1.5 -- round-trip tests for the sampling/run parameters
history_service.get_history_detail() exposes in its `params` dict
(oversample_pool, n_candidates_after_token_filter, log_full_candidates,
index_pool, and the existing C/D fields). Covers two things explicitly
required by the task:

  1. A NEW run_history.jsonl record (with every new field present) round-
     trips every one of those fields into `params` unchanged.
  2. An OLD run_history.jsonl record (written before this field set
     existed) yields None for the new fields -- never a guessed value,
     per the "backward compatibility" hard constraint.

No LLM calls, no real Neo4j/DuckDB -- history_service reads run_history
.jsonl files directly off disk, so these tests just write a fixture file
to tmp_path and point history_service.HISTORY_PATHS at it.
"""

import json

from app.services import history_service


def _write_run_history(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(record) + "\n")


def test_new_record_round_trips_every_new_field(tmp_path, monkeypatch):
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "condition": "C",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "oversample_pool": 1536,
        "n_candidates_after_token_filter": 1204,
        "log_full_candidates": True,
        "top_k": 5,
        "n_anchor": 3,
        "n_semantic_expansion": 3,
        "fusion_mode": "trust_weighted",
        "fusion_w_path_trust": 0.7,
        "fusion_w_answer_intrinsic_trust": 0.3,
        "semantic_expansion_trust_cap": 0.4,
        "require_grounding": True,
        "enable_semantic_expansion": True,
        "output_path": str(tmp_path / "condition_c_test.jsonl"),
        "duration_sec": 12.3,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    detail = history_service.get_history_detail(history_id)

    assert detail is not None
    params = detail["params"]
    assert params["oversample_pool"] == 1536
    assert params["n_candidates_after_token_filter"] == 1204
    assert params["log_full_candidates"] is True
    assert params["fusion_mode"] == "trust_weighted"
    assert params["fusion_w_path_trust"] == 0.7
    assert params["fusion_w_intrinsic"] == 0.3  # mapped from fusion_w_answer_intrinsic_trust
    assert params["semantic_expansion_trust_cap"] == 0.4
    assert params["require_grounding"] is True
    assert params["enable_semantic_expansion"] is True
    assert params["n_anchor"] == 3
    assert params["n_semantic_expansion"] == 3


def test_new_record_b_condition_index_pool_and_log_full_candidates(tmp_path, monkeypatch):
    history_path = tmp_path / "b_rag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "condition": "B",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "oversample_pool": 1536,
        "n_candidates_after_token_filter": 980,
        "require_citation": True,
        "log_full_candidates": True,
        "index_pool": 8000,
        "top_k": 5,
        "output_path": str(tmp_path / "condition_b_test.jsonl"),
        "duration_sec": 8.1,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "B", history_path)

    history_id = history_service._history_id("B", record)
    detail = history_service.get_history_detail(history_id)

    params = detail["params"]
    assert params["oversample_pool"] == 1536
    assert params["n_candidates_after_token_filter"] == 980
    assert params["log_full_candidates"] is True
    assert params["index_pool"] == 8000
    assert params["require_citation"] is True


def test_old_record_without_new_fields_yields_none_not_guessed(tmp_path, monkeypatch):
    """A pre-Phase-1 run_history.jsonl record (exactly the shape the five
    n=10 pilot runs had) must show None for every new field -- never 0,
    never False-as-a-guess, never silently omitted from `params`."""
    history_path = tmp_path / "d_lightrag" / "logs" / "run_history.jsonl"
    old_record = {
        "run_started_at": "2026-09-28T18:27:19.576518+00:00",
        "condition": "D",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "top_k": 5,
        "n_low_level": 3,
        "n_high_level": 3,
        "require_grounding": True,
        "output_path": str(tmp_path / "condition_d_test.jsonl"),
        "duration_sec": 71.1,
    }
    _write_run_history(history_path, old_record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "D", history_path)

    history_id = history_service._history_id("D", old_record)
    detail = history_service.get_history_detail(history_id)

    params = detail["params"]
    assert params["oversample_pool"] is None
    assert params["n_candidates_after_token_filter"] is None
    assert params["log_full_candidates"] is None
    assert params["index_pool"] is None
    # Fields the old record DID have still come through unaffected.
    assert params["top_k"] == 5
    assert params["n_low_level"] == 3
    assert params["require_grounding"] is True
    # run_started_at (2026-09-28T18:27:19Z) predates PROMPT_V2_CUTOFF
    # (2026-09-28T19:06:21Z) -- reliably inferable as v1, for BOTH fields
    # since this is Condition D.
    assert params["prompt_version"] == "v1 (inferred)"
    assert params["d_retrieval_version"] == "v1 (inferred)"


def test_prompt_version_present_on_new_record_is_used_verbatim(tmp_path, monkeypatch):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "prompt_version": "v2",
        "output_path": str(tmp_path / "condition_a_test.jsonl"),
        "duration_sec": 5.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    history_id = history_service._history_id("A", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["params"]["prompt_version"] == "v2"
    # d_retrieval_version is D-only -- always None for other conditions,
    # never inferred or defaulted.
    assert detail["params"]["d_retrieval_version"] is None


def test_prompt_version_missing_after_cutoff_shows_dash_not_a_guess(tmp_path, monkeypatch):
    """A record dated AFTER PROMPT_V2_CUTOFF but still missing
    prompt_version is genuinely ambiguous -- must be None ("--" in the UI),
    never inferred as either v1 or v2."""
    history_path = tmp_path / "b_rag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-12-01T00:00:00+00:00",
        "condition": "B",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "output_path": str(tmp_path / "condition_b_test.jsonl"),
        "duration_sec": 5.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "B", history_path)

    history_id = history_service._history_id("B", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["params"]["prompt_version"] is None


def test_c_retrieval_version_inferred_v1_before_fix_cutoff_c_only(tmp_path, monkeypatch):
    """c_retrieval_version is C-only (None for every other condition, same
    as d_retrieval_version is D-only) and infers 'v1 (inferred)' for a C
    record dated before C_D_RETRIEVAL_FIX_CUTOFF that predates the field."""
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-28T18:28:30.978892+00:00",  # the real C-079efd82cc pilot run
        "condition": "C",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "fusion_mode": "trust_weighted",
        "output_path": str(tmp_path / "condition_c_test.jsonl"),
        "duration_sec": 58.1,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["params"]["c_retrieval_version"] == "v1 (inferred)"
    assert detail["params"]["d_retrieval_version"] is None  # C-condition record -- D field always None


def test_c_retrieval_version_v2_recorded_verbatim_on_new_record(tmp_path, monkeypatch):
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "condition": "C",
        "status": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "n_sample_target": 10,
        "n_processed": 10,
        "seed": 42,
        "c_retrieval_version": "v2",
        "output_path": str(tmp_path / "condition_c_test2.jsonl"),
        "duration_sec": 58.1,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["params"]["c_retrieval_version"] == "v2"
