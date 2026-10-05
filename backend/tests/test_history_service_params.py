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

import pytest

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


# ---------------------------------------------------------------------------
# run_label -- 3-axis factorial label (docs/GROUNDING_FACTOR_UI.md), surfaced
# on get_history_detail()'s top level (not inside params -- it's identity,
# not a tunable param). derive_run_label()'s own exhaustive 9-label +
# legacy-inference coverage lives in llm/evaluation/tests/test__run_metadata.py
# (the manually-synced twin of this function) -- these tests only prove the
# BACKEND copy is wired into get_history_detail() correctly.
# ---------------------------------------------------------------------------

def test_history_detail_surfaces_run_label_for_c_uniform_grounded(tmp_path, monkeypatch):
    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "condition": "C",
        "status": "success",
        "provider": "openai", "model": "gpt-4o-mini",
        "n_sample_target": 10, "n_processed": 10, "seed": 42,
        "fusion_mode": "uniform",
        "grounding": "on",
        "output_path": str(tmp_path / "condition_c_label_test.jsonl"),
        "duration_sec": 1.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["run_label"] == "C-uniform-grounded"
    assert detail["grounding"] == "on"
    assert detail["grounding_inferred"] is False
    assert detail["params"]["fusion_mode"] == "uniform"  # full value, unaffected by the shortened label


def test_history_detail_surfaces_run_label_for_b_legacy_fusion_mode_hack(tmp_path, monkeypatch):
    """B briefly recorded its grounding state inside fusion_mode
    ("plain"/"grounded") before the dedicated "grounding" field existed --
    the dashboard must still show the correct run_label for those
    existing entries, marked inferred, and never leak "plain"/"grounded"
    back out as params.fusion_mode."""
    history_path = tmp_path / "b_rag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-09-29T00:00:00+00:00",
        "condition": "B",
        "status": "success",
        "provider": "openai", "model": "gpt-4o-mini",
        "n_sample_target": 10, "n_processed": 10, "seed": 42,
        "fusion_mode": "grounded",
        "output_path": str(tmp_path / "condition_b_legacy_test.jsonl"),
        "duration_sec": 1.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "B", history_path)

    history_id = history_service._history_id("B", record)
    detail = history_service.get_history_detail(history_id)

    assert detail["run_label"] == "B-grounded"
    assert detail["grounding_inferred"] is True
    assert detail["params"]["fusion_mode"] is None


def test_history_detail_refuses_a_deliberately_mixed_config_hash_output_file(tmp_path, monkeypatch):
    """The dashboard-summary half of the 2026-10-05 pilot incident fix:
    get_history_detail() must never silently summarize an output file
    whose records come from two DIFFERENT configs/prompt_versions (e.g. a
    stale pre-v3 file that a v3 run partially resumed into) -- it must
    raise, not average/report across the mix. See
    llm.manifest.assert_single_config_hash / MixedConfigHashError."""
    import sys

    sys.path.insert(0, str(history_service.REPO_ROOT))
    from llm.manifest import MixedConfigHashError

    output_path = tmp_path / "condition_c_mixed.jsonl"
    with open(output_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "cosine_similarity": 0.5, "config_hash": "aaaa111111"}) + "\n")
        f.write(json.dumps({"question_id": 2, "cosine_similarity": 0.6, "config_hash": "bbbb222222"}) + "\n")

    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00",
        "condition": "C",
        "status": "success",
        "provider": "openai", "model": "gpt-4o-mini",
        "n_sample_target": 2, "n_processed": 2, "seed": 42,
        "output_path": str(output_path),
        "duration_sec": 1.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    with pytest.raises(MixedConfigHashError):
        history_service.get_history_detail(history_id)


def test_history_detail_computes_citation_split_from_output_file_when_run_history_lacks_it(tmp_path, monkeypatch):
    """The 2026-10-05 pilot re-run bug: engine_service.py's run_condition_b/
    c/d never called compute_citation_report(), so run_history.jsonl
    entries for dashboard-launched runs have NO pct_citation_valid/
    pct_citation_no_citation/pct_citation_invalid_only/
    fabricated_citation_rate/citation_precision fields at all --
    get_history_detail() must recompute them straight from the output
    file's own has_citation/has_valid_citation/llm_answer/
    retrieved_context fields, with no re-run needed."""
    output_path = tmp_path / "condition_c_citation_test.jsonl"
    with open(output_path, "w") as f:
        # 1 valid citation, 1 with no citation at all -- 50/50 split.
        f.write(json.dumps({
            "question_id": 1, "cosine_similarity": 0.5,
            "llm_answer": "See [SO-123].", "has_citation": True, "has_valid_citation": True,
            "retrieved_context": [{"question_id": 123, "answer_id": 999}],
        }) + "\n")
        f.write(json.dumps({
            "question_id": 2, "cosine_similarity": 0.6,
            "llm_answer": "No citation here.", "has_citation": False, "has_valid_citation": False,
            "retrieved_context": [],
        }) + "\n")

    history_path = tmp_path / "c_graphrag" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00",
        "condition": "C",
        "status": "success",
        "provider": "openai", "model": "gpt-4o-mini",
        "n_sample_target": 2, "n_processed": 2, "seed": 42,
        "output_path": str(output_path),
        "duration_sec": 1.0,
        # no pct_citation_* / fabricated_citation_rate / citation_precision
        # fields at all -- exactly what a dashboard-launched pre-fix run
        # looks like.
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "C", history_path)

    history_id = history_service._history_id("C", record)
    detail = history_service.get_history_detail(history_id)

    summary = detail["summary"]
    assert summary["pct_citation_valid"] == 50.0
    assert summary["pct_citation_no_citation"] == 50.0
    assert summary["pct_citation_invalid_only"] == 0.0
    assert summary["fabricated_citation_rate"] == 0.0
    assert summary["citation_precision"] is not None


def test_history_detail_skips_citation_split_for_condition_a_records_without_has_citation(tmp_path, monkeypatch):
    """Condition A never has the has_citation concept at all -- the
    fallback must not compute a misleading 100%-no-citation split for it."""
    output_path = tmp_path / "condition_a_no_citation_concept.jsonl"
    with open(output_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "cosine_similarity": 0.5, "llm_answer": "An answer."}) + "\n")

    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00",
        "condition": "A",
        "status": "success",
        "provider": "openai", "model": "gpt-4o-mini",
        "n_sample_target": 1, "n_processed": 1, "seed": 42,
        "output_path": str(output_path),
        "duration_sec": 1.0,
    }
    _write_run_history(history_path, record)
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    history_id = history_service._history_id("A", record)
    detail = history_service.get_history_detail(history_id)

    assert "pct_citation_valid" not in detail["summary"]
