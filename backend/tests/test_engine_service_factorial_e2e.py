"""End-to-end regression test for the 2026-10-05 pilot incident: all 9
factorial configs (A + B-plain/grounded + C-uniform/trust x plain/grounded
+ D-plain/grounded) silently produced ZERO records via a deterministic
output-filename collision with stale pre-existing files (see
llm.manifest.read_already_done's docstring and build_output_path's for the
two-part fix). This test calls the SAME entry points the dashboard uses
(engine_service.run_condition_a/b/c/d) with a mocked LLM client, n_sample=2,
and asserts each of the 9 configs actually produces records, a
run_history.jsonl entry, and a manifest with status="completed" -- so a
regression of either half of the fix (the prompt_version gate on resume, or
a filename that doesn't disambiguate a run from its siblings) is caught
here instead of during a real pilot.

Neo4j/FAISS/DuckDB/the real embedding model are all replaced with tiny
fakes -- this test makes no network/GPU calls and reads no real repo data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.services import engine_service, run_registry


class FakeEmbedModel:
    """encode() always returns the SAME fixed vector -- similarity scoring
    isn't what this test is checking, only that generation completes."""

    def encode(self, text, convert_to_tensor=False, **kwargs):
        vec = torch.tensor([1.0, 0.0, 0.0])
        return vec if convert_to_tensor else vec.numpy()


class FakeDriver:
    def close(self):
        pass


def _fake_call_llm_fn(client, messages, model):
    return {
        "content": "A plain answer with no citation tokens.",
        "response_id": "resp_fake_1",
        "response_model": model,
        "response_created": 1700000000,
        "system_fingerprint": "fp_fake",
        "finish_reason": "stop",
        "usage_full": {"prompt_tokens": 10, "completion_tokens": 5},
        "request_params": {"model": model},
    }


def _sample_candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "Id": [101, 102],
        "Title": ["How do I do X?", "Why does Y happen?"],
        "Body": ["Body text for X.", "Body text for Y."],
        "Tags": ["python", "python"],
        "AcceptedAnswerId": [9101, 9102],
        "n_tokens": [50, 50],
        "ViewCount": [10, 20],
        "Score": [1, 2],
    })


def _accepted_answers() -> pd.DataFrame:
    return pd.DataFrame({
        "AcceptedAnswerId": [9101, 9102],
        "AcceptedAnswerBody": ["Reference answer for X.", "Reference answer for Y."],
    })


def _patch_sampling(monkeypatch, cond):
    """Bypasses DuckDB/parquet entirely with a fixed 2-question sample --
    shared shape across A/B/C/D (all four modules expose the identical
    get_candidate_questions/filter_by_token_limit/sample_questions/
    get_accepted_answers signatures)."""
    monkeypatch.setattr(cond, "get_candidate_questions", lambda *a, **k: _sample_candidates())
    monkeypatch.setattr(cond, "filter_by_token_limit", lambda df, *a, **k: df)
    monkeypatch.setattr(cond, "sample_questions", lambda df, n, seed: df.head(n).reset_index(drop=True))
    monkeypatch.setattr(cond, "get_accepted_answers", lambda con, path, ids: _accepted_answers())
    monkeypatch.setattr(cond, "get_llm_client", lambda *a, **k: (object(), _fake_call_llm_fn))


def _redirect_io(monkeypatch, tmp_path: Path, cond, log_subdir: str):
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cond, "LOG_DIR", tmp_path / log_subdir)
    monkeypatch.setattr(run_registry, "RUNS_DIR", tmp_path / "_runs")


def _run_and_collect(params: dict, run_fn, cond, run_label: str, stale_path: Path | None = None):
    """`stale_path`: when given, a pre-existing file at the OLD (pre-fix)
    naming convention -- no prompt_version/config_hash in the filename --
    already populated with 2 "stale" records for the SAME question ids
    (101/102) under a DIFFERENT (older) config/prompt_version. Asserts the
    new run's output_path is a DIFFERENT file (the filename fix), that the
    stale file is left completely untouched (never appended to, per
    "never overwrite old data"), and that the new file contains EXACTLY
    this run's own 2 records sharing ONE config_hash (the resume-gate fix
    -- see llm.manifest.read_already_done/compute_config_hash)."""
    stale_lines_before = stale_path.read_text().splitlines() if stale_path and stale_path.exists() else None

    run_id = run_registry.create_run(params["condition"], params["mode"], params)
    run_fn(run_id, params)
    state = run_registry.get_run(run_id)
    assert state is not None and state["status"] == "completed", (
        f"{run_label}: run ended with status={state['status'] if state else None!r} "
        f"error={state.get('error') if state else None!r}"
    )

    history_path = cond.LOG_DIR / "run_history.jsonl"
    assert history_path.exists(), f"{run_label}: no run_history.jsonl written at all"
    history_records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    assert history_records, f"{run_label}: run_history.jsonl exists but has no entries"
    last = history_records[-1]

    assert last["status"] == "success", (
        f"{run_label}: expected status=success, got {last['status']!r} "
        f"(n_processed={last.get('n_processed')}) -- this is the exact pilot no-op bug if 0"
    )
    assert last["n_processed"] == 2, f"{run_label}: expected 2 processed records, got {last.get('n_processed')}"

    output_path = Path(last["output_path"])
    assert output_path.exists(), f"{run_label}: output_path from run_history doesn't exist"
    output_lines = [l for l in output_path.read_text().splitlines() if l.strip()]
    assert len(output_lines) == 2, f"{run_label}: expected 2 lines in output file, got {len(output_lines)}"

    output_records = [json.loads(l) for l in output_lines]
    hashes = {r.get("config_hash") for r in output_records}
    assert len(hashes) == 1 and None not in hashes, (
        f"{run_label}: expected exactly 1 non-null config_hash across the new file's records, got {hashes}"
    )

    if stale_path is not None:
        assert output_path != stale_path, (
            f"{run_label}: new run wrote to the SAME path as the stale v2-style file -- "
            f"the filename collision fix regressed"
        )
        assert stale_path.read_text().splitlines() == stale_lines_before, (
            f"{run_label}: the stale v2-style file was modified -- it must be left untouched"
        )

    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    assert manifest_path.exists(), f"{run_label}: no manifest written"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed", f"{run_label}: manifest status={manifest['status']!r}, expected completed"
    assert manifest["run_label"] == run_label, f"{run_label}: manifest run_label={manifest['run_label']!r}"
    assert manifest["config_hash"] == next(iter(hashes)), f"{run_label}: manifest config_hash doesn't match records"

    return state


def _write_stale_v2_file(path: Path) -> None:
    """A file at the OLD (pre-fix) naming convention, already "complete"
    for the SAME 2 question ids this test's sample uses -- pre-dates
    prompt_version/config_hash entirely (no such fields), same shape as
    the REAL Sept-2026 files that caused the 2026-10-05 pilot incident."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid in (101, 102):
            f.write(json.dumps({"question_id": qid, "llm_answer": "stale v2 answer", "cosine_similarity": 0.1}) + "\n")


@pytest.fixture
def common_patches(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_service, "_embed_model", lambda: FakeEmbedModel())
    monkeypatch.setattr(engine_service, "_embed_model_cpu", lambda: FakeEmbedModel())
    monkeypatch.setattr(
        engine_service, "_faiss_cache",
        lambda kg_dir_str: (None, np.array([101, 102]), np.zeros((2, 3), dtype="float32"), {101: 0, 102: 1}),
    )
    return tmp_path


def test_condition_a_pilot_config_produces_records_history_and_manifest(common_patches, monkeypatch):
    tmp_path = common_patches
    cond = engine_service._condition_a_module()
    _patch_sampling(monkeypatch, cond)
    _redirect_io(monkeypatch, tmp_path, cond, "a_logs")

    params = {"condition": "A", "mode": "batch", "provider": "openai", "model": "gpt-4o-mini",
              "n_sample": 2, "seed": 42}
    stale_path = tmp_path / "llm" / "a_pure_llm" / "results" / "condition_a_openai_gpt-4o-mini_n2_seed42.jsonl"
    _write_stale_v2_file(stale_path)
    _run_and_collect(params, engine_service.run_condition_a, cond, "A", stale_path=stale_path)


@pytest.mark.parametrize("require_grounding,run_label", [(False, "B-plain"), (True, "B-grounded")])
def test_condition_b_pilot_configs_produce_records_history_and_manifest(
    common_patches, monkeypatch, require_grounding, run_label,
):
    tmp_path = common_patches
    cond = engine_service._condition_b_module()
    _patch_sampling(monkeypatch, cond)
    _redirect_io(monkeypatch, tmp_path, cond, "b_logs")
    monkeypatch.setattr(cond, "build_retrieval_corpus", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cond, "chunk_corpus", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(cond, "build_or_load_faiss_index", lambda *a, **k: (None, pd.DataFrame()))
    monkeypatch.setattr(cond, "retrieve_context", lambda *a, **k: [])

    params = {"condition": "B", "mode": "batch", "provider": "openai", "model": "gpt-4o-mini",
              "n_sample": 2, "seed": 42, "top_k": 5,
              "require_citation": True, "require_grounding": require_grounding}
    # Pre-fix, B's filename never encoded grounding at all -- both B-plain
    # and B-grounded shared this SAME stale name (the structural half of
    # the bug, on top of the missing prompt_version/hash gate).
    stale_path = tmp_path / "llm" / "b_rag" / "results" / "condition_b_openai_gpt-4o-mini_n2_seed42.jsonl"
    _write_stale_v2_file(stale_path)
    _run_and_collect(params, engine_service.run_condition_b, cond, run_label, stale_path=stale_path)


@pytest.mark.parametrize("fusion_mode,require_grounding,run_label", [
    ("uniform", False, "C-uniform-plain"),
    ("uniform", True, "C-uniform-grounded"),
    ("trust_weighted", False, "C-trust-plain"),
    ("trust_weighted", True, "C-trust-grounded"),
])
def test_condition_c_pilot_configs_produce_records_history_and_manifest(
    common_patches, monkeypatch, fusion_mode, require_grounding, run_label,
):
    tmp_path = common_patches
    cond = engine_service._condition_c_module()
    _patch_sampling(monkeypatch, cond)
    _redirect_io(monkeypatch, tmp_path, cond, "c_logs")
    monkeypatch.setattr(cond, "get_all_answer_ids_for_questions", lambda *a, **k: {})
    monkeypatch.setattr(cond, "connect_neo4j", lambda *a, **k: (FakeDriver(), "neo4j"))
    monkeypatch.setattr(cond, "anchor_via_vector_search", lambda *a, **k: [(101, 0.9)])
    monkeypatch.setattr(cond, "traverse_graph", lambda *a, **k: [{
        "answer_id": 9101, "via_question_id": 101, "via_question_title": "How do I do X?",
        "answer_body": "Reference answer for X.", "edge_weight": 0.8, "answer_trust_score": 0.5,
        "hop": 1, "rel_type": "ANSWERED", "is_accepted": True,
    }])

    params = {"condition": "C", "mode": "batch", "provider": "openai", "model": "gpt-4o-mini",
              "n_sample": 2, "seed": 42, "top_k": 5, "n_anchor": 3, "n_semantic_expansion": 3,
              "fusion_mode": fusion_mode, "require_grounding": require_grounding,
              # Ablated off to keep the mock surface small (semantic_expansion()
              # is Neo4j+FAISS-backed too, same as traverse_graph) -- unrelated
              # to the bug under test (filename collision / resume gating).
              "enable_semantic_expansion": False}
    # Pre-fix, C's "grounded" (require_grounding=True, the historical
    # default) filename had NO suffix at all for that axis -- identical to
    # a pre-grounding-axis file. Only "ungrounded" ever added a suffix.
    old_fusion_suffix = "uniform" if fusion_mode == "uniform" else "fw0-7-0-3"
    old_suffix = old_fusion_suffix if require_grounding else f"{old_fusion_suffix}_ungrounded"
    stale_path = tmp_path / "llm" / "c_graphrag" / "results" / f"condition_c_openai_gpt-4o-mini_n2_seed42_{old_suffix}.jsonl"
    _write_stale_v2_file(stale_path)
    _run_and_collect(params, engine_service.run_condition_c, cond, run_label, stale_path=stale_path)


@pytest.mark.parametrize("require_grounding,run_label", [(False, "D-plain"), (True, "D-grounded")])
def test_condition_d_pilot_configs_produce_records_history_and_manifest(
    common_patches, monkeypatch, require_grounding, run_label,
):
    tmp_path = common_patches
    cond = engine_service._condition_d_module()
    _patch_sampling(monkeypatch, cond)
    _redirect_io(monkeypatch, tmp_path, cond, "d_logs")
    monkeypatch.setattr(cond, "get_all_answer_ids_for_questions", lambda *a, **k: {})
    monkeypatch.setattr(cond, "connect_neo4j", lambda *a, **k: (FakeDriver(), "neo4j"))
    monkeypatch.setattr(cond, "anchor_via_vector_search", lambda *a, **k: [(101, 0.9)])
    monkeypatch.setattr(cond, "retrieve_low_level", lambda *a, **k: [{
        "answer_id": 9101, "via_question_id": 101, "via_question_title": "How do I do X?",
        "answer_body": "Reference answer for X.", "relevance_score": 0.9, "hop": 1, "rel_type": "ANSWERED",
    }])
    monkeypatch.setattr(cond, "retrieve_high_level", lambda *a, **k: [])

    params = {"condition": "D", "mode": "batch", "provider": "openai", "model": "gpt-4o-mini",
              "n_sample": 2, "seed": 42, "top_k": 5, "n_low_level": 3, "n_high_level": 3,
              "require_grounding": require_grounding}
    # Pre-fix, D already suffixed by grounding -- its collision was purely
    # the missing prompt_version/config_hash gate (an older run at the
    # SAME grounded/ungrounded suffix, different prompt_version).
    old_suffix = "grounded" if require_grounding else "ungrounded"
    stale_path = tmp_path / "llm" / "d_lightrag" / "results" / f"condition_d_openai_gpt-4o-mini_n2_seed42_{old_suffix}.jsonl"
    _write_stale_v2_file(stale_path)
    _run_and_collect(params, engine_service.run_condition_d, cond, run_label, stale_path=stale_path)


def test_all_nine_factorial_run_labels_are_covered_by_the_tests_above():
    """Documents the full 9-run coverage this file provides as one place --
    if a 10th config is ever added to the factorial design, this list
    should grow too."""
    covered = {
        "A", "B-plain", "B-grounded",
        "C-uniform-plain", "C-uniform-grounded", "C-trust-plain", "C-trust-grounded",
        "D-plain", "D-grounded",
    }
    assert len(covered) == 9
