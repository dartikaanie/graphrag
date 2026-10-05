"""POST /api/runs/check-completed -- the Factorial Batch page's
skip-detection, moved server-side and keyed on config_hash (the SAME rule
the resume gate itself uses, llm.manifest.read_already_done), per the
user's explicit follow-up after the 2026-10-05 pilot incident fix. No LLM
calls, no real Neo4j/DuckDB -- history_service reads run_history.jsonl
fixture files written to tmp_path.
"""

import json

from app.services import engine_service, history_service


def _write_history(path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_compute_run_config_hash_matches_build_config_for_condition_a():
    cond = engine_service._condition_a_module()
    params = {"provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42,
              "oversample_pool": 1536, "mode": "batch", "condition": "A"}
    expected = engine_service._compute_config_hash(
        cond.build_config("openai", "gpt-4o-mini", 10, 42, 1536)
    )
    assert engine_service.compute_run_config_hash("A", params) == expected


def test_is_run_already_completed_true_when_config_hash_and_success_match(tmp_path, monkeypatch):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    params = {"provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42,
              "oversample_pool": 1536, "mode": "batch", "condition": "A"}
    config_hash = engine_service.compute_run_config_hash("A", params)
    _write_history(history_path, [{
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "A", "status": "success",
        "config_hash": config_hash, "output_path": str(tmp_path / "out.jsonl"), "duration_sec": 1.0,
    }])
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    completed, returned_hash = engine_service.is_run_already_completed("A", params)
    assert completed is True
    assert returned_hash == config_hash


def test_is_run_already_completed_false_when_hash_differs(tmp_path, monkeypatch):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    params = {"provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42,
              "oversample_pool": 1536, "mode": "batch", "condition": "A"}
    _write_history(history_path, [{
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "A", "status": "success",
        "config_hash": "totally-different-hash", "output_path": str(tmp_path / "out.jsonl"),
        "duration_sec": 1.0,
    }])
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    completed, _ = engine_service.is_run_already_completed("A", params)
    assert completed is False


def test_is_run_already_completed_false_for_older_run_with_no_config_hash_at_all(tmp_path, monkeypatch):
    """An older run_history entry that predates config_hash entirely must
    NEVER count as "already completed" -- it isn't verifiably the same
    config, matching read_already_done()'s resume-gate semantics."""
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    params = {"provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42,
              "oversample_pool": 1536, "mode": "batch", "condition": "A"}
    _write_history(history_path, [{
        "run_started_at": "2026-09-12T00:00:00+00:00", "condition": "A", "status": "success",
        "output_path": str(tmp_path / "out.jsonl"), "duration_sec": 1.0,
        # no "config_hash" field at all -- pre-dates the fix
    }])
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    completed, _ = engine_service.is_run_already_completed("A", params)
    assert completed is False


def test_check_completed_endpoint_reports_all_nine_as_launch_when_history_is_pre_hash(tmp_path, monkeypatch):
    """The exact scenario motivating this follow-up: the dry-run must show
    all 9 rows as "will launch" when the only matching-config history
    entries predate config_hash (the 3 real pilot successes, C-trust-plain/
    C-uniform-plain/D-plain, which this fix makes stale by filename too)."""
    from fastapi.testclient import TestClient

    from app.main import app

    for condition, folder in [("A", "a_pure_llm"), ("B", "b_rag"), ("C", "c_graphrag"), ("D", "d_lightrag")]:
        history_path = tmp_path / folder / "logs" / "run_history.jsonl"
        _write_history(history_path, [{
            "run_started_at": "2026-09-29T00:00:00+00:00", "condition": condition, "status": "success",
            "output_path": str(tmp_path / f"{condition}_old.jsonl"), "duration_sec": 1.0,
            # no config_hash -- pre-dates the fix, like the 3 real survivors
        }])
        monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)

    base = {"provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42, "mode": "batch"}
    runs = [
        {**base, "condition": "A"},
        {**base, "condition": "B", "require_citation": True, "require_grounding": False},
        {**base, "condition": "B", "require_citation": True, "require_grounding": True},
        {**base, "condition": "C", "fusion_mode": "uniform", "require_grounding": False},
        {**base, "condition": "C", "fusion_mode": "uniform", "require_grounding": True},
        {**base, "condition": "C", "fusion_mode": "trust_weighted", "require_grounding": False},
        {**base, "condition": "C", "fusion_mode": "trust_weighted", "require_grounding": True},
        {**base, "condition": "D", "require_grounding": False},
        {**base, "condition": "D", "require_grounding": True},
    ]

    client = TestClient(app)
    resp = client.post("/api/runs/check-completed", json={"runs": runs})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 9
    assert all(r["already_completed"] is False for r in results), results


def test_judge_v1_output_paths_include_judge_prompt_version():
    base, failures, resolved = engine_service.judge_v1_output_paths("primary")
    prompt_version = engine_service._judge_v1_module().PROMPT_VERSION
    assert f"_{prompt_version}.jsonl" in base.name
    assert f"_{prompt_version}_failures.jsonl" in failures.name
    assert f"_{prompt_version}_failures_resolved.jsonl" in resolved.name


def test_config_hash_excludes_batch_id_and_batch_launched_at():
    """batch_id/batch_launched_at are Factorial Batch bookkeeping only
    (which "Confirm & Launch" click produced a run, for the Hallucination
    Judge page's run picker) -- they must NEVER affect config_hash, which
    covers only parameters that affect the answers themselves. Two runs
    with identical generation params but different batch_id/
    batch_launched_at must hash identically, so the skip-detection/resume
    logic correctly treats them as the same config."""
    base_params = {
        "provider": "openai", "model": "gpt-4o-mini", "n_sample": 10, "seed": 42,
        "oversample_pool": 1536, "mode": "batch", "condition": "A",
    }
    params_1 = {**base_params, "batch_id": "batch-aaaaaaaa", "batch_launched_at": "2026-10-05T00:00:00+00:00"}
    params_2 = {**base_params, "batch_id": "batch-bbbbbbbb", "batch_launched_at": "2026-11-01T12:34:56+00:00"}

    hash_1 = engine_service.compute_run_config_hash("A", params_1)
    hash_2 = engine_service.compute_run_config_hash("A", params_2)
    hash_no_batch = engine_service.compute_run_config_hash("A", base_params)

    assert hash_1 == hash_2 == hash_no_batch
