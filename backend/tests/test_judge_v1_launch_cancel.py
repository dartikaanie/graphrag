"""End-to-end (mocked judge client, no real LLM calls) test for the
judge-v1 dashboard job driver: launch -> status -> cancel -> resume.
Calls engine_service.run_judge_v1_batch_dashboard()/start_judge_v1_run()
directly (not through a real background thread) for determinism.
"""

import json

import pytest

from app.services import (
    engine_service,
    history_service,
    invalid_runs_service,
    run_registry,
    superseded_runs_service,
)


def _fake_call_judge_with_retry(client, config, messages):
    payload = {
        "raw": '{"claims": [], "reasoning": "ok", "label": "FAKTUAL"}',
        "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.01,
        "response_id": "resp_fake", "response_model": config.model, "response_created": 1234567890,
        "system_fingerprint": "fp_fake", "finish_reason": "stop",
        "usage_full": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "request_params": {"model": config.model, "temperature": 0.0},
    }
    return payload, [{"attempt": 1, "error": None, "status_code": None}]


@pytest.fixture
def judge_v1_fixture(tmp_path, monkeypatch):
    # Generation run fixture (2 questions).
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    output_path = tmp_path / "condition_a_test_fixture.jsonl"
    with open(output_path, "w") as f:
        for qid in (1, 2):
            f.write(json.dumps({
                "question_id": qid, "llm_answer": f"answer {qid}", "title": "t", "tags": "",
                "ground_truth_answer": "ref", "config_hash": "hashABC",
            }) + "\n")
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "A", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 2, "n_processed": 2, "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": "hashABC",
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)

    engine_service._judge_v1_module()  # self-heals sys.path for the bare import below
    import _run_metadata
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "A", history_path)

    run_id = history_service._history_id("A", record)

    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())

    judge_out_dir = tmp_path / "jv1_results"
    judge_out_dir.mkdir()

    def fake_paths(judge_id):
        base = judge_out_dir / f"jv1_dashboard_{judge_id}.jsonl"
        return base, judge_out_dir / f"jv1_dashboard_{judge_id}_failures.jsonl", judge_out_dir / f"jv1_dashboard_{judge_id}_resolved.jsonl"

    monkeypatch.setattr(engine_service, "judge_v1_output_paths", fake_paths)
    monkeypatch.setattr(engine_service, "judge_v1_run_log_path", lambda jr: tmp_path / "jv1.log")

    mod = engine_service._judge_v1_module()
    monkeypatch.setattr(mod, "get_judge_client", lambda judge_id: (object(), engine_service._judge_clients_module().JUDGE_REGISTRY[judge_id]))
    monkeypatch.setattr(mod, "call_judge_with_retry", _fake_call_judge_with_retry)
    monkeypatch.setattr(mod, "load_question_bodies", lambda parquet, ids: {i: "<p>body</p>" for i in ids})
    monkeypatch.setattr(engine_service, "_questions_parquet", lambda: "fake_parquet")

    # CRITICAL: append_manifest() (imported into llm_judge_hallucination_v1
    # from _judge_common) reads _judge_common's OWN LOG_DIR global, which
    # _judge_v1_module() points at the REAL llm/evaluation/logs/ directory.
    # Without this, every test run here would append a real row to the
    # real judge_v1_run_history.jsonl -- exactly the test-isolation bug
    # this comment is guarding against (caught via a real smoke test: see
    # docs/agent_prompt_phase2_judge_ui.md's final report).
    import _judge_common
    monkeypatch.setattr(_judge_common, "LOG_DIR", tmp_path / "jv1_logs")
    monkeypatch.setattr(_judge_common, "RESULTS_DIR", judge_out_dir)

    return run_id, judge_out_dir


def test_launch_runs_to_completion_and_writes_canonical_output(judge_v1_fixture):
    run_id, judge_out_dir = judge_v1_fixture
    job_run_id = run_registry.create_run("JUDGE_V1", "batch", {})
    params = {"run_ids": [run_id], "judge_ids": ["primary"], "workers": {"primary": 2}}

    engine_service.run_judge_v1_batch_dashboard(job_run_id, params)

    state = run_registry.get_run(job_run_id)
    assert state["status"] == "completed", state.get("error")
    assert state["summary"]["per_judge"]["primary"]["judged"] == 2

    out_path = judge_out_dir / "jv1_dashboard_primary.jsonl"
    assert out_path.exists()
    records = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(r["run_id"] == run_id for r in records)
    assert (out_path.with_name(out_path.name + ".manifest.json")).exists()


def test_resume_skips_already_judged_items(judge_v1_fixture):
    run_id, judge_out_dir = judge_v1_fixture
    params = {"run_ids": [run_id], "judge_ids": ["primary"], "workers": {"primary": 2}}

    job1 = run_registry.create_run("JUDGE_V1", "batch", {})
    engine_service.run_judge_v1_batch_dashboard(job1, params)

    out_path = judge_out_dir / "jv1_dashboard_primary.jsonl"
    assert len(out_path.read_text().splitlines()) == 2

    # Second launch over the SAME run/judge must not duplicate records.
    job2 = run_registry.create_run("JUDGE_V1", "batch", {})
    engine_service.run_judge_v1_batch_dashboard(job2, params)

    assert len(out_path.read_text().splitlines()) == 2
    state2 = run_registry.get_run(job2)
    assert state2["summary"]["per_judge"]["primary"]["judged"] == 0
    assert state2["summary"]["per_judge"]["primary"]["skipped"] == 2


def test_cancel_before_semaphore_marks_cancelled_without_running(judge_v1_fixture, monkeypatch):
    run_id, judge_out_dir = judge_v1_fixture
    job_run_id = run_registry.create_run("JUDGE_V1", "batch", {})
    params = {"run_ids": [run_id], "judge_ids": ["primary"], "workers": {"primary": 2}}

    run_registry.request_cancel(job_run_id)
    engine_service.start_judge_v1_run(job_run_id, params)

    state = run_registry.get_run(job_run_id)
    assert state["status"] == "cancelled"
    out_path = judge_out_dir / "jv1_dashboard_primary.jsonl"
    assert not out_path.exists()
