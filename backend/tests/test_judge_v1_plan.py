"""Tests for engine_service.plan_judge_v1() -- the judge-v1 Launch UI's
pre-launch confirmation table. No LLM calls; a tiny fixture generation
run_history + output file stand in for a real pilot run.
"""

import json

import pytest

from app.services import engine_service, history_service, invalid_runs_service, superseded_runs_service


def _write_generation_run(tmp_path, monkeypatch, condition="A", n=2, config_hash="hashABC"):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    output_path = tmp_path / "out.jsonl"
    with open(output_path, "w") as f:
        for qid in range(1, n + 1):
            f.write(json.dumps({
                "question_id": qid, "llm_answer": f"answer {qid}", "title": "t", "tags": "",
                "ground_truth_answer": "ref", "config_hash": config_hash,
            }) + "\n")
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": condition, "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": n, "n_processed": n, "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0, "config_hash": config_hash,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, condition, history_path)
    run_id = history_service._history_id(condition, record)
    return run_id, output_path


def test_plan_refuses_invalid_run(tmp_path, monkeypatch):
    run_id, _ = _write_generation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: {run_id})

    result = engine_service.plan_judge_v1([run_id], ["primary"], {})
    assert result["rows"][0]["refused_reason"].startswith("run is marked invalid")


def test_plan_refuses_superseded_run(tmp_path, monkeypatch):
    run_id, _ = _write_generation_run(tmp_path, monkeypatch)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: {run_id})

    result = engine_service.plan_judge_v1([run_id], ["primary"], {})
    assert result["rows"][0]["refused_reason"].startswith("run is marked superseded")


def test_plan_refuses_mixed_config_hash_output_file(tmp_path, monkeypatch):
    history_path = tmp_path / "a_pure_llm" / "logs" / "run_history.jsonl"
    output_path = tmp_path / "mixed_out.jsonl"
    with open(output_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "config_hash": "aaa"}) + "\n")
        f.write(json.dumps({"question_id": 2, "config_hash": "bbb"}) + "\n")
    record = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "A", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 2, "n_processed": 2, "seed": 42,
        "output_path": str(output_path), "duration_sec": 1.0,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(record) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, "A", history_path)
    run_id = history_service._history_id("A", record)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())

    result = engine_service.plan_judge_v1([run_id], ["primary"], {})
    assert "mix" in result["rows"][0]["refused_reason"].lower()


def test_plan_is_resume_aware(tmp_path, monkeypatch):
    run_id, output_path = _write_generation_run(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())

    # Pre-populate the canonical judge-v1 output file with 1 of the 2 items
    # already judged by "primary" -- plan() must see only 1 remaining.
    primary_out, _, _ = engine_service.judge_v1_output_paths("primary")
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_failures.jsonl",
                                      tmp_path / f"jv1_{jid}_resolved.jsonl"))
    from app.services import engine_service as es
    judge_model = es._judge_clients_module().JUDGE_REGISTRY["primary"].model
    blinding_version = es._judge_v1_module().BLINDING_VERSION

    primary_out = tmp_path / "jv1_primary.jsonl"
    primary_out.write_text(json.dumps({
        "run_id": run_id, "config_hash": "hashABC", "question_id": 1, "judge_id": "primary",
        "judge_model": judge_model, "prompt_version": "judge-v1", "blinding_version": blinding_version,
        "call_failed": False, "label": "FAKTUAL",
    }) + "\n")

    result = engine_service.plan_judge_v1([run_id], ["primary"], {"primary": 4})
    row = result["rows"][0]
    assert row["items_total"] == 2
    assert row["already_judged"] == 1
    assert row["items_to_judge"] == 1


def test_plan_totals_sum_across_rows(tmp_path, monkeypatch):
    run_id, _ = _write_generation_run(tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_f.jsonl", tmp_path / f"jv1_{jid}_r.jsonl"))

    result = engine_service.plan_judge_v1([run_id], ["primary", "secondary"], {})
    assert result["totals"]["items_to_judge"] == 6  # 3 items x 2 judges, none done yet


# ---------------------------------------------------------------------
# Token/cost/time estimation -- fallback order, "unknown" path, totals.
# ---------------------------------------------------------------------

def _write_judge_output(results_dir, filename, records):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_judge_token_latency_stats_from_history(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    results_dir = tmp_path / "llm" / "evaluation" / "results"
    _write_judge_output(results_dir, "jv1_old_name.jsonl", [
        {"judge_model": "some-model", "prompt_tokens": 1000, "completion_tokens": 500, "latency_s": 20.0, "call_failed": False},
        {"judge_model": "some-model", "prompt_tokens": 1200, "completion_tokens": 700, "latency_s": 30.0, "call_failed": False},
        {"judge_model": "other-model", "prompt_tokens": 99999, "completion_tokens": 99999, "latency_s": 1.0, "call_failed": False},
        {"judge_model": "some-model", "prompt_tokens": 5, "completion_tokens": 5, "latency_s": 5.0, "call_failed": True},  # excluded
    ])

    avg_in, avg_out, avg_lat, source = engine_service._judge_token_latency_stats("some-model")
    assert avg_in == 1100  # (1000+1200)/2
    assert avg_out == 600  # (500+700)/2
    assert avg_lat == 25.0  # (20+30)/2
    assert source == "from history"


def test_judge_token_latency_stats_scans_any_jv1_filename_not_just_canonical(tmp_path, monkeypatch):
    """Explicitly covers "including older filenames" -- a file that
    predates the judge-prompt-version-in-filename change must still be
    found by the fallback scan."""
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    results_dir = tmp_path / "llm" / "evaluation" / "results"
    _write_judge_output(results_dir, "jv1_smoke_primary_OLD_NAME_v2.jsonl", [
        {"judge_model": "legacy-model", "prompt_tokens": 2000, "completion_tokens": 1000, "latency_s": 40.0, "call_failed": False},
    ])
    avg_in, avg_out, avg_lat, source = engine_service._judge_token_latency_stats("legacy-model")
    assert avg_in == 2000
    assert source == "from history"


def test_judge_token_latency_stats_none_when_no_history_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    avg_in, avg_out, avg_lat, source = engine_service._judge_token_latency_stats("never-seen-model")
    assert (avg_in, avg_out, avg_lat, source) == (None, None, None, None)


def test_plan_judge_v1_falls_back_to_registry_defaults_when_no_history(tmp_path, monkeypatch):
    run_id, _ = _write_generation_run(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)  # empty results/ -- no history anywhere
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_f.jsonl", tmp_path / f"jv1_{jid}_r.jsonl"))

    result = engine_service.plan_judge_v1([run_id], ["primary"], {"primary": 4})
    row = result["rows"][0]
    config = engine_service._judge_clients_module().JUDGE_REGISTRY["primary"]
    assert row["token_source"] == "default estimate"
    assert row["est_tokens_in"] == config.default_input_tokens_per_item * 2
    assert row["est_tokens_out"] == config.default_output_tokens_per_item * 2
    assert row["cost_unknown"] is False
    assert row["est_cost_usd"] is not None and row["est_cost_usd"] > 0


def test_plan_judge_v1_never_reports_zero_cost_for_unregistered_judge(tmp_path, monkeypatch):
    run_id, _ = _write_generation_run(tmp_path, monkeypatch, n=2)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_f.jsonl", tmp_path / f"jv1_{jid}_r.jsonl"))

    result = engine_service.plan_judge_v1([run_id], ["not-a-real-judge-id"], {})
    row = result["rows"][0]
    assert row["cost_unknown"] is True
    assert row["est_cost_usd"] is None
    assert row["token_source"] is None

    # Totals must exclude it (never silently sum as $0).
    totals = result["totals"]
    per_judge = totals["per_judge"]["not-a-real-judge-id"]
    assert per_judge["cost_unknown_rows"] == 1
    assert per_judge["est_cost_usd"] is None
    assert totals["est_cost_usd"] is None  # no judge had a known cost at all
    assert totals["cost_unknown_rows"] == 1


def test_plan_judge_v1_total_time_sums_across_runs_divided_by_workers(tmp_path, monkeypatch):
    """Runs are processed sequentially per judge (workers parallelize
    WITHIN a run, not across runs) -- two 10-item runs at workers=2 with
    a 30s/item default latency must take ~ (10+10)/2*30 = 300s, NOT
    max(10/2*30, 10/2*30) = 150s (the original bug)."""
    run_id_1, _ = _write_generation_run(tmp_path, monkeypatch, condition="A", n=10, config_hash="hashA")
    history_path_b = tmp_path / "b_rag" / "logs" / "run_history.jsonl"
    output_path_b = tmp_path / "out_b.jsonl"
    with open(output_path_b, "w") as f:
        for qid in range(1, 11):
            f.write(json.dumps({"question_id": qid, "llm_answer": "a", "title": "t", "tags": "",
                                 "ground_truth_answer": "r", "config_hash": "hashB"}) + "\n")
    record_b = {
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "B", "status": "success",
        "provider": "openai", "model": "gpt-4o-mini", "n_sample_target": 10, "n_processed": 10, "seed": 42,
        "output_path": str(output_path_b), "duration_sec": 1.0, "config_hash": "hashB",
    }
    history_path_b.parent.mkdir(parents=True, exist_ok=True)
    history_path_b.write_text(json.dumps(record_b) + "\n")
    monkeypatch.setitem(history_service.HISTORY_PATHS, "B", history_path_b)
    run_id_2 = history_service._history_id("B", record_b)

    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_f.jsonl", tmp_path / f"jv1_{jid}_r.jsonl"))

    result = engine_service.plan_judge_v1([run_id_1, run_id_2], ["primary"], {"primary": 2})
    config = engine_service._judge_clients_module().JUDGE_REGISTRY["primary"]
    expected_time = round(20 / 2 * config.default_latency_sec_per_item)
    assert result["totals"]["per_judge"]["primary"]["est_time_sec"] == expected_time
    assert result["totals"]["est_time_sec"] == expected_time


def test_plan_judge_v1_overall_time_is_slowest_judge_not_sum(tmp_path, monkeypatch):
    """Judges run in PARALLEL -- overall wall-clock is the max across
    judges, never their sum."""
    run_id, _ = _write_generation_run(tmp_path, monkeypatch, n=10)
    monkeypatch.setattr(invalid_runs_service, "load_invalid_run_ids", lambda: set())
    monkeypatch.setattr(superseded_runs_service, "load_superseded_run_ids", lambda: set())
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(engine_service, "judge_v1_output_paths",
                         lambda jid: (tmp_path / f"jv1_{jid}.jsonl", tmp_path / f"jv1_{jid}_f.jsonl", tmp_path / f"jv1_{jid}_r.jsonl"))

    # primary: 10 items / 1 worker * 30s = 300s. secondary: 10 items / 10 workers * 30s = 30s.
    result = engine_service.plan_judge_v1([run_id], ["primary", "secondary"], {"primary": 1, "secondary": 10})
    primary_time = result["totals"]["per_judge"]["primary"]["est_time_sec"]
    secondary_time = result["totals"]["per_judge"]["secondary"]["est_time_sec"]
    assert primary_time > secondary_time
    assert result["totals"]["est_time_sec"] == primary_time  # the slower one, not primary_time + secondary_time
