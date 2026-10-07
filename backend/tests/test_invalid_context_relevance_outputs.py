"""Tests for invalid_context_relevance_outputs_service.py and its use in
engine_service._ctxrel_token_latency_stats() -- a marked smoke-test
output file must never contribute to a real plan-table token estimate.
"""

import json

from app.services import engine_service, invalid_context_relevance_outputs_service as svc


def test_mark_invalid_then_load_round_trips(tmp_path, monkeypatch):
    path = tmp_path / "invalid_context_relevance_outputs.jsonl"
    monkeypatch.setattr(svc, "INVALID_CTXREL_OUTPUTS_PATH", path)

    assert svc.load_invalid_output_basenames() == set()
    svc.mark_invalid("ctxrel_v1_smoke_test.jsonl", "smoke test")
    assert svc.load_invalid_output_basenames() == {"ctxrel_v1_smoke_test.jsonl"}


def test_token_latency_stats_skips_marked_smoke_test_file(tmp_path, monkeypatch):
    results_dir = tmp_path / "llm" / "evaluation" / "results"
    results_dir.mkdir(parents=True)

    smoke_path = results_dir / "ctxrel_v1_smoke_test.jsonl"
    with open(smoke_path, "w") as f:
        f.write(json.dumps({"judge_model": "m", "call_failed": False, "prompt_tokens": 99999,
                             "completion_tokens": 99999, "latency_s": 999.0}) + "\n")

    canonical_path = results_dir / "ctxrel_v1_primary_ctxrel-v1.jsonl"
    with open(canonical_path, "w") as f:
        f.write(json.dumps({"judge_model": "m", "call_failed": False, "prompt_tokens": 100,
                             "completion_tokens": 50, "latency_s": 2.0}) + "\n")

    invalid_path = tmp_path / "invalid_context_relevance_outputs.jsonl"
    invalid_path.write_text(json.dumps({"output_file": "ctxrel_v1_smoke_test.jsonl", "reason": "smoke test"}) + "\n")
    monkeypatch.setattr(svc, "INVALID_CTXREL_OUTPUTS_PATH", invalid_path)
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)

    avg_in, avg_out, avg_lat, source = engine_service._ctxrel_token_latency_stats("m")
    assert avg_in == 100
    assert avg_out == 50
    assert avg_lat == 2.0
    assert source == "from history"


def test_token_latency_stats_uses_smoke_test_file_when_not_marked(tmp_path, monkeypatch):
    """Sanity check for the test above: WITHOUT the invalid-outputs
    marker, the smoke-test file's numbers WOULD be picked up -- proving
    the marker (not some other filter) is what excludes it."""
    results_dir = tmp_path / "llm" / "evaluation" / "results"
    results_dir.mkdir(parents=True)

    smoke_path = results_dir / "ctxrel_v1_smoke_test.jsonl"
    with open(smoke_path, "w") as f:
        f.write(json.dumps({"judge_model": "m", "call_failed": False, "prompt_tokens": 500,
                             "completion_tokens": 500, "latency_s": 5.0}) + "\n")

    invalid_path = tmp_path / "invalid_context_relevance_outputs.jsonl"
    monkeypatch.setattr(svc, "INVALID_CTXREL_OUTPUTS_PATH", invalid_path)
    monkeypatch.setattr(engine_service, "REPO_ROOT", tmp_path)

    avg_in, avg_out, avg_lat, source = engine_service._ctxrel_token_latency_stats("m")
    assert avg_in == 500
    assert source == "from history"
