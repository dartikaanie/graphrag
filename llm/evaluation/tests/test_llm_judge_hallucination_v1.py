"""Offline tests (no LLM calls) for llm_judge_hallucination_v1.py's pure
logic: tag formatting, the resumability key (run_id, question_id,
judge_id, prompt_version), and the failed-call / failures-file routing.
Filename -> condition/run_label/fusion_mode inference now lives in
_run_metadata.py (see test__run_metadata.py) -- this runner only consumes
it via resolve_run_metadata().
"""

import json

import pytest

from llm_judge_hallucination_v1 import format_tags, load_already_done, load_items, run_batch
from llm.manifest import MixedConfigHashError


def test_load_items_refuses_a_deliberately_mixed_config_hash_file(tmp_path, monkeypatch):
    """The exact scenario the 2026-10-05 pilot incident fix must prevent
    at the JUDGE stage: a run file that silently mixes answers from two
    DIFFERENT configs (e.g. an old prompt_version's leftover rows plus a
    new run's rows at the same path) must never be judged as if it were
    one coherent run -- see llm.manifest.assert_single_config_hash."""
    import _run_metadata

    output_path = tmp_path / "condition_c_mixed.jsonl"
    with open(output_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "title": "t1", "tags": "<a>",
                             "ground_truth_answer": "g1", "llm_answer": "c1",
                             "config_hash": "aaaa111111"}) + "\n")
        f.write(json.dumps({"question_id": 2, "title": "t2", "tags": "<a>",
                             "ground_truth_answer": "g2", "llm_answer": "c2",
                             "config_hash": "bbbb222222"}) + "\n")

    history_path = tmp_path / "run_history.jsonl"
    history_path.write_text(json.dumps({
        "run_started_at": "2026-10-05T00:00:00+00:00", "condition": "C",
        "status": "success", "output_path": str(output_path), "fusion_mode": "uniform",
        "require_grounding": True,
    }) + "\n")
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "C", history_path)

    with pytest.raises(MixedConfigHashError):
        load_items([str(output_path)], limit=None)


def test_format_tags_converts_sord_angle_bracket_format():
    assert format_tags("<c#><uwp>") == "c#, uwp"
    assert format_tags("<rust><rust-clippy>") == "rust, rust-clippy"
    assert format_tags("") == ""
    assert format_tags(None) == ""


def _done_key(run_id, qid, judge_id, model="m", prompt_version="judge-v1", blinding_version="blind-v2", config_hash="h1"):
    # parse_version/max_tokens (2026-10-07 follow-up) trail as None --
    # the records this helper matches against never set either field.
    return (run_id, config_hash, qid, judge_id, model, prompt_version, blinding_version, None, None)


def test_load_already_done_keys_on_run_id_config_hash_question_id_judge_id_model_prompt_version_blinding_version(tmp_path):
    path = tmp_path / "jv1_output.jsonl"
    records = [
        {"run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "primary", "judge_model": "m",
         "prompt_version": "judge-v1", "blinding_version": "blind-v2"},
        {"run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "secondary", "judge_model": "m",
         "prompt_version": "judge-v1", "blinding_version": "blind-v2"},
        {"run_id": "r1", "config_hash": "h1", "question_id": 2, "judge_id": "primary", "judge_model": "m",
         "prompt_version": "judge-v1", "blinding_version": "blind-v2"},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    done = load_already_done(path)
    assert _done_key("r1", 1, "primary") in done
    assert _done_key("r1", 1, "secondary") in done
    assert _done_key("r1", 2, "primary") in done
    assert _done_key("r1", 2, "secondary") not in done  # never written, must NOT be "done"
    # A DIFFERENT config_hash for the SAME (run_id, question_id, judge_id)
    # must NOT count as already done -- the whole point of adding it to
    # the key (a generation re-run under a new config_hash forces re-judge).
    assert _done_key("r1", 1, "primary", config_hash="h2") not in done


def test_load_already_done_returns_empty_set_for_missing_file(tmp_path):
    assert load_already_done(tmp_path / "does_not_exist.jsonl") == set()


def _fake_item(question_id=999999):
    return {
        "run_id": "C-958c05cc40", "condition": "C", "run_label": "C-uniform",
        "fusion_mode": "uniform", "retrieval_version": "v2", "source_file": "/fake/path.jsonl",
        "question_id": question_id, "title": "t", "tags": "", "candidate": "c",
        "ground_truth_answer_html": "ref",
    }


class _FakeConfig:
    model = "fake-model"
    base_url = None


_FAKE_ATTEMPTS_FAILED = [{"attempt": 1, "error": "boom", "status_code": None}]


def _fake_success_payload():
    return {
        "raw": '{"claims": [], "reasoning": "ok", "label": "FAKTUAL"}',
        "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.01,
        "response_id": "resp_123", "response_model": "fake-model", "response_created": 1234567890,
        "system_fingerprint": "fp_fake", "finish_reason": "stop",
        "usage_full": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "request_params": {"model": "fake-model", "temperature": 0.0},
    }


def test_failed_call_never_written_to_output_and_is_retried_next_run(monkeypatch, tmp_path):
    """A (run_id, question_id, judge_id, prompt_version) that fails after
    all retries must NOT appear in --out (it would then be wrongly
    treated as "done" and skipped forever) -- it goes to a separate
    failures file instead, and a second run_batch() call over the same
    item must still attempt it (not skip)."""
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (None, _FAKE_ATTEMPTS_FAILED),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 0
    assert summary1["failed"] == 1
    assert not output_path.exists()
    assert failures_path.exists()

    summary2 = runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 0  # NOT treated as already-done -- retried again
    assert summary2["failed"] == 1


def test_successful_call_written_to_output_and_skipped_on_resume(monkeypatch, tmp_path):
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (
            _fake_success_payload(),
            [{"attempt": 1, "error": None, "status_code": None}],
        ),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 1
    assert summary1["failed"] == 0
    assert output_path.exists()
    assert not failures_path.exists()

    summary2 = run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 1  # now treated as already-done
    assert summary2["judged"] == 0


def test_output_record_carries_blinding_version(monkeypatch, tmp_path):
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (
            _fake_success_payload(),
            [{"attempt": 1, "error": None, "status_code": None}],
        ),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    record = json.loads(output_path.read_text().strip())
    assert record["blinding_version"] == "blind-v2"


def test_output_record_carries_full_raw_response_metadata(monkeypatch, tmp_path):
    """Step 5 data retention: every successful judge call must keep the
    full raw response metadata (id/model/created/system_fingerprint/
    finish_reason/usage), the request params actually sent, a SHA-256 of
    the exact messages, and the attempt history -- not just the parsed
    label/claims."""
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (
            _fake_success_payload(),
            [{"attempt": 1, "error": None, "status_code": None}],
        ),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    record = json.loads(output_path.read_text().strip())
    assert record["response_id"] == "resp_123"
    assert record["response_model"] == "fake-model"
    assert record["response_created"] == 1234567890
    assert record["system_fingerprint"] == "fp_fake"
    assert record["finish_reason"] == "stop"
    assert record["usage_full"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    assert record["request_params"] == {"model": "fake-model", "temperature": 0.0}
    assert record["attempt_count"] == 1
    assert record["attempts"] == [{"attempt": 1, "error": None, "status_code": None}]
    assert len(record["messages_sha256"]) == 64  # hex sha256


def test_failed_call_keeps_full_attempt_history_with_status_codes(monkeypatch, tmp_path):
    import llm_judge_hallucination_v1 as runner

    attempts = [
        {"attempt": 1, "error": "RateLimitError: 429", "status_code": 429},
        {"attempt": 2, "error": "APITimeoutError: timed out", "status_code": None},
    ]
    monkeypatch.setattr(runner, "call_judge_with_retry", lambda client, config, messages: (None, attempts))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    runner.run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    failure_record = json.loads(failures_path.read_text().strip())
    assert failure_record["attempt_count"] == 2
    assert failure_record["attempts"] == attempts
    assert failure_record["error"] == "APITimeoutError: timed out"  # last attempt's error


def test_resolved_retry_is_logged_separately_without_touching_failures_file(monkeypatch, tmp_path):
    """A (run_id, question_id, judge_id, prompt_version) that failed on a
    PRIOR run and succeeds on a LATER run must be logged to a separate
    <out>_failures_resolved.jsonl -- the original failure record in
    <out>_failures.jsonl must be left untouched, never deleted/rewritten."""
    import llm_judge_hallucination_v1 as runner

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    resolved_path = tmp_path / "out_failures_resolved.jsonl"

    # First run: fails.
    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (None, _FAKE_ATTEMPTS_FAILED),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})
    runner.run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    original_failure_content = failures_path.read_text()
    assert original_failure_content  # the first failure was recorded

    # Second run: same item now succeeds.
    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (
            _fake_success_payload(),
            [{"attempt": 1, "error": None, "status_code": None}],
        ),
    )
    runner.run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    # Original failure record is untouched -- never deleted/rewritten.
    assert failures_path.read_text() == original_failure_content
    assert output_path.exists()
    assert resolved_path.exists()
    resolution = json.loads(resolved_path.read_text().strip())
    assert resolution["resolved"] is True
    assert resolution["run_id"] == "C-958c05cc40"
    assert resolution["judge_id"] == "secondary"


def test_run_batch_calls_on_progress_for_each_task(monkeypatch, tmp_path):
    """The dashboard's live progress/throughput/ETA depends on this --
    on_progress must fire once per task (success or failure), with a
    running done_count/total_count, and the CLI (which never passes
    on_progress) must be unaffected (tested elsewhere by the many calls
    already omitting it)."""
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(
        runner, "call_judge_with_retry",
        lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item(1), _fake_item(2)]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    events = []
    runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1,
                      on_progress=lambda e: events.append(e))

    assert len(events) == 2
    assert {e["status"] for e in events} == {"done"}
    assert {e["total_count"] for e in events} == {2}
    assert sorted(e["done_count"] for e in events) == [1, 2]


def test_run_batch_check_cancel_before_start_submits_nothing(monkeypatch, tmp_path):
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    def _boom(*a, **k):
        raise AssertionError("judge should never be called when cancelled before start")

    monkeypatch.setattr(runner, "call_judge_with_retry", _boom)

    items = [_fake_item(1)]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary = runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1,
                                check_cancel=lambda: True)
    assert summary["judged"] == 0
    assert summary["failed"] == 0
    assert not output_path.exists()


def test_judge_one_flags_served_model_mismatch_and_logs_warning(monkeypatch):
    import llm_judge_hallucination_v1 as runner

    class _ConfigWithExpectation:
        model = "requested-model"
        base_url = None
        expected_served_model = "requested-model-Turbo"

    payload = _fake_success_payload()
    payload["response_model"] = "requested-model-Turbo"  # matches expected -- no mismatch
    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (payload, [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    record = runner.judge_one(_fake_item(), "primary", None, _ConfigWithExpectation(), "body", "ref")
    assert record["served_model_mismatch"] is False


def test_judge_one_detects_and_logs_served_model_mismatch(monkeypatch, capsys):
    import llm_judge_hallucination_v1 as runner

    class _ConfigWithExpectation:
        model = "requested-model"
        base_url = None
        expected_served_model = "requested-model-Turbo"

    payload = _fake_success_payload()
    payload["response_model"] = "some-other-model"  # provider silently serving something else
    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (payload, [{"attempt": 1, "error": None, "status_code": None}]))

    record = runner.judge_one(_fake_item(), "primary", None, _ConfigWithExpectation(), "body", "ref")
    assert record["served_model_mismatch"] is True
    captured = capsys.readouterr()
    assert "WARN" in captured.out
    assert "some-other-model" in captured.out


def test_resume_key_includes_config_hash_judge_model_prompt_and_blinding_version():
    from llm_judge_hallucination_v1 import _resume_key

    record = {
        "run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "primary",
        "judge_model": "m1", "prompt_version": "judge-v1", "blinding_version": "blind-v2",
    }
    # parse_version/max_tokens (2026-10-07 follow-up) trail as None -- v1
    # records never carry either field.
    assert _resume_key(record) == ("r1", "h1", 1, "primary", "m1", "judge-v1", "blind-v2", None, None)
