"""Offline tests (no LLM calls) for llm_judge_hallucination_v1.py's pure
logic: tag formatting, the resumability key (run_id, question_id,
judge_id, prompt_version), and the failed-call / failures-file routing.
Filename -> condition/run_label/fusion_mode inference now lives in
_run_metadata.py (see test__run_metadata.py) -- this runner only consumes
it via resolve_run_metadata().
"""

import json

from llm_judge_hallucination_v1 import format_tags, load_already_done, run_batch


def test_format_tags_converts_sord_angle_bracket_format():
    assert format_tags("<c#><uwp>") == "c#, uwp"
    assert format_tags("<rust><rust-clippy>") == "rust, rust-clippy"
    assert format_tags("") == ""
    assert format_tags(None) == ""


def test_load_already_done_keys_on_run_id_question_id_judge_id_prompt_version(tmp_path):
    path = tmp_path / "jv1_output.jsonl"
    records = [
        {"run_id": "r1", "question_id": 1, "judge_id": "primary", "prompt_version": "judge-v1"},
        {"run_id": "r1", "question_id": 1, "judge_id": "secondary", "prompt_version": "judge-v1"},
        {"run_id": "r1", "question_id": 2, "judge_id": "primary", "prompt_version": "judge-v1"},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    done = load_already_done(path)
    assert ("r1", 1, "primary", "judge-v1") in done
    assert ("r1", 1, "secondary", "judge-v1") in done
    assert ("r1", 2, "primary", "judge-v1") in done
    assert ("r1", 2, "secondary", "judge-v1") not in done  # never written, must NOT be "done"


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


def test_failed_call_never_written_to_output_and_is_retried_next_run(monkeypatch, tmp_path):
    """A (run_id, question_id, judge_id, prompt_version) that fails after
    all retries must NOT appear in --out (it would then be wrongly
    treated as "done" and skipped forever) -- it goes to a separate
    failures file instead, and a second run_batch() call over the same
    item must still attempt it (not skip)."""
    import llm_judge_hallucination_v1 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry", lambda client, config, messages: (None, "boom"))
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
            {"raw": '{"claims": [], "reasoning": "ok", "label": "FAKTUAL"}',
             "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.01},
            None,
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
            {"raw": '{"claims": [], "reasoning": "ok", "label": "FAKTUAL"}',
             "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.01},
            None,
        ),
    )
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    record = json.loads(output_path.read_text().strip())
    assert record["blinding_version"] == "blind-v2"
