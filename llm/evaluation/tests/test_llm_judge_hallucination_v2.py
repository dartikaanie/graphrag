"""Offline tests (no LLM calls) for llm_judge_hallucination_v2.py --
mirrors test_llm_judge_hallucination_v1.py's coverage, adapted for
judge-v2's output record shape (label==derived_label is the OFFICIAL
label, judge_reported_label is the judge's own self-report)."""

import json

import pytest

from llm_judge_hallucination_v2 import format_tags, load_already_done, load_items, run_batch
from llm.manifest import MixedConfigHashError


def test_load_items_refuses_a_deliberately_mixed_config_hash_file(tmp_path, monkeypatch):
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
    assert format_tags("") == ""
    assert format_tags(None) == ""


def _done_key(run_id, qid, judge_id, model="m", prompt_version="judge-v2", blinding_version="blind-v2", config_hash="h1"):
    # parse_version/max_tokens (2026-10-07 follow-up) trail as None --
    # the records this helper matches against never set either field.
    return (run_id, config_hash, qid, judge_id, model, prompt_version, blinding_version, None, None)


def test_load_already_done_keys_match_v1s_resume_tuple_shape(tmp_path):
    """Same 7-tuple SHAPE as judge-v1 (shared _judge_common.judge_resume_
    key), but prompt_version="judge-v2" means a v1 record for the same
    (run_id, question_id, judge_id) never counts as already-done here,
    and vice versa -- the mixed-version separation the user required."""
    path = tmp_path / "jv2_output.jsonl"
    records = [
        {"run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "primary", "judge_model": "m",
         "prompt_version": "judge-v2", "blinding_version": "blind-v2"},
    ]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    done = load_already_done(path)
    assert _done_key("r1", 1, "primary") in done
    assert _done_key("r1", 1, "primary", prompt_version="judge-v1") not in done


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


def _fake_success_payload(answer_attempted=True, completeness="FULL", claims=None, label="FAKTUAL"):
    raw = json.dumps({
        "answer_attempted": answer_attempted, "completeness": completeness,
        "claims": claims or [], "reasoning": "ok", "label": label,
    })
    return {
        "raw": raw,
        "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.01,
        "response_id": "resp_123", "response_model": "fake-model", "response_created": 1234567890,
        "system_fingerprint": "fp_fake", "finish_reason": "stop",
        "usage_full": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "request_params": {"model": "fake-model", "temperature": 0.0},
    }


def test_failed_call_never_written_to_output_and_is_retried_next_run(monkeypatch, tmp_path):
    import llm_judge_hallucination_v2 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry", lambda client, config, messages: (None, _FAKE_ATTEMPTS_FAILED))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 0
    assert summary1["failed"] == 1
    assert not output_path.exists()

    summary2 = runner.run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 0
    assert summary2["failed"] == 1


def test_successful_call_official_label_overrides_judge_self_report(monkeypatch, tmp_path):
    """The whole point of judge-v2: label (OFFICIAL) == derived_label,
    recomputed from claims, even when the judge's own self-reported
    label disagrees."""
    import llm_judge_hallucination_v2 as runner

    payload = _fake_success_payload(
        claims=[{"claim": "x", "verdict": "FABRICATED", "severity": "CORE",
                 "reference_conflict": False, "evidence": "e"}],
        label="FAKTUAL",  # judge self-reports FAKTUAL despite a CORE fabrication
    )
    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (payload, [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    summary = runner.run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary["judged"] == 1

    record = json.loads(output_path.read_text().strip())
    assert record["label"] == "HALUSINASI_PENUH"
    assert record["derived_label"] == "HALUSINASI_PENUH"
    assert record["label"] == record["derived_label"]
    assert record["judge_reported_label"] == "FAKTUAL"
    assert record["consistent"] is False


def test_successful_call_skipped_on_resume(monkeypatch, tmp_path):
    import llm_judge_hallucination_v2 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 1

    summary2 = run_batch(items, ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 1
    assert summary2["judged"] == 0


def test_output_record_carries_prompt_version_and_blinding_version(monkeypatch, tmp_path):
    import llm_judge_hallucination_v2 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)

    record = json.loads(output_path.read_text().strip())
    assert record["prompt_version"] == "judge-v2"
    assert record["blinding_version"] == "blind-v2"


def test_attempted_claims_conflict_counted_in_summary(monkeypatch, tmp_path):
    import llm_judge_hallucination_v2 as runner

    payload = _fake_success_payload(
        answer_attempted=False, completeness="NONE",
        claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None, "reference_conflict": False, "evidence": "e"}],
        label="ABSTAIN",
    )
    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (payload, [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    summary = runner.run_batch([_fake_item()], ["secondary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary["attempted_claims_conflicts"] == 1

    record = json.loads(output_path.read_text().strip())
    assert record["label"] == "ABSTAIN"
    assert record["attempted_claims_conflict"] is True


def test_judge_one_flags_served_model_mismatch(monkeypatch):
    import llm_judge_hallucination_v2 as runner

    class _ConfigWithExpectation:
        model = "requested-model"
        base_url = None
        expected_served_model = "requested-model-Turbo"

    payload = _fake_success_payload()
    payload["response_model"] = "some-other-model"
    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (payload, [{"attempt": 1, "error": None, "status_code": None}]))

    record = runner.judge_one(_fake_item(), "primary", None, _ConfigWithExpectation(), "body", "ref")
    assert record["served_model_mismatch"] is True


def test_resume_key_differs_from_v1_only_by_prompt_version():
    from llm_judge_hallucination_v2 import _resume_key

    record = {
        "run_id": "r1", "config_hash": "h1", "question_id": 1, "judge_id": "primary",
        "judge_model": "m1", "prompt_version": "judge-v2", "blinding_version": "blind-v2",
    }
    # parse_version/max_tokens (2026-10-07 follow-up) trail as None here
    # since this record doesn't set them -- see the dedicated
    # parse_version/max_tokens resume-key tests below for the case where
    # they do differ.
    assert _resume_key(record) == ("r1", "h1", 1, "primary", "m1", "judge-v2", "blind-v2", None, None)
