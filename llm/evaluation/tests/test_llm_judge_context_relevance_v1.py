"""Offline tests (no LLM calls) for llm_judge_context_relevance_v1.py:
cross-run dedup on (question_id, answer_id, text_hash), the resume key,
failed-call routing, condition-A exclusion, and compute_run_context_
relevance()'s metrics math (including rank-position split)."""

import json

import pytest

from llm_judge_context_relevance_v1 import (
    compute_run_context_relevance,
    format_tags,
    load_already_done,
    load_ctxrel_records,
    load_items,
    run_batch,
)
from llm.manifest import MixedConfigHashError


def _write_run_file(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _context_item(answer_id, chunk_text, qid_field=None):
    item = {"answer_id": answer_id, "chunk_text": chunk_text,
            "trust_weight": 0.9, "combined_score": 0.5, "hop": 1,
            "source_stage": "anchor", "is_accepted": True, "rel_type": "ANSWERS"}
    if qid_field is not None:
        item["question_id"] = qid_field
    return item


def _history(tmp_path, condition, output_path, **extra):
    history_path = tmp_path / f"run_history_{condition}.jsonl"
    record = {"run_started_at": "2026-10-05T00:00:00+00:00", "condition": condition,
              "status": "success", "output_path": str(output_path)}
    record.update(extra)
    history_path.write_text(json.dumps(record) + "\n")
    return history_path


def test_format_tags_converts_sord_angle_bracket_format():
    assert format_tags("<c#><uwp>") == "c#, uwp"
    assert format_tags("") == ""
    assert format_tags(None) == ""


def test_load_items_dedups_identical_context_across_plain_and_grounded_files(tmp_path, monkeypatch):
    """The whole point of Step 3's dedup requirement: a plain/grounded
    pair of the SAME retrieval method shares byte-identical
    retrieved_context (verified on the real 9-run pilot), so judging
    must only cost one call per unique (question_id, answer_id,
    text_hash), not two."""
    import _run_metadata

    shared_ctx = [_context_item("A1", "some helpful text"), _context_item("A2", "other text")]
    plain_path = tmp_path / "condition_c_plain.jsonl"
    grounded_path = tmp_path / "condition_c_grounded.jsonl"
    _write_run_file(plain_path, [{"question_id": 1, "title": "t", "tags": "<a>", "retrieved_context": shared_ctx}])
    _write_run_file(grounded_path, [{"question_id": 1, "title": "t", "tags": "<a>", "retrieved_context": shared_ctx}])

    history_path = tmp_path / "run_history_C.jsonl"
    with open(history_path, "w") as f:
        f.write(json.dumps({"run_started_at": "2026-10-05T00:00:00+00:00", "condition": "C",
                             "status": "success", "output_path": str(plain_path)}) + "\n")
        f.write(json.dumps({"run_started_at": "2026-10-05T00:00:01+00:00", "condition": "C",
                             "status": "success", "output_path": str(grounded_path)}) + "\n")
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "C", history_path)

    items = load_items([str(plain_path), str(grounded_path)], limit=None)
    assert len(items) == 2  # 2 unique context items, not 4


def test_load_items_keeps_distinct_text_as_separate_items(tmp_path, monkeypatch):
    import _run_metadata

    path = tmp_path / "condition_c_x.jsonl"
    _write_run_file(path, [
        {"question_id": 1, "title": "t", "tags": "", "retrieved_context": [
            _context_item("A1", "text one"), _context_item("A1", "text one but different"),
        ]},
    ])
    history_path = _history(tmp_path, "C", path)
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "C", history_path)

    items = load_items([str(path)], limit=None)
    # same answer_id, different chunk_text -- NOT deduplicated together
    assert len(items) == 2


def test_load_items_refuses_a_deliberately_mixed_config_hash_file(tmp_path, monkeypatch):
    import _run_metadata

    path = tmp_path / "condition_c_mixed.jsonl"
    _write_run_file(path, [
        {"question_id": 1, "title": "t1", "retrieved_context": [_context_item("A1", "x")], "config_hash": "aaaa111111"},
        {"question_id": 2, "title": "t2", "retrieved_context": [_context_item("A2", "y")], "config_hash": "bbbb222222"},
    ])
    history_path = _history(tmp_path, "C", path, fusion_mode="uniform", require_grounding=True)
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "C", history_path)

    with pytest.raises(MixedConfigHashError):
        load_items([str(path)], limit=None)


def _fake_item(question_id=1, answer_id="A1", text="some context text"):
    import hashlib
    return {
        "question_id": question_id, "answer_id": answer_id,
        "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "chunk_text": text, "title": "t", "tags": "", "source_files": ["/fake.jsonl"],
    }


class _FakeConfig:
    model = "fake-model"
    base_url = None
    expected_served_model = None


def _fake_success_payload(label="RELEVANT"):
    return {
        "raw": json.dumps({"label": label, "reason": "ok"}),
        "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.01,
        "response_id": "resp_1", "response_model": "fake-model", "finish_reason": "stop",
    }


def test_resume_key_is_content_based_not_run_based():
    import llm_judge_context_relevance_v1 as runner

    r = {"question_id": 1, "answer_id": "A1", "text_hash": "h1", "judge_id": "primary",
         "judge_model": "m", "prompt_version": "ctxrel-v1"}
    assert runner._resume_key(r) == (1, "A1", "h1", "primary", "m", "ctxrel-v1")


def test_successful_call_written_to_output_and_skipped_on_resume(monkeypatch, tmp_path):
    import llm_judge_context_relevance_v1 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = run_batch(items, ["primary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 1
    assert summary1["failed"] == 0
    assert json.loads(output_path.read_text().strip())["label"] == "RELEVANT"

    summary2 = run_batch(items, ["primary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 1
    assert summary2["judged"] == 0


def test_failed_call_never_written_to_output_and_is_retried_next_run(monkeypatch, tmp_path):
    import llm_judge_context_relevance_v1 as runner

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (None, [{"attempt": 1, "error": "boom", "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    items = [_fake_item()]
    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"

    summary1 = run_batch(items, ["primary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary1["judged"] == 0
    assert summary1["failed"] == 1
    assert not output_path.exists()

    summary2 = run_batch(items, ["primary"], "fake_parquet", output_path, failures_path, workers=1)
    assert summary2["skipped"] == 0  # not treated as done -- retried
    assert summary2["failed"] == 1


def test_resolved_retry_is_logged_separately_without_touching_failures_file(tmp_path, monkeypatch):
    import llm_judge_context_relevance_v1 as runner

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    resolved_path = tmp_path / "out_failures_resolved.jsonl"

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (None, [{"attempt": 1, "error": "boom", "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _FakeConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})
    runner.run_batch([_fake_item()], ["primary"], "fake_parquet", output_path, failures_path, workers=1)
    original = failures_path.read_text()
    assert original

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]))
    runner.run_batch([_fake_item()], ["primary"], "fake_parquet", output_path, failures_path, workers=1)

    assert failures_path.read_text() == original  # untouched
    assert resolved_path.exists()
    resolution = json.loads(resolved_path.read_text().strip())
    assert resolution["resolved"] is True
    assert resolution["answer_id"] == "A1"


def test_served_model_mismatch_flagged(monkeypatch, tmp_path):
    import llm_judge_context_relevance_v1 as runner

    class _MismatchConfig(_FakeConfig):
        expected_served_model = "expected-model-x"

    monkeypatch.setattr(runner, "call_judge_with_retry",
                         lambda client, config, messages: (_fake_success_payload(), [{"attempt": 1, "error": None, "status_code": None}]))
    monkeypatch.setattr(runner, "get_judge_client", lambda judge_id: (None, _MismatchConfig()))
    monkeypatch.setattr(runner, "load_question_bodies", lambda parquet, ids: {})

    output_path = tmp_path / "out.jsonl"
    failures_path = tmp_path / "out_failures.jsonl"
    runner.run_batch([_fake_item()], ["primary"], "fake_parquet", output_path, failures_path, workers=1)

    record = json.loads(output_path.read_text().strip())
    assert record["served_model_mismatch"] is True


def test_condition_a_has_no_context_items_to_judge(tmp_path, monkeypatch):
    """Condition A never has retrieved_context -- load_items() over an A
    file must yield zero items (nothing to judge), matching main()'s
    "no context items found" error path rather than crashing."""
    import _run_metadata

    path = tmp_path / "condition_a_x.jsonl"
    _write_run_file(path, [{"question_id": 1, "title": "t", "llm_answer": "x"}])  # no retrieved_context key
    history_path = _history(tmp_path, "A", path)
    monkeypatch.setitem(_run_metadata.HISTORY_PATHS, "A", history_path)

    items = load_items([str(path)], limit=None)
    assert items == []


def test_compute_run_context_relevance_metrics_and_rank_split(tmp_path):
    import hashlib

    run_path = tmp_path / "condition_c_run.jsonl"
    ctx = [
        _context_item("A1", "relevant text"),
        _context_item("A2", "partial text"),
        _context_item("A3", "irrelevant text"),
    ]
    _write_run_file(run_path, [{"question_id": 1, "retrieved_context": ctx}])

    def h(t):
        return hashlib.sha256(t.encode()).hexdigest()

    ctxrel_by_key = {
        (1, "A1", h("relevant text")): {"label": "RELEVANT"},
        (1, "A2", h("partial text")): {"label": "PARTIAL"},
        (1, "A3", h("irrelevant text")): {"label": "IRRELEVANT"},
    }

    metrics = compute_run_context_relevance(str(run_path), ctxrel_by_key)
    assert metrics["n_items_total"] == 3
    assert metrics["n_items_judged"] == 3
    assert metrics["n_items_unjudged"] == 0
    assert metrics["pct_relevant"] == round(100 / 3, 2)
    assert metrics["n_questions"] == 1
    assert metrics["n_questions_with_relevant"] == 1
    assert metrics["pct_questions_with_relevant"] == 100.0
    assert metrics["mean_relevance_score"] == round((1.0 + 0.5 + 0.0) / 3, 4)
    # rank 1 = "relevant text" (score 1.0); ranks 2-3 = partial+irrelevant (mean 0.25)
    assert metrics["mean_relevance_score_rank1"] == 1.0
    assert metrics["mean_relevance_score_rank2plus"] == 0.25


def test_compute_run_context_relevance_excludes_unjudged_items(tmp_path):
    run_path = tmp_path / "condition_c_run.jsonl"
    ctx = [_context_item("A1", "judged text"), _context_item("A2", "never judged")]
    _write_run_file(run_path, [{"question_id": 1, "retrieved_context": ctx}])

    import hashlib
    ctxrel_by_key = {(1, "A1", hashlib.sha256(b"judged text").hexdigest()): {"label": "RELEVANT"}}

    metrics = compute_run_context_relevance(str(run_path), ctxrel_by_key)
    assert metrics["n_items_total"] == 2
    assert metrics["n_items_judged"] == 1
    assert metrics["n_items_unjudged"] == 1
    assert metrics["pct_relevant"] == 100.0  # computed only over the 1 judged item


def test_load_ctxrel_records_keys_by_question_answer_text_hash(tmp_path):
    path = tmp_path / "ctxrel_out.jsonl"
    _write_run_file(path, [{"question_id": 1, "answer_id": "A1", "text_hash": "h1", "label": "RELEVANT"}])

    records = load_ctxrel_records([path])
    assert records[(1, "A1", "h1")]["label"] == "RELEVANT"


def test_load_already_done_is_content_keyed(tmp_path):
    path = tmp_path / "out.jsonl"
    _write_run_file(path, [{"question_id": 1, "answer_id": "A1", "text_hash": "h1",
                             "judge_id": "primary", "judge_model": "m", "prompt_version": "ctxrel-v1"}])
    done = load_already_done(path)
    assert (1, "A1", "h1", "primary", "m", "ctxrel-v1") in done
