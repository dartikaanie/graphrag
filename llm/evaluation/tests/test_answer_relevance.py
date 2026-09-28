"""Test offline (stub call_fn) untuk llm_judge_answer_relevance.py."""

import json

import llm_judge_answer_relevance as ansrel


class StubCallFn:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, client, messages, model, temperature=None):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.responses.pop(0)


def test_prompt_never_leaks_reference_or_context():
    messages = ansrel.build_answer_relevance_prompt("judul", "body", "jawaban LLM di sini")
    text = json.dumps(messages)
    assert "ground_truth" not in text.lower()
    assert "retrieved_context" not in text.lower()
    assert "jawaban LLM di sini" in text
    # eksplisit dinyatakan di prompt bahwa TIDAK ADA reference answer diberikan
    assert "no reference answer" in text.lower()


def test_label_validation_ok():
    raw = json.dumps({"answer_relevance_label": "MENJAWAB", "justification": "ok"})
    parsed = ansrel._parse_answer_relevance_output(raw)
    assert parsed["answer_relevance_label"] == "MENJAWAB"


def test_invalid_label_rejected():
    raw = json.dumps({"answer_relevance_label": "MUNGKIN"})
    assert ansrel._parse_answer_relevance_output(raw) is None


def test_malformed_json_rejected():
    assert ansrel._parse_answer_relevance_output("not json at all") is None


def test_retry_then_fail_path():
    record = {"question_id": 1, "title": "t", "llm_answer": "jawaban"}
    call_fn = StubCallFn(["not json", "still not json"])
    result = ansrel.judge_answer_relevance_record(None, call_fn, "model-x", 0.1, record, {}, 1000)
    assert result["status"] == "failed"
    assert result["answer_relevance_score"] is None
    assert len(call_fn.calls) == 2


def test_label_to_score_mapping():
    record = {"question_id": 1, "title": "t", "llm_answer": "jawaban"}
    for label, expected_score in (("MENJAWAB", 1.0), ("MENJAWAB_SEBAGIAN", 0.5), ("TIDAK_MENJAWAB", 0.0)):
        raw = json.dumps({"answer_relevance_label": label, "justification": "ok"})
        call_fn = StubCallFn([raw])
        result = ansrel.judge_answer_relevance_record(None, call_fn, "model-x", 0.1, record, {}, 1000)
        assert result["status"] == "ok"
        assert result["answer_relevance_score"] == expected_score


def test_resume_and_already_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "condition_a_test.jsonl"
    records = [
        {"question_id": 1, "title": "t1", "llm_answer": "jawaban1"},
        {"question_id": 2, "title": "t2", "llm_answer": "jawaban2"},
    ]
    with open(input_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    responses = [
        json.dumps({"answer_relevance_label": "MENJAWAB", "justification": "ok"}),
        json.dumps({"answer_relevance_label": "TIDAK_MENJAWAB", "justification": "ok"}),
    ]
    call_fn = StubCallFn(responses)
    monkeypatch.setattr(ansrel, "get_llm_client", lambda provider, model, log=print: (None, call_fn))
    monkeypatch.setattr(ansrel, "load_question_bodies", lambda parquet, ids: {})

    summary1 = ansrel.run_ansrel_batch(
        input_path, "A", "openai", "gpt-4o-mini", 0.1, "dummy.parquet", 1000, force=False,
    )
    assert summary1["n_evaluated_baru"] == 2
    assert summary1["n_dari_cache"] == 0

    summary2 = ansrel.run_ansrel_batch(
        input_path, "A", "openai", "gpt-4o-mini", 0.1, "dummy.parquet", 1000, force=False,
    )
    assert summary2["n_evaluated_baru"] == 0
    assert summary2["n_dari_cache"] == 2
