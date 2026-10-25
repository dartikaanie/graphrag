"""secondary_metrics.py (Amendment 16): plan/dry run makes no call and counts exactly; the judge is the single
registry judge "primary" at temperature 0; the answer is blinded; official files stay behind the Amendment 14 gate."""

import json

import pytest

import judge_versions as jv
import secondary_metrics as sm


def _run_file(tmp_path, name, records, split="dev"):
    p = tmp_path / name
    p.write_text("".join(json.dumps({**r, "config": {"sample_split": split}}) + "\n" for r in records))
    return str(p)


def _item(aid, text):
    return {"answer_id": aid, "chunk_text": text}


@pytest.fixture
def no_client(monkeypatch):
    import judge_clients

    def boom(*a, **k):
        raise AssertionError("a judge client was created in a dry run")

    monkeypatch.setattr(judge_clients, "get_judge_client", boom)


def test_plan_counts_calls_exactly_and_creates_no_client(tmp_path, no_client):
    b = _run_file(tmp_path, "b.jsonl", [
        {"question_id": 1, "llm_answer": "a", "retrieved_context": [_item(10, "x"), _item(11, "y")]},
        {"question_id": 2, "llm_answer": "b", "retrieved_context": [_item(10, "x")]}])
    c = _run_file(tmp_path, "c.jsonl", [
        {"question_id": 1, "llm_answer": "a", "retrieved_context": [_item(10, "x"), _item(12, "z")]}])
    a = _run_file(tmp_path, "a.jsonl", [{"question_id": 1, "llm_answer": "a"}, {"question_id": 2, "llm_answer": "b"}])
    p = sm.plan({"A": ("A", a), "B": ("B", b), "C": ("C", c)})
    # context relevance: unique (question, answer, text) items over B and C: (1,10), (1,11), (2,10), (1,12) = 4
    # answer relevance: one call per answer of every file: 2 + 2 + 1 = 5
    assert p["calls"] == {"ctxrel": 4, "ansrel": 5} and p["total_calls"] == 9
    assert p["judge"]["model"] == "meta-llama/Llama-3.3-70B-Instruct" and p["judge"]["temperature"] == 0.0
    assert "NO CALL WAS MADE" in sm.plan_text(p)


def test_projection_for_the_four_runs(no_client):
    p = sm.project(384, 5)
    assert p["calls"] == {"ctxrel": 3 * 384 * 5, "ansrel": 4 * 384} and p["total_calls"] == 7296
    assert set(p["per_run"]) == {"A", "B-grounded", "C-primary-grounded", "D-grounded"}
    assert p["per_run"]["A"]["ctxrel_items_before_dedup"] == 0  # A has no context
    assert p["estimated_cost_usd"] > 0


def test_one_judge_temperature_zero_existing_prompts():
    import llm_judge_context_relevance_v1 as cx
    from judge_clients import JUDGE_REGISTRY

    assert sm.JUDGE_ID == "primary" and JUDGE_REGISTRY["primary"].model == "meta-llama/Llama-3.3-70B-Instruct"
    assert cx.REQUEST_TEMPERATURE == 0.0
    assert sm.CTXREL_PROMPT in ("ctxrel-v1", "ctxrel-v2") and cx.get_prompt_builder(sm.CTXREL_PROMPT)


def test_official_files_are_refused_before_amendment14_approval(tmp_path, monkeypatch):
    f = _run_file(tmp_path, "c.jsonl", [{"question_id": 1}], split="test")
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", False)
    with pytest.raises(jv.JudgeVersionError, match="Amendment 14"):
        sm._gate([f])
    sm._gate([_run_file(tmp_path, "dev.jsonl", [{"question_id": 1}], split="dev")])  # dev files need no approval
    monkeypatch.setattr(jv, "AMENDMENT14_APPROVED", True)
    sm._gate([f])


def test_answer_relevance_prompt_is_blinded_and_sees_only_question_and_answer(tmp_path, monkeypatch):
    import _judge_common
    import llm_judge_context_relevance_v1 as cx

    seen = []

    def fake_call(client, config, messages):
        seen.append(messages)
        return {"raw": json.dumps({"answer_relevance_label": "MENJAWAB", "justification": "ok"}),
                "response_model": "m", "prompt_tokens": 1, "completion_tokens": 1}, []

    class Cfg:
        model = "meta-llama/Llama-3.3-70B-Instruct"

    import judge_clients
    monkeypatch.setattr(judge_clients, "get_judge_client", lambda jid: (object(), Cfg()))
    monkeypatch.setattr(cx, "call_judge_with_retry", fake_call)
    monkeypatch.setattr(_judge_common, "load_question_bodies", lambda parquet, ids: {i: "<p>question body</p>" for i in ids})
    monkeypatch.setattr(sm, "OUT_DIR", tmp_path)
    monkeypatch.setattr(sm, "ANSREL_FILE", tmp_path / "ansrel.jsonl")
    answer = "Use foo() [SO-123]. The retrieved sources do not cover bar. Sources: [SO-123]"
    f = _run_file(tmp_path, "c.jsonl", [{"question_id": 7, "title": "T", "llm_answer": answer,
                                         "ground_truth_answer": "REFERENCE", "retrieved_context": [_item(1, "CTX")],
                                         "condition": "C", "trust_weight": 0.9}])
    out = sm.run_ansrel([f], workers=1)
    assert out["answer_relevance_judged"] == 1 and len(seen) == 1
    text = json.dumps(seen[0])
    for leak in ("[SO-123]", "retrieved sources", "REFERENCE", "CTX", "trust", "GraphRAG", "Condition"):
        assert leak not in text, leak
    assert "Use foo()" in text and "question body" in text
    rec = json.loads((tmp_path / "ansrel.jsonl").read_text())
    assert rec["label"] == "MENJAWAB" and rec["score"] == 1.0 and rec["temperature"] == 0.0
    # resume: a second call judges nothing new
    assert sm.run_ansrel([f], workers=1)["answer_relevance_judged"] == 0


def test_report_section_reads_stored_judgments(tmp_path, monkeypatch):
    f = _run_file(tmp_path, "a.jsonl", [{"question_id": 1}, {"question_id": 2}])
    canon = str(sm.archive_paths.canonical(f))
    rows = [{"source_file": canon, "question_id": 1, "prompt_version": sm.ANSREL_PROMPT, "label": "MENJAWAB", "score": 1.0},
            {"source_file": canon, "question_id": 2, "prompt_version": sm.ANSREL_PROMPT, "label": "TIDAK_MENJAWAB", "score": 0.0}]
    (tmp_path / "ansrel.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sm, "ANSREL_FILE", tmp_path / "ansrel.jsonl")
    monkeypatch.setattr(sm, "CTXREL_FILE", tmp_path / "missing.jsonl")
    out = sm.report_section(f, "A")
    assert out["answer_relevance"]["mean_score"] == 0.5 and out["answer_relevance"]["n_judged"] == 2
    assert out["context_relevance"] is None  # condition A has no context
