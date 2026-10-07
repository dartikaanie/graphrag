"""Test offline (stub call_fn, tanpa panggilan LLM sungguhan) untuk
llm_judge_context_relevance.py."""

import json

import _judge_common
import llm_judge_context_relevance as ctxrel


class StubCallFn:
    """Callable FIFO -- mengembalikan response yang sudah discript, dipakai
    menggantikan call_llm_fn tanpa panggilan network apa pun."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, client, messages, model, temperature=None):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.responses.pop(0)


def _make_retrieved(n):
    return [
        {
            "question_id": 1000 + i, "answer_id": 2000 + i, "chunk_text": f"chunk {i}",
            "trust_weight": 0.9, "combined_score": 0.8, "hop": 1, "rel_type": "X",
            "source_stage": "graph_traversal", "is_accepted": True,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------
# Prompt tidak pernah bocor metadata retrieval
# ---------------------------------------------------------------------

def test_item_prompt_never_leaks_metadata():
    retrieved = _make_retrieved(3)
    messages = ctxrel.build_item_relevance_prompt("judul", "body", retrieved)
    text = json.dumps(messages)
    for forbidden in ("trust_weight", "combined_score", "source_stage", "is_accepted",
                       "rel_type", "graph_traversal", "0.9", "0.8"):
        assert forbidden not in text
    for i in range(3):
        assert f"chunk {i}" in text


def test_sufficiency_prompt_never_leaks_metadata():
    retrieved = _make_retrieved(2)
    messages = ctxrel.build_sufficiency_prompt("judul", "body", retrieved, "ground truth text")
    text = json.dumps(messages)
    for forbidden in ("trust_weight", "combined_score", "source_stage", "is_accepted", "rel_type"):
        assert forbidden not in text
    assert "ground truth text" in text


# ---------------------------------------------------------------------
# Validasi jumlah item & label
# ---------------------------------------------------------------------

def test_item_count_and_label_validation_ok():
    raw = json.dumps({"items": [
        {"idx": 1, "label": "RELEVAN", "reason": "ok"},
        {"idx": 2, "label": "TIDAK_RELEVAN", "reason": "no"},
    ]})
    parsed = ctxrel._parse_items_output(raw, 2)
    assert parsed is not None
    assert [p["label"] for p in parsed] == ["RELEVAN", "TIDAK_RELEVAN"]


def test_item_count_mismatch_rejected():
    raw = json.dumps({"items": [{"idx": 1, "label": "RELEVAN"}]})
    assert ctxrel._parse_items_output(raw, 2) is None


def test_invalid_label_rejected():
    raw = json.dumps({"items": [{"idx": 1, "label": "MAYBE"}, {"idx": 2, "label": "RELEVAN"}]})
    assert ctxrel._parse_items_output(raw, 2) is None


def test_duplicate_idx_rejected():
    raw = json.dumps({"items": [{"idx": 1, "label": "RELEVAN"}, {"idx": 1, "label": "TIDAK_RELEVAN"}]})
    assert ctxrel._parse_items_output(raw, 2) is None


def test_sufficiency_invalid_label_rejected():
    raw = json.dumps({"sufficiency": "MUNGKIN", "justification": "x"})
    assert ctxrel._parse_sufficiency_output(raw) is None


# ---------------------------------------------------------------------
# Retry-then-fail path
# ---------------------------------------------------------------------

def test_retry_then_fail_path():
    retrieved = _make_retrieved(2)
    record = {"question_id": 1, "title": "t", "retrieved_context": retrieved, "ground_truth_answer": "gt"}
    call_fn = StubCallFn(["not json", "still not json"])
    result = ctxrel.judge_context_relevance_record(
        None, call_fn, "model-x", 0.1, record, question_bodies={}, max_body_chars=1000, skip_sufficiency=True,
    )
    assert result["status"] == "failed"
    assert result["item_labels"] == []
    assert len(call_fn.calls) == 2  # persis 1 retry, tidak lebih


# ---------------------------------------------------------------------
# Zero-context: tidak ada panggilan LLM sama sekali
# ---------------------------------------------------------------------

def test_zero_context_no_llm_call():
    record = {"question_id": 1, "title": "t", "retrieved_context": [], "ground_truth_answer": "gt"}
    call_fn = StubCallFn([])
    result = ctxrel.judge_context_relevance_record(
        None, call_fn, "model-x", 0.1, record, question_bodies={}, max_body_chars=1000, skip_sufficiency=False,
    )
    assert result["status"] == "no_context"
    assert result["n_items"] == 0
    assert result["context_precision_strict"] is None
    assert len(call_fn.calls) == 0


# ---------------------------------------------------------------------
# Metrik turunan: precision strict/lenient, first_relevant_rank, RR
# ---------------------------------------------------------------------

def test_derived_metrics():
    retrieved = _make_retrieved(4)
    record = {"question_id": 1, "title": "t", "retrieved_context": retrieved, "ground_truth_answer": "gt answer"}
    items_raw = json.dumps({"items": [
        {"idx": 1, "label": "TIDAK_RELEVAN"},
        {"idx": 2, "label": "RELEVAN"},
        {"idx": 3, "label": "SEBAGIAN"},
        {"idx": 4, "label": "RELEVAN"},
    ]})
    suff_raw = json.dumps({"sufficiency": "CUKUP", "justification": "ok"})
    call_fn = StubCallFn([items_raw, suff_raw])
    result = ctxrel.judge_context_relevance_record(
        None, call_fn, "model-x", 0.1, record, question_bodies={}, max_body_chars=1000, skip_sufficiency=False,
    )
    assert result["status"] == "ok"
    assert result["context_precision_strict"] == 0.5        # 2 RELEVAN / 4
    assert result["context_precision_lenient"] == 0.625      # (2 + 0.5*1) / 4
    assert result["first_relevant_rank"] == 2
    assert result["reciprocal_rank"] == 0.5
    assert result["sufficiency"] == "CUKUP"
    assert result["item_labels"][1]["answer_id"] == retrieved[1]["answer_id"]
    assert result["item_labels"][1]["question_id"] == retrieved[1]["question_id"]


def test_derived_metrics_no_relevant_item():
    retrieved = _make_retrieved(2)
    record = {"question_id": 1, "title": "t", "retrieved_context": retrieved, "ground_truth_answer": "gt"}
    items_raw = json.dumps({"items": [
        {"idx": 1, "label": "TIDAK_RELEVAN"}, {"idx": 2, "label": "TIDAK_RELEVAN"},
    ]})
    call_fn = StubCallFn([items_raw])
    result = ctxrel.judge_context_relevance_record(
        None, call_fn, "model-x", 0.1, record, question_bodies={}, max_body_chars=1000, skip_sufficiency=True,
    )
    assert result["context_precision_strict"] == 0.0
    assert result["first_relevant_rank"] is None
    assert result["reciprocal_rank"] == 0.0


# ---------------------------------------------------------------------
# Resume & already_complete
# ---------------------------------------------------------------------

def test_resume_and_already_complete(tmp_path, monkeypatch):
    # Explicit, not just monkeypatch.chdir(tmp_path) + RESULTS_DIR's
    # cwd-relative default: build_output_path() (in _judge_common.py)
    # reads _judge_common.RESULTS_DIR directly, and backend/
    # engine_service.py's _judge_v1_module()/_context_relevance_v1_
    # module()/_judge_module() all do a RAW (non-monkeypatch) assignment
    # to that SAME global to point it at the real llm/evaluation/results/
    # dir -- which never reverts once any earlier test in the same
    # process has imported engine_service and triggered one of those.
    # Relying on chdir alone made this test's isolation depend on test
    # ORDER across the whole suite; pinning RESULTS_DIR here directly
    # removes that dependency regardless of what ran before it. (This is
    # also what caused the real leak into llm/evaluation/results/
    # ctxrel__condition_c_test__openai-gpt-4o-mini_t0-1.jsonl -- see
    # docs/agent_prompt_judge_results_export.md's follow-up.)
    monkeypatch.setattr(_judge_common, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "condition_c_test.jsonl"
    records = [
        {"question_id": 1, "title": "t1", "retrieved_context": [], "ground_truth_answer": "g"},
        {"question_id": 2, "title": "t2", "retrieved_context": [], "ground_truth_answer": "g"},
    ]
    with open(input_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(ctxrel, "get_llm_client", lambda provider, model, log=print: (None, StubCallFn([])))
    monkeypatch.setattr(ctxrel, "load_question_bodies", lambda parquet, ids: {})

    summary1 = ctxrel.run_ctxrel_batch(
        input_path, "C", "openai", "gpt-4o-mini", 0.1, "dummy.parquet", 1000, skip_sufficiency=True, force=False,
    )
    assert summary1["n_evaluated_baru"] == 2
    assert summary1["n_dari_cache"] == 0

    summary2 = ctxrel.run_ctxrel_batch(
        input_path, "C", "openai", "gpt-4o-mini", 0.1, "dummy.parquet", 1000, skip_sufficiency=True, force=False,
    )
    assert summary2["n_evaluated_baru"] == 0
    assert summary2["n_dari_cache"] == 2
