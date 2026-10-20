"""Record schema r2 (llm/record_schema.py) in Condition B, on the tiny
offline KG corpus with a fake LLM client (no API call)."""

import pandas as pd

from llm.b_rag import b_condition_b_rag as b
from llm.b_rag.kg_corpus import load_kg_answer_index

RUN_ID = "11111111-2222-3333-4444-555555555555"


def test_b_record_has_r2_fields(tiny_corpus, minilm, monkeypatch, tmp_path):
    root, _, _ = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    monkeypatch.setattr(b, "log", lambda *a, **k: None)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    sample = pd.DataFrame([{"Id": 503, "Title": "How to git rebase conflict (case 503)?", "Body": "<p>?</p>",
                            "Tags": "<git>", "AcceptedAnswerId": 1013, "AcceptedAnswerBody": "x", "n_tokens": 5,
                            "ViewCount": 1, "Score": 1}])
    rec = b.process_sample(sample, None, lambda c, m, mo: {"content": "ok [SO-520]"}, minilm, kg, None, 5, "fake",
                           tmp_path / "o.jsonl", run_id=RUN_ID)[0]
    assert (rec["run_id"], rec["condition"], rec["record_schema_version"]) == (RUN_ID, "B", "r3")
    for key in ("retrieval_latency_sec", "generation_latency_sec", "total_latency_sec"):
        assert isinstance(rec[key], float) and rec[key] >= 0, key
    assert rec["total_latency_sec"] >= rec["retrieval_latency_sec"] + rec["generation_latency_sec"] - 1e-3
    # schema r3: end-to-end timing around retrieval + generation (NF3)
    assert rec["answer_started_at_utc"] <= rec["answer_finished_at_utc"]
    assert rec["retrieval_latency_sec"] + rec["generation_latency_sec"] - 2e-3 <= rec["end_to_end_latency_sec"] \
        <= rec["total_latency_sec"] + 1e-3
    assert {"answer_id", "question_id", "is_accepted", "score"} <= set(rec["retrieved_context"][0])
