"""A and B write one [latency] line per answered question to their file log, carrying the same numbers
the record stores (llm/latency_report.py parses it). C and D: see the live tests."""

import pandas as pd
import pytest

from llm import latency_report as lr


def test_a_and_b_write_the_latency_line_to_their_log(tmp_path, monkeypatch, tiny_corpus, minilm):
    """Real process_sample() of A and B with fake LLMs: one [latency] line per answer, with the same
    numbers the record stores (retrieval_latency_sec, end_to_end_latency_sec)."""
    from llm.a_pure_llm import a_baseline_replication as a
    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import load_kg_answer_index

    lines = []
    monkeypatch.setattr(a, "log", lines.append)
    monkeypatch.setattr(b, "log", lines.append)
    monkeypatch.setattr(a.time, "sleep", lambda s: None)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    sample = pd.DataFrame([{"Id": 503, "Title": "How to git rebase conflict (case 503)?", "Body": "<p>?</p>",
                            "Tags": "<git>", "AcceptedAnswerId": 1013, "AcceptedAnswerBody": "x", "n_tokens": 5,
                            "ViewCount": 1, "Score": 1}])
    root, _, _ = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    model = minilm
    rec_a = a.process_sample(sample, None, lambda c, m, mo: {"content": "ok"}, model, "fake", tmp_path / "a.jsonl")[0]
    rec_b = b.process_sample(sample, None, lambda c, m, mo: {"content": "ok [SO-520]"}, model, kg, None, 5, "fake",
                             tmp_path / "b.jsonl", require_citation=True)[0]
    got = {}
    for line in lines:
        m = lr.LINE.search(line)
        if m:
            got[m["cond"]] = (m["retrieval"], m["total"])
    assert got["A"][0] == "NA" and float(got["A"][1]) == pytest.approx(rec_a["end_to_end_latency_sec"], abs=2e-3)
    assert float(got["B"][0]) == pytest.approx(rec_b["retrieval_latency_sec"], abs=2e-3)
    assert float(got["B"][1]) == pytest.approx(rec_b["end_to_end_latency_sec"], abs=2e-3)
