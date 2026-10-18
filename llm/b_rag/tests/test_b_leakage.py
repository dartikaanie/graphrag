"""Condition B with the expanded exclusion (Amendment 8) on the tiny offline
corpus: related questions and all their answers never come back, top_k is
kept by over-fetching, and a short result is warned about and recorded."""

import pandas as pd

from llm.b_rag import b_condition_b_rag as b
from llm.b_rag.kg_corpus import encode_query, load_kg_answer_index
from llm.common.leakage import ExclusionTable, LeakageContext, assert_no_leakage


def _ctx(rows_by_q: dict, mode="dup+neardup0.95"):
    rows = {q: (frozenset(eq), frozenset(ea), 1, 0, 0) for q, (eq, ea) in rows_by_q.items()}
    return LeakageContext(mode, ExclusionTable(mode, rows))


def test_duplicate_question_and_its_answers_never_retrieved(tiny_corpus, minilm):
    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    # 503 and 513 share a topic (same title stem): treat 513 as 503's Duplicate
    dup_answers = [r["answer_id"] for r in rows if r["question_id"] == 513]
    ctx = _ctx({503: ({503, 513}, set(dup_answers))})
    ex = ctx.exclusion_for(503, kg.thread_answer_ids([503])[503], None)
    title = next(r["question_title"] for r in rows if r["question_id"] == 503)
    stats = {}
    out = b.retrieve_context_kg(minilm, kg, title, 5, ex, encode_query(minilm, title), stats=stats)
    assert len(out) == 5 and stats["n_hits_filtered"] > 0
    assert not {i["question_id"] for i in out} & {503, 513}
    assert_no_leakage(out, ex)


def test_short_result_is_warned_and_recorded(tiny_corpus, minilm, monkeypatch, tmp_path):
    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    keep_q = 520
    all_q = {r["question_id"] for r in rows}
    excluded_q = all_q - {keep_q}  # leaves only 520's 4 answers
    ctx = _ctx({503: (excluded_q, {r["answer_id"] for r in rows if r["question_id"] != keep_q})})
    logs = []
    monkeypatch.setattr(b, "log", lambda msg, *a, **k: logs.append(str(msg)))
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    sample = pd.DataFrame([{"Id": 503, "Title": "How to git rebase conflict (case 503)?", "Body": "<p>?</p>",
                            "Tags": "<git>", "AcceptedAnswerId": 1013, "AcceptedAnswerBody": "x", "n_tokens": 5,
                            "ViewCount": 1, "Score": 1}])
    recs = b.process_sample(sample, None, lambda c, m, mo: {"content": "ok [SO-520]"}, minilm, kg, None, 5, "fake",
                            tmp_path / "o.jsonl", leakage=ctx)
    rec = recs[0]
    assert rec["n_context_items_used"] == 4
    assert rec["leakage"]["final_k"] == 4 and rec["leakage"]["final_k_lt_k"] is True
    assert rec["leakage"]["mode"] == "dup+neardup0.95" and rec["leakage"]["n_excluded_q"] == len(excluded_q)
    assert any("[WARN]" in m and "< top_k=5" in m for m in logs)
    assert any("[leakage] Id=503" in m for m in logs)
