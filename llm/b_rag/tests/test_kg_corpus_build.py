"""Offline tests of the KG-answer corpus build and Condition B's runtime
retrieval over it (reports/b_corpus_parity.md). No Neo4j, no external data."""

import numpy as np
import pytest

from llm.b_rag import build_kg_answer_index as build
from llm.b_rag.kg_corpus import load_kg_answer_index, read_json
from llm.common.leakage import build_exclusion
from llm.common.text import answer_doc_text, get_tokenizer, truncate_doc_text

from kg_tiny_corpus import write_tiny_corpus


def test_doc_text_is_c_candidate_text():
    enc = get_tokenizer()
    rec = {"answer_id": 1, "question_id": 2, "question_title": "T", "answer_body": "word " * 1000,
           "is_accepted": True, "rel_type": "HAS_ACCEPTED_ANSWER"}
    (row,) = build.make_doc_rows([rec], enc, 400)
    assert row["doc_text"] == truncate_doc_text(enc, "Q: T\nA: " + "word " * 1000, 400)
    assert row["truncated"] and len(enc.encode(row["doc_text"])) <= 400
    (short,) = build.make_doc_rows([{**rec, "answer_body": "hi"}], enc, 400)
    assert short["doc_text"] == answer_doc_text("T", "hi") == "Q: T\nA: hi" and not short["truncated"]


def test_counts_agree_vectors_meta_parquet(tiny_corpus):
    root, paths, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    assert kg.ntotal == len(kg.answer_ids) == len(rows) == read_json(paths.export_manifest)["n_rows"]
    assert list(kg.answer_ids) == sorted(r["answer_id"] for r in rows)
    assert kg.cache_key.startswith("kg_answers_v1_tok400_all-MiniLM-L6-v2_")


def test_encode_resume_is_bit_identical(tmp_path, minilm):
    full_paths, _ = write_tiny_corpus(tmp_path / "full")
    build.stage_encode(full_paths, "all-MiniLM-L6-v2", batch_size=16, device="cpu", unit_rows=32, model=minilm)

    paths, _ = write_tiny_corpus(tmp_path / "resumed")
    ckpt = build.stage_encode(paths, "all-MiniLM-L6-v2", batch_size=16, device="cpu", unit_rows=32, model=minilm,
                              stop_after_units=3)
    assert 0 < ckpt["next_row"] < ckpt["n_total"]  # really stopped part-way
    ckpt = build.stage_encode(paths, "all-MiniLM-L6-v2", batch_size=16, device="cpu", unit_rows=32, model=minilm)
    assert ckpt["next_row"] == ckpt["n_total"]

    assert full_paths.embeddings.read_bytes() == paths.embeddings.read_bytes()
    assert full_paths.embed_ids.read_bytes() == paths.embed_ids.read_bytes()


def test_resume_with_different_batch_size_is_refused(tmp_path, minilm):
    paths, _ = write_tiny_corpus(tmp_path)
    build.stage_encode(paths, "all-MiniLM-L6-v2", batch_size=16, device="cpu", unit_rows=32, model=minilm,
                       stop_after_units=1)
    with pytest.raises(SystemExit, match="different parameters"):
        build.stage_encode(paths, "all-MiniLM-L6-v2", batch_size=8, device="cpu", unit_rows=32, model=minilm)


def test_vectors_are_l2_normalized(tiny_corpus):
    _, paths, rows = tiny_corpus
    emb = np.memmap(paths.embeddings, dtype="float32", mode="r", shape=(len(rows), 384))
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_search_excludes_own_thread_and_keeps_top_k(tiny_corpus, minilm):
    from llm.b_rag.kg_corpus import encode_query

    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    qid = 503
    thread = kg.thread_answer_ids([qid])[qid]
    assert thread == sorted(r["answer_id"] for r in rows if r["question_id"] == qid)
    # query == the thread's own title: without exclusion its answers rank first
    title = next(r["question_title"] for r in rows if r["question_id"] == qid)
    q_emb = encode_query(minilm, title)
    unfiltered = [int(kg.answer_ids[r]) for r, _ in kg.search(q_emb, 5)]
    assert set(unfiltered) & set(thread)
    excl = build_exclusion(qid, thread)
    hits = kg.search(q_emb, 5, excl)
    assert len(hits) == 5
    assert not any(excl.excludes(kg.question_ids[r], kg.answer_ids[r]) for r, _ in hits)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_widens_when_overfetch_is_not_enough(tiny_corpus, minilm):
    """Exclusion that is not just the thread (here: every answer of 10
    questions on the same topic) still yields top_k results."""
    from llm.b_rag.kg_corpus import encode_query
    from llm.common.leakage import LeakageExclusion

    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    same_topic_q = {r["question_id"] for r in rows if (r["question_id"] - 500) % 10 == 0}
    same_topic_a = {r["answer_id"] for r in rows if r["question_id"] in same_topic_q}
    # only ONE answer id in the exclusion -> initial over-fetch is top_k + 6,
    # but 24 same-topic answers rank first and are excluded via question_id
    excl = LeakageExclusion(frozenset(same_topic_q), frozenset({min(same_topic_a)}))
    hits = kg.search(encode_query(minilm, rows[0]["question_title"]), 5, excl)
    assert len(hits) == 5 and not ({int(kg.answer_ids[r]) for r, _ in hits} & same_topic_a)


def test_retrieved_text_comes_from_parquet_doc_text(tiny_corpus, minilm):
    from llm.b_rag import b_condition_b_rag as b

    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    by_id = {r["answer_id"]: r for r in rows}
    out = b.retrieve_context_kg(minilm, kg, "how to sort a python list", 5, build_exclusion(1, []))
    assert len(out) == 5
    for item in out:
        src = by_id[item["answer_id"]]
        assert item["chunk_text"] == src["doc_text"]
        assert item["question_id"] == src["question_id"] and item["is_accepted"] == src["is_accepted"]
