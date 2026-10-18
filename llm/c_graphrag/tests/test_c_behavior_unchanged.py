"""Condition C's behaviour is unchanged by the llm/common/ refactor
(reports/b_corpus_parity.md): on 12 dev questions x 4 configs (v2 / v3
alpha=0.5, precomputed query vector / embed_query() fallback), the texts C
passes to embed_query() and the retrieved context (ids, scores, sha256 of
chunk_text) equal the snapshot taken with the code BEFORE the refactor.

Reads Neo4j + the parquet/KG workspace (about 2 minutes) -- opt-in with
GRAPHRAG_INTEGRATION=1."""

import json
import os

import pytest

import c_retrieval_snapshot as snapshot


@pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1", reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
def test_c_retrieval_and_query_text_identical_to_pre_refactor_snapshot(question_faiss):
    expected = json.loads(snapshot.FIXTURE.read_text())
    actual = json.loads(json.dumps(snapshot.take_snapshot(faiss_cache=question_faiss), sort_keys=True))
    assert len(expected["dev_question_ids"]) >= 10
    assert actual["dev_question_ids"] == expected["dev_question_ids"]
    for label in expected["configs"]:
        assert actual["configs"][label] == expected["configs"][label], label



@pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1", reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
def test_default_leakage_mode_is_equivalent_for_questions_without_duplicates(question_faiss):
    """Amendment 8 default mode (Duplicate + near-dup 0.95, Linked off): none of
    the 12 snapshot questions has a Duplicate or near-duplicate, so C's
    retrieval must equal the pre-refactor snapshot exactly."""
    from llm.common.leakage import LeakageContext, default_mode

    ctx = LeakageContext.from_mode(default_mode())
    expected = json.loads(snapshot.FIXTURE.read_text())
    assert all(ctx.table.counts(q)["n_dup"] == 0 and ctx.table.counts(q)["n_neardup"] == 0
               for q in expected["dev_question_ids"])
    actual = json.loads(json.dumps(snapshot.take_snapshot(leakage_mode=default_mode(), faiss_cache=question_faiss), sort_keys=True))
    for label in expected["configs"]:
        assert actual["configs"][label] == expected["configs"][label], label
