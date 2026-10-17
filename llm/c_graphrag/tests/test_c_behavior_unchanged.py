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
def test_c_retrieval_and_query_text_identical_to_pre_refactor_snapshot():
    expected = json.loads(snapshot.FIXTURE.read_text())
    actual = json.loads(json.dumps(snapshot.take_snapshot(), sort_keys=True))
    assert len(expected["dev_question_ids"]) >= 10
    assert actual["dev_question_ids"] == expected["dev_question_ids"]
    for label in expected["configs"]:
        assert actual["configs"][label] == expected["configs"][label], label

