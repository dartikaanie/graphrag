"""Acceptance check for D's item provenance on the REAL KG (read-only):

1. Retrieval is unchanged: the new retrieve_low_level()/retrieve_high_level()
   return, for the anchors of the 12 dev questions in the committed C snapshot,
   exactly the rows (same order) the pre-provenance Cypher returns -- the old
   query text is kept below, verbatim from commit 6af872f.
2. Every fused item validates against the schema and every edge of every path
   (incl. alt_paths) exists in Neo4j; trust_weight is None.

Needs Neo4j only (anchors come from the C snapshot fixture; no FAISS). Opt-in:
GRAPHRAG_INTEGRATION=1."""

import json
import os
from pathlib import Path

import pytest

import d_lightrag as dl
from llm.common import item_provenance as ip

pytestmark = pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1",
                                reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = REPO_ROOT / "llm" / "c_graphrag" / "tests" / "fixtures" / "c_retrieval_snapshot_dev_expansion_answered.json"

OLD_LOW = """
            UNWIND $anchor_ids AS anchor_id
            MATCH (anchor:Question {id: anchor_id})-[r:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN anchor_id AS via_question_id, anchor.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, 1 AS hop,
                   type(r) AS rel_type, a.isAccepted AS is_accepted
            ORDER BY a.id ASC
            """
OLD_HIGH = """
            UNWIND $anchor_ids AS anchor_id
            MATCH (anchor:Question {id: anchor_id})-[r2:IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM]->(q2:Question)
            WHERE NOT q2.id IN $exclude_q
              AND EXISTS {
                MATCH (q2)-[:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(ax:Answer)
                WHERE NOT ax.id IN $exclude_a
              }
            WITH DISTINCT q2, r2.weight AS relevance_score, type(r2) AS r2_type
            ORDER BY relevance_score DESC, q2.id ASC
            LIMIT $n_high_level
            MATCH (q2)-[r3:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN q2.id AS via_question_id, q2.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, 2 AS hop,
                   r2_type AS rel_type, relevance_score, a.isAccepted AS is_accepted
            ORDER BY q2.id ASC, a.id ASC
            """
LOW_COLS = ("via_question_id", "answer_id", "hop", "rel_type", "is_accepted", "answer_body")
HIGH_COLS = LOW_COLS + ("relevance_score",)


@pytest.fixture(scope="module")
def env():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    snap = json.loads(SNAPSHOT.read_text())
    cases = [(q["question_id"], q["anchor_question_ids"]) for q in snap["configs"]["v2_precomputed"]]
    driver, database = dl.connect_neo4j(lambda *a, **k: None)
    yield {"driver": driver, "database": database, "cases": cases}
    driver.close()


def _old(env, query, **params):
    with env["driver"].session(database=env["database"]) as session:
        return [dict(r) for r in session.run(query, **params)]


def _project(rows, cols):
    return [tuple(r[c] for c in cols) for r in rows]


def test_retrieval_rows_identical_to_pre_provenance_queries_and_provenance_is_valid(env, capsys):
    lookup = ip.neo4j_edge_lookup(env["driver"], env["database"])
    stats = {"questions": 0, "low": 0, "high": 0, "multi_anchor_rows": 0, "alt_paths": 0}
    for qid, anchor_ids in env["cases"]:
        anchors = [(a, 0.9 - 0.1 * i) for i, a in enumerate(anchor_ids)]
        exclude_q, exclude_a = {qid}, set()
        low = dl.retrieve_low_level(env["driver"], env["database"], anchors, exclude_a)
        high = dl.retrieve_high_level(env["driver"], env["database"], anchors, exclude_q, exclude_a, 3)
        old_low = _old(env, OLD_LOW, anchor_ids=anchor_ids, exclude_a=[-1])
        old_high = _old(env, OLD_HIGH, anchor_ids=anchor_ids, exclude_q=[qid], exclude_a=[-1], n_high_level=3)
        assert _project(low, LOW_COLS) == _project(old_low, LOW_COLS), qid
        assert _project(high, HIGH_COLS) == _project(old_high, HIGH_COLS), qid
        stats["multi_anchor_rows"] += sum(1 for c in high if c["alt_anchor_ids"])
        items = dl.fuse_dual_level(low, high, 5, 400)
        stats["questions"] += 1
        for item in items:
            assert ip.validate_item(item) == [], (qid, item["answer_id"])
            assert ip.validate_item_in_kg(item, lookup) == [], (qid, item["answer_id"])
            stats["low" if item["source_stage"] == "low_level" else "high"] += 1
            stats["alt_paths"] += len(item["alt_paths"])
            assert item["trust_weight"] is None
    with capsys.disabled():
        print(f"\n[live provenance D] {stats}")
    assert stats["low"] + stats["high"] > 0
