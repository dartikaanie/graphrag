"""Condition D item provenance (llm/common/item_provenance.py), offline.

* Regression: fuse_dual_level()'s items, minus the keys provenance added
  (llm/common/item_provenance.ADDED_ITEM_KEYS, rank, trust_weight), equal what
  the code BEFORE provenance returned on the same candidates.
* Provenance: low-level = anchor -> answer, high-level = anchor -> q2 -> answer,
  trust_weight is the path's edge weight (audit only), alt_paths keep the paths
  (incl. other anchors) that lost the dedup.
* retrieve_high_level() keeps its Cypher's row set/order/LIMIT while collecting
  the anchors that reached each row.

The live-Neo4j check is in test_item_provenance_d_live.py (GRAPHRAG_INTEGRATION=1)."""

import json
from pathlib import Path

import pytest

import d_lightrag as dl
import d_provenance_fixtures as pf
from llm.common import item_provenance as ip

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "fuse_dual_level_golden_pre_provenance.json").read_text())
CASES = {"top5": 5, "top2": 2, "short_chunk": 5}


def _fuse(case):
    return dl.fuse_dual_level(pf.low_candidates(), pf.high_candidates(), CASES[case], 3 if case == "short_chunk" else 400)


@pytest.mark.parametrize("case", list(CASES))
def test_ranking_and_scores_identical_to_pre_provenance_code(case):
    new = [{k: v for k, v in ip.legacy_view(i, drop_rank=True).items() if k != "trust_weight"} for i in _fuse(case)]
    assert new == GOLDEN[case]


@pytest.mark.parametrize("case", list(CASES))
def test_every_item_validates_against_the_schema(case):
    for item in _fuse(case):
        assert ip.validate_item(item) == [], item["answer_id"]


def _by_answer():
    return {i["answer_id"]: i for i in _fuse("top5")}


def test_low_level_item_path_is_anchor_to_answer():
    item = _by_answer()[2001]
    assert item["source_stage"] == "low_level" and item["hop"] == 1
    assert [s["node_id"] for s in item["path"]] == [200, 2001] == [item["anchor_id"], item["answer_id"]]
    assert item["path"][0]["edge_type_to_next"] == "HAS_ANSWER"
    assert item["trust_weight"] == 0.8 and item["relevance_score"] == 0.85


def test_high_level_item_path_is_anchor_q2_answer_and_rel_type_is_unchanged():
    item = _by_answer()[3001]
    assert [(s["node_type"], s["node_id"], s["edge_type_to_next"]) for s in item["path"]] == [
        ("Question", 100, "TAG_COOCCUR"), ("Question", 300, "HAS_ANSWER"), ("Answer", 3001, None)]
    assert item["rel_type"] == "TAG_COOCCUR"                      # D's rel_type stays the r2 type only
    assert item["path"][0]["edge_weight_to_next"] == 0.7 and item["path"][1]["edge_weight_to_next"] == 0.8
    assert item["trust_weight"] == round(0.7 * 0.8, 4)


def test_other_anchor_behind_the_same_high_level_row_is_an_alt_path():
    item = _by_answer()[3001]
    assert item["anchor_id"] == 100
    assert [(a["path"][0]["node_id"], a["relevance_score"]) for a in item["alt_paths"]] == [(200, 0.7)]


def test_dedup_loser_is_kept_as_alt_path():
    item = _by_answer()[1001]                                       # low-level .91 beats high-level .4
    assert item["source_stage"] == "low_level"
    assert [(a["source_stage"], a["hop"], a["relevance_score"]) for a in item["alt_paths"]] == [("high_level", 2, 0.4)]


def test_trust_weight_does_not_influence_the_ranking():
    low, high = pf.low_candidates(), pf.high_candidates()
    base = [i["answer_id"] for i in dl.fuse_dual_level(low, high, 5, 400)]
    for c in low + high:
        c["edge_weight"] = 1.0 - (c.get("edge_weight") or 0.0)      # invert every trust weight
    assert [i["answer_id"] for i in dl.fuse_dual_level(low, high, 5, 400)] == base


def test_candidates_from_older_callers_still_fuse():
    low = [{"via_question_id": 5, "via_question_title": "t", "answer_id": 50, "answer_body": "b", "hop": 1,
            "rel_type": "HAS_ANSWER", "is_accepted": None, "relevance_score": 0.5}]
    item = dl.fuse_dual_level(low, [], 5, 400)[0]
    assert item["trust_weight"] is None and item["anchor_id"] == 5 and len(item["path"]) == 2


# --- candidate generation (fake Neo4j) -----------------------------------------

class _Session:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        return [dict(r) for r in self.rows]


class _Driver:
    def __init__(self, rows):
        self.s = _Session(rows)

    def session(self, database=None):
        return self.s


def test_retrieve_high_level_resolves_the_anchor_and_keeps_the_others():
    rows = [{"via_question_id": 300, "via_question_title": "t", "answer_id": 3001, "answer_body": "b", "hop": 2,
             "rel_type": "TAG_COOCCUR", "relevance_score": 0.7, "is_accepted": False,
             "via_anchor_ids": [100, 200, 150], "r3_type": "HAS_ANSWER", "r3_weight": 0.8}]
    driver = _Driver(rows)
    # anchor cosines: 200 is the most similar anchor
    out = dl.retrieve_high_level(driver, "db", [(100, 0.80), (200, 0.90), (150, 0.80)], set(), set(), 3)
    c = out[0]
    assert (c["anchor_id"], c["alt_anchor_ids"]) == (200, [100, 150])        # tie 100/150 -> lowest id first
    assert c["edge_weight"] == 0.7 * 0.8 and "via_anchor_ids" not in c
    q = driver.s.queries[0]
    # row set / order / LIMIT clauses of the retrieval query are untouched
    assert "ORDER BY relevance_score DESC, q2.id ASC\n            LIMIT $n_high_level" in q
    assert "ORDER BY q2.id ASC, a.id ASC" in q and "EXISTS {" in q
    assert "WITH q2, r2.weight AS relevance_score, type(r2) AS r2_type, collect(DISTINCT anchor_id) AS via_anchor_ids" in q


def test_retrieve_low_level_sets_anchor_and_returns_the_edge_weight():
    rows = [{"via_question_id": 100, "via_question_title": "t", "answer_id": 1, "answer_body": "b", "hop": 1,
             "rel_type": "HAS_ANSWER", "is_accepted": None, "edge_weight": 0.5}]
    driver = _Driver(rows)
    c = dl.retrieve_low_level(driver, "db", [(100, 0.9)], set())[0]
    assert (c["anchor_id"], c["relevance_score"], c["edge_weight"]) == (100, 0.9, 0.5)
    assert "r.weight AS edge_weight" in driver.s.queries[0]
