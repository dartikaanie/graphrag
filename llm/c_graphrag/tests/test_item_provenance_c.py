"""Condition C item provenance (llm/common/item_provenance.py), offline.

* Regression: with provenance added, fuse_and_rank()'s items (provenance keys
  dropped) equal -- value for value, order for order -- what the code BEFORE
  provenance produced on the same candidates. The golden file
  fixtures/fuse_and_rank_golden_pre_provenance.json was generated from the
  committed HEAD (6af872f) c_graphrag.py by running c_provenance_fixtures.run_config()
  on it, before any provenance edit.
* Provenance: 2-hop path = anchor -> q2 -> answer, semantic path starts at its
  seed, alt_paths keep the paths that lost the dedup, schema validates.
* semantic_expansion() attributes every expanded Question to exactly one seed.

The live-Neo4j check (paths re-validated against the KG) is in
test_item_provenance_c_live.py (GRAPHRAG_INTEGRATION=1)."""

import json
from pathlib import Path

import pytest

import c_graphrag as cg
import c_provenance_fixtures as pf
import test_expansion_answered as tea
from llm.common import item_provenance as ip

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "fuse_and_rank_golden_pre_provenance.json").read_text())


@pytest.mark.parametrize("label", list(pf.CONFIGS))
def test_ranking_and_scores_identical_to_pre_provenance_code(label):
    out = pf.run_config(cg.fuse_and_rank, label)
    new_items = [ip.legacy_view(i, drop_rank=label.startswith("v2")) for i in out["items"]]
    assert new_items == GOLDEN[label]["items"]
    assert out["meta"] == GOLDEN[label]["meta"]


@pytest.mark.parametrize("label", list(pf.CONFIGS))
def test_every_item_validates_against_the_schema(label):
    items = pf.run_config(cg.fuse_and_rank, label)["items"]
    assert items
    for item in items:
        assert ip.validate_item(item) == [], (label, item["answer_id"])
    assert [i["rank"] for i in items] == list(range(1, len(items) + 1))


def _by_answer(label):
    return {i["answer_id"]: i for i in pf.run_config(cg.fuse_and_rank, label)["items"]}


def test_two_hop_items_have_anchor_q2_answer_paths():
    two_hop = [i for i in pf.run_config(cg.fuse_and_rank, "v2_trust_weighted")["items"] if i["hop"] == 2]
    assert two_hop
    for item in two_hop:
        assert len(item["path"]) == 3
        assert item["path"][0]["node_id"] == item["anchor_id"]
        assert item["path"][1]["node_id"] == item["question_id"] == item["via_question_id"]
        assert item["path"][2]["node_id"] == item["answer_id"]
    i3002 = _by_answer("v2_trust_weighted")[3002]
    assert [(s["node_type"], s["node_id"], s["edge_type_to_next"], s["edge_weight_to_next"]) for s in i3002["path"]] == [
        ("Question", 100, "IS_RELATED_TO", 0.9), ("Question", 302, "HAS_ANSWER", 0.9), ("Answer", 3002, None, None)]


def test_one_hop_item_anchor_is_its_via_question():
    item = _by_answer("v2_trust_weighted")[2001]
    assert item["anchor_id"] == item["via_question_id"] == 200
    assert [s["node_id"] for s in item["path"]] == [200, 2001]
    assert item["path"][0]["edge_weight_to_next"] == 0.8     # the KG edge weight


def test_dedup_winner_path_and_alt_paths_trust_weighted():
    # 3001: anchor-100 2-hop (combined .46) beats expansion via seed 300 (.432) and anchor-200 2-hop (.404)
    item = _by_answer("v2_trust_weighted")[3001]
    assert item["anchor_id"] == 100 and item["hop"] == 2
    alts = item["alt_paths"]
    assert [(a["source_stage"], a["path"][0]["node_id"], a["combined_score"]) for a in alts] == [
        ("semantic_expansion", 300, 0.432), ("graph_traversal", 200, 0.404)]
    # 1001: the 1-hop accepted edge wins over the 2-hop route via q2=301
    item = _by_answer("v2_trust_weighted")[1001]
    assert item["hop"] == 1 and [a["hop"] for a in item["alt_paths"]] == [2]


def test_parallel_edges_to_one_answer_keep_the_other_edge_type_as_alt_path():
    item = _by_answer("v2_trust_weighted")[1002]
    assert item["rel_type"] == "HAS_ANSWER"                                   # first seen wins the tie (unchanged rule)
    assert [a["rel_type"] for a in item["alt_paths"]] == ["HAS_ACCEPTED_ANSWER"]
    assert item["alt_paths"][0]["path"][0]["edge_type_to_next"] == "HAS_ACCEPTED_ANSWER"


def test_uniform_mode_first_seen_wins_and_alts_are_kept():
    item = _by_answer("v2_uniform")[3001]
    assert item["anchor_id"] == 100 and len(item["alt_paths"]) == 2


def test_semantic_items_start_at_their_seed():
    items = pf.run_config(cg.fuse_and_rank, "v3_alpha_0.5")["items"]
    sem = [i for i in items if i["source_stage"] == "semantic_expansion"]
    assert sem
    for item in sem:
        assert item["path"][0]["node_id"] is not None and item["anchor_id"] == item["path"][0]["node_id"]
        assert item["path"][0]["edge_type_to_next"] == "SEMANTIC_SIM"
        assert item["seed_similarity"] == item["path"][0]["edge_weight_to_next"]
        assert item["path"][1]["node_id"] == item["via_question_id"]
    i4001 = next(i for i in sem if i["answer_id"] == 4001)
    assert (i4001["anchor_id"], i4001["seed_similarity"]) == (300, 0.8123)
    # trust_weight keeps its meaning (the proxy); the KG edge's own weight is in the path
    assert i4001["trust_weight"] == round(0.7 * 0.4, 4) and i4001["path"][1]["edge_weight_to_next"] == 0.7


def test_candidates_from_older_callers_still_fuse():
    """No anchor_id / r2_weight / seed keys (e.g. the existing fuse_and_rank tests)."""
    graph = [{"via_question_id": 5, "via_question_title": "t", "answer_id": 50, "answer_body": "b",
              "answer_trust_score": 0.1, "is_accepted": False, "edge_weight": 0.3, "hop": 1, "rel_type": "HAS_ANSWER"},
             {"via_question_id": 6, "via_question_title": "t", "answer_id": 60, "answer_body": "b",
              "answer_trust_score": 0.1, "is_accepted": False, "edge_weight": 0.3, "hop": 2,
              "rel_type": "TAG_COOCCUR->HAS_ANSWER"}]
    items, _ = cg.fuse_and_rank(graph, [], 5, 400)
    assert items[0]["anchor_id"] == 5                  # 1-hop: anchor == via_question_id
    assert items[1]["anchor_id"] is None               # 2-hop: origin unknown -> None, no crash
    assert len(items[1]["path"]) == 3


# --- candidate generation ----------------------------------------------------

@pytest.mark.parametrize("require_answer", [True, False])
def test_semantic_expansion_records_the_triggering_seed(monkeypatch, require_answer):
    ranked = {"s1": tea._ranked([(1001, .91), (1002, .82)]), "s2": tea._ranked([(1003, .73), (1004, .64)])}
    cands, stats, _, _ = tea._run(monkeypatch, ranked, {1001, 1002, 1003, 1004}, require_answer)
    seeds = {c["via_question_id"]: (c["seed_id"], c["seed_similarity"]) for c in cands}
    assert seeds[1001] == (900, 0.91) and seeds[1002] == (900, 0.82)       # seed 900 = title "s1"
    assert seeds[1003] == (901, 0.73) and seeds[1004] == (901, 0.64)
    assert all(c["seed_id"] is not None for c in cands)


class _Session:
    def __init__(self, rows_by_hop):
        self.rows, self.queries = rows_by_hop, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        return list(self.rows["2" if "r2:" in query else "1"])


class _Driver:
    def __init__(self, rows_by_hop):
        self.session_obj = _Session(rows_by_hop)

    def session(self, database=None):
        return self.session_obj


def test_traverse_graph_adds_anchor_ids_and_the_cypher_carries_them():
    one = {"via_question_id": 100, "via_question_title": "t", "answer_id": 1, "answer_body": "b",
           "answer_trust_score": 0.1, "is_accepted": True, "edge_weight": 1.0, "hop": 1, "rel_type": "HAS_ACCEPTED_ANSWER"}
    two = {"via_question_id": 300, "via_question_title": "t", "answer_id": 2, "answer_body": "b",
           "answer_trust_score": 0.1, "is_accepted": False, "edge_weight": 0.4, "hop": 2,
           "rel_type": "TAG_COOCCUR->HAS_ANSWER", "anchor_id": 100, "r2_weight": 0.5, "r3_weight": 0.8}
    driver = _Driver({"1": [one], "2": [two]})
    out = cg.traverse_graph(driver, "db", [100], set(), set())
    assert [(c["hop"], c["anchor_id"]) for c in out] == [(1, 100), (2, 100)]
    q2 = driver.session_obj.queries[1]
    assert "WITH anchor_id, q2, a, r2, r3," in q2 and "anchor_id, r2.weight AS r2_weight, r3.weight AS r3_weight" in q2
    # retrieval clauses unchanged
    assert "ORDER BY edge_weight DESC, a.id ASC" in q2 and "LIMIT 50" in q2
