"""llm/common/item_provenance.py: builders, schema validation, KG-path validation
(against a fake edge set -- the live-Neo4j validation is in the C/D integration
tests), alt_paths ordering. No Neo4j, no LLM."""

import copy

import pytest

from llm.common import item_provenance as ip


def _item(stage, hop, path, **over):
    base = {"question_id": path[-2]["node_id"] if len(path) > 1 else 7, "answer_id": path[-1]["node_id"],
            "chunk_text": "t", "source_stage": stage, "hop": hop, "is_accepted": None, "trust_weight": 0.5,
            "rel_type": "->".join(s["edge_type_to_next"] for s in path[:-1]) or None}
    base.update(ip.provenance_fields(source_stage=stage, hop=hop, path=path, rank=1,
                                     via_question_id=base["question_id"]))
    base.update(over)
    return base


ONE = lambda: _item("graph_traversal", 1, ip.one_hop_path(10, 100, "HAS_ACCEPTED_ANSWER"))
TWO = lambda: _item("graph_traversal", 2, ip.two_hop_path(10, 20, 200, "TAG_COOCCUR", "HAS_ANSWER"))
SEM = lambda: _item("semantic_expansion", "semantic", ip.semantic_path(20, 30, 300, "HAS_ANSWER"),
                    rel_type="HAS_ANSWER", seed_similarity=0.71)
LOW = lambda: _item("low_level", 1, ip.one_hop_path(10, 100, "HAS_ANSWER"), trust_weight=None)
HIGH = lambda: _item("high_level", 2, ip.two_hop_path(10, 20, 200, "EMBED_SIM", "HAS_ANSWER"),
                     rel_type="EMBED_SIM", trust_weight=None)
VEC = lambda: _item("vector_only", None, ip.vector_only_path(400), question_id=40, trust_weight=None, rel_type=None)
VEC_FIX = lambda: {**VEC(), "via_question_id": 40, "owner_question_id": 40}


@pytest.mark.parametrize("make", [ONE, TWO, SEM, LOW, HIGH, VEC_FIX])
def test_valid_items(make):
    assert ip.validate_item(make()) == []


def test_two_hop_path_is_anchor_q2_answer():
    item = TWO()
    assert [s["node_id"] for s in item["path"]] == [10, 20, 200]
    assert item["anchor_id"] == item["path"][0]["node_id"] == 10
    assert item["via_question_id"] == item["question_id"] == 20


def test_vector_only_has_no_anchor():
    item = VEC_FIX()
    assert item["anchor_id"] is None and item["hop"] is None and item["trust_weight"] is None
    assert len(item["path"]) == 1


@pytest.mark.parametrize("mutate, fragment", [
    (lambda i: i.pop("path"), "missing keys"),
    (lambda i: i.update(source_stage="bogus"), "unknown source_stage"),
    (lambda i: i.update(hop=3), "hop 3 not allowed"),
    (lambda i: i.update(anchor_id=999), "anchor_id"),
    (lambda i: i["path"][-1].update(node_id=1), "path ends at"),
    (lambda i: i["path"][0].update(node_id=None), "node_id None"),
    (lambda i: i.update(rel_type="IS_RELATED_TO->HAS_ANSWER"), "rel_type"),
    (lambda i: i.update(rank=0), "rank"),
    (lambda i: i.update(via_question_id=1), "via_question_id"),
])
def test_invalid_two_hop_items_are_flagged(mutate, fragment):
    item = copy.deepcopy(TWO())
    mutate(item)
    errs = ip.validate_item(item)
    assert errs and any(fragment in e for e in errs), errs


def test_two_hop_with_a_one_hop_path_is_flagged():
    item = TWO()
    item["path"] = item["path"][1:]
    assert any("2-hop path" in e for e in ip.validate_item(item))


def test_semantic_requires_similarity_and_seed():
    item = SEM()
    item["seed_similarity"] = None
    assert any("seed_similarity" in e for e in ip.validate_item(item))
    item = SEM()
    item["path"][0]["edge_type_to_next"] = "TAG_COOCCUR"      # a seed link must be SEMANTIC_SIM
    assert any("semantic path" in e for e in ip.validate_item(item))


@pytest.mark.parametrize("make", [LOW, HIGH])
def test_d_stages_reject_a_trust_weight(make):
    item = make()
    item["trust_weight"] = 0.3
    assert any("trust_weight" in e for e in ip.validate_item(item))


def test_vector_only_rejects_trust_weight_and_anchor():
    item = VEC_FIX()
    item["trust_weight"] = 0.3
    assert any("trust_weight" in e for e in ip.validate_item(item))
    item = VEC_FIX()
    item["anchor_id"] = 5
    assert any("anchor_id" in e for e in ip.validate_item(item))


def test_validate_record_checks_version_and_items():
    rec = {"provenance_version": "v3", "retrieved_context": [ONE(), TWO()]}
    assert ip.validate_record(rec) == []
    assert ip.validate_record({"retrieved_context": [ONE()]})  # legacy record: no version
    bad = {"provenance_version": "v3", "retrieved_context": [ONE(), {**TWO(), "anchor_id": 1}]}
    assert any(e.startswith("item[1]") for e in ip.validate_record(bad))


# --- KG validation --------------------------------------------------------

KG = {("Question", 10, "HAS_ACCEPTED_ANSWER", "Answer", 100): [1.0],
      ("Question", 10, "TAG_COOCCUR", "Question", 20): [0.6],
      ("Question", 20, "HAS_ANSWER", "Answer", 200): [0.9]}


def _lookup(*key):
    return KG.get(key, [])


def test_kg_validation_accepts_real_edges_and_skips_semantic_link():
    assert ip.validate_item_in_kg(ONE(), _lookup) == []
    assert ip.validate_item_in_kg(TWO(), _lookup) == []
    sem = SEM()
    KG[("Question", 30, "HAS_ANSWER", "Answer", 300)] = [0.8]
    try:
        assert ip.validate_item_in_kg(sem, _lookup) == []     # (20)-SEMANTIC_SIM->(30) is not looked up
    finally:
        del KG[("Question", 30, "HAS_ANSWER", "Answer", 300)]


def test_kg_validation_flags_missing_edge():
    item = TWO()
    item["path"][1]["node_id"] = 21                      # q2 that does not exist in the fake KG
    assert any("no HAS_ANSWER edge" in e or "no TAG_COOCCUR edge" in e for e in ip.validate_item_in_kg(item, _lookup))


def test_kg_validation_covers_alt_paths():
    item = ONE()
    item["alt_paths"] = [{"path": ip.one_hop_path(11, 100, "HAS_ANSWER"), "source_stage": "graph_traversal",
                          "hop": 1, "rel_type": "HAS_ANSWER"}]
    assert any(e.startswith("alt_paths[0]") for e in ip.validate_item_in_kg(item, _lookup))


# --- alt_paths ------------------------------------------------------------

def _cand(aid, path, stage="graph_traversal", hop=1, rel="HAS_ANSWER", score=0.5):
    return {"answer_id": aid, "_path": path, "source_stage": stage, "hop": hop, "rel_type": rel, "combined_score": score}


def test_alt_paths_exclude_winner_and_exact_duplicates_and_are_ordered():
    win = _cand(1, ip.one_hop_path(10, 1, "HAS_ANSWER"), score=0.9)
    dup = _cand(1, ip.one_hop_path(10, 1, "HAS_ANSWER"), score=0.9)                    # same path+rel_type
    low = _cand(1, ip.one_hop_path(11, 1, "HAS_ANSWER"), score=0.4)
    high = _cand(1, ip.two_hop_path(12, 13, 1, "TAG_COOCCUR", "HAS_ANSWER"), hop=2,
                 rel="TAG_COOCCUR->HAS_ANSWER", score=0.6)
    other_answer = _cand(2, ip.one_hop_path(10, 2, "HAS_ANSWER"))
    alts = ip.alt_paths_for(win, [low, other_answer, win, dup, high], "combined_score")
    assert [a["combined_score"] for a in alts] == [0.6, 0.4]
    assert alts == ip.alt_paths_for(win, [high, win, low, dup], "combined_score")           # input order irrelevant


def test_legacy_view_drops_only_added_keys():
    item = {**TWO(), "extra": 1}
    legacy = ip.legacy_view(item)
    assert set(item) - set(legacy) == {"via_question_id", "owner_question_id", "anchor_id", "path", "alt_paths", "seed_similarity"}
    assert "rank" not in ip.legacy_view(item, drop_rank=True) and "rank" in legacy


# --- record-level trace -----------------------------------------------------

def _tc(aid, path):
    return {"answer_id": aid, "_path": path}


def test_build_trace_lists_unselected_ids_dedups_and_caps():
    selected = [ONE()]                                              # answer 100 via question 10
    cands = [_tc(100, ip.one_hop_path(10, 100, "HAS_ANSWER")),
             _tc(200, ip.two_hop_path(10, 20, 200, "TAG_COOCCUR", "HAS_ANSWER")),
             _tc(200, ip.two_hop_path(11, 20, 200, "TAG_COOCCUR", "HAS_ANSWER")),
             _tc(300, ip.one_hop_path(30, 300, "HAS_ANSWER"))]
    t = ip.build_trace(cands, selected, [[200, 2, 0.5], [300, 3, 0.4]])
    assert t["not_selected"] == {"answer_ids": [200, 300], "question_ids": [20, 11, 30], "n_answer_ids": 2, "n_question_ids": 3}
    assert t["below_cutoff"] == [[200, 2, 0.5], [300, 3, 0.4]]
    many = [_tc(1000 + i, ip.one_hop_path(5000 + i, 1000 + i, "HAS_ANSWER")) for i in range(150)]
    t = ip.build_trace(many, [], [[i, i, 0.1] for i in range(50)])
    assert len(t["not_selected"]["answer_ids"]) == len(t["not_selected"]["question_ids"]) == 100
    assert t["not_selected"]["n_answer_ids"] == 150 and len(t["below_cutoff"]) == 20


def test_validate_record_requires_the_trace_for_c_and_d_only():
    base = {"provenance_version": "v3", "retrieved_context": [ONE()]}
    assert ip.validate_record({**base, "condition": "B"}) == []
    assert any("not_selected" in e for e in ip.validate_record({**base, "condition": "C"}))
    ok = {**base, "condition": "D", **ip.build_trace([], [], [])}
    assert ip.validate_record(ok) == []
    assert ip.validate_record({**ok, "below_cutoff": [[1, 2, 0.1]] * 21})
