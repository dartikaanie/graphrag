"""Condition D retrieval v4 (Amendment 16): ONE score on both levels = cosine(query, owner Question), offline.

A fake Neo4j driver answers the three Cypher queries from a small in-memory graph (it honours the exclusion
parameters the same way the real queries do) and the embeddings are a tiny normalised matrix. v3 stays callable."""

import numpy as np
import pytest

import d_lightrag as dl
from llm.common.leakage import LeakageError, build_exclusion, assert_no_leakage
from llm.manifest import compute_config_hash

DIM = 8


def _unit(*v):
    a = np.zeros(DIM, dtype=np.float32)
    a[:len(v)] = v
    return a / np.linalg.norm(a)


# question id -> embedding; the query is the evaluation question 1 (excluded from every result)
EMB = {1: _unit(1, 0), 10: _unit(1, 0.1), 20: _unit(1, 0.4), 30: _unit(1, 0.2), 31: _unit(1, 0.9), 32: _unit(1, 0.3),
       33: _unit(1, 0.3), 34: _unit(0, 1), 35: _unit(1, 0.05)}
IDS = sorted(EMB)
MATRIX = np.stack([EMB[q] for q in IDS])
ID_TO_ROW = {q: i for i, q in enumerate(IDS)}
QUERY = EMB[1]

# edges (anchor -> neighbour, type). Edge types/weights are deliberately NOT aligned with the cosine order: 34 is
# TAG_COOCCUR (v3 score 0.6) but far from the query; 35 is EMBED_SIM (v3 score 0.4) and the closest.
EDGES = [(10, 30, "TAG_COOCCUR"), (10, 31, "EMBED_SIM"), (10, 34, "TAG_COOCCUR"), (10, 35, "EMBED_SIM"),
         (20, 31, "IS_RELATED_TO"), (20, 32, "EMBED_SIM"), (20, 33, "EMBED_SIM"), (10, 1, "IS_RELATED_TO")]
ANSWERS = {10: [(1001, "HAS_ACCEPTED_ANSWER", True)], 20: [(2001, "HAS_ANSWER", False)],
           30: [(3001, "HAS_ANSWER", False)], 31: [(3101, "HAS_ACCEPTED_ANSWER", True), (3102, "HAS_ANSWER", False)],
           32: [(3201, "HAS_ANSWER", False)], 33: [(3301, "HAS_ANSWER", False)], 34: [(3401, "HAS_ANSWER", False)],
           35: [(3501, "HAS_ANSWER", False)], 1: [(9001, "HAS_ACCEPTED_ANSWER", True), (9002, "HAS_ANSWER", False)]}


class _Result(list):
    def data(self):
        return [dict(r) for r in self]


class FakeDriver:
    def __init__(self):
        self.queries = []

    def session(self, database=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _answers(self, q, exclude_a):
        return [(a, t, acc) for a, t, acc in ANSWERS.get(q, []) if a not in exclude_a]

    def run(self, query, **p):
        self.queries.append(query)
        ex_a, ex_q = set(p.get("exclude_a", [])), set(p.get("exclude_q", []))
        if "RETURN q2.id AS qid" in query:  # v4 neighbour query
            rows = [{"qid": q2, "anchor_id": a, "r2_type": t} for a in p["anchor_ids"] for (aa, q2, t) in EDGES
                    if aa == a and q2 not in ex_q and self._answers(q2, ex_a)]
            return _Result(sorted(rows, key=lambda r: (r["qid"], r["anchor_id"], r["r2_type"])))
        if "UNWIND $qids" in query:  # v4 answers of the chosen neighbours
            rows = [{"via_question_id": q, "via_question_title": f"title {q}", "answer_id": a, "answer_body": f"body {a}",
                     "is_accepted": acc, "r3_type": t} for q in p["qids"] for a, t, acc in self._answers(q, ex_a)]
            return _Result(sorted(rows, key=lambda r: (r["via_question_id"], r["answer_id"])))
        assert "anchor.title AS via_question_title" in query  # low-level 1-hop (shared with v3)
        rows = [{"via_question_id": q, "via_question_title": f"title {q}", "answer_id": a, "answer_body": f"body {a}",
                 "hop": 1, "rel_type": t, "is_accepted": acc} for q in p["anchor_ids"] for a, t, acc in self._answers(q, ex_a)]
        return _Result(sorted(rows, key=lambda r: r["answer_id"]))


ANCHORS = [(10, 0.99), (20, 0.8)]
EXCL = build_exclusion(1, [9001, 9002], 9001)


def _levels(driver=None, n_high=3, cap=dl.HIGH_LEVEL_CANDIDATE_CAP, stats=None):
    driver = driver or FakeDriver()
    low = dl.retrieve_low_level_v4(driver, "g", ANCHORS, set(EXCL.answer_ids), QUERY, MATRIX, ID_TO_ROW)
    high = dl.retrieve_high_level_v4(driver, "g", ANCHORS, set(EXCL.question_ids), set(EXCL.answer_ids), n_high,
                                     QUERY, MATRIX, ID_TO_ROW, cap=cap, stats=stats)
    return low, high, driver


def test_both_levels_are_scored_by_the_same_question_cosine():
    low, high, _ = _levels()
    scores = dl.question_scores(QUERY, MATRIX, ID_TO_ROW, EMB)
    assert low and high
    for c in low + high:
        assert c["relevance_score"] == scores[c["via_question_id"]], c
    # a Question that is both an anchor-side owner and a neighbour would get the very same number
    assert dl.question_scores(QUERY, MATRIX, ID_TO_ROW, [31])[31] == scores[31]


def test_high_level_picks_by_cosine_not_by_edge_type_or_weight():
    _, high, _ = _levels(n_high=3)
    picked = sorted({c["via_question_id"] for c in high})
    # neighbours of anchors 10/20: 30, 31, 32, 33, 34, 35. By cosine to the query: 35 > 30 > 32 = 33 > 31 > 34.
    assert picked == [30, 32, 35] or picked == [30, 33, 35]
    assert 34 not in picked  # TAG_COOCCUR (old score 0.6) but semantically far
    assert 35 in picked      # EMBED_SIM (old score 0.4) but the closest


def test_no_edge_weight_is_read_or_used():
    _, _, driver = _levels()
    assert driver.queries and not any("weight" in q.lower() for q in driver.queries)
    low, high, _ = _levels()
    for c in low + high:
        assert not {"edge_weight", "r2_weight", "r3_weight"} & set(c)


def test_ties_are_deterministic():
    # 32 and 33 have identical embeddings -> same score -> ordered by question id, then answer id
    chosen, _ = dl.select_high_level_questions({32: [(20, "EMBED_SIM")], 33: [(20, "EMBED_SIM")]}, {20: 0.8},
                                               {32: 0.5, 33: 0.5}, 1)
    assert [q for q, *_ in chosen] == [32]
    runs = [[(c["via_question_id"], c["answer_id"], c["relevance_score"]) for c in _levels()[1]] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    fused = [dl.fuse_dual_level(*_levels()[:2], 5, 400) for _ in range(2)]
    assert [i["answer_id"] for i in fused[0]] == [i["answer_id"] for i in fused[1]]
    tied = [{"via_question_id": 5, "via_question_title": "t", "answer_id": a, "answer_body": "b", "hop": 1,
             "rel_type": "HAS_ANSWER", "relevance_score": 0.5, "is_accepted": False, "anchor_id": 5} for a in (9, 3, 7)]
    assert [i["answer_id"] for i in dl.fuse_dual_level(tied, [], 5, 400)] == [3, 7, 9]


def test_cap_keeps_the_smallest_ids_and_is_reported():
    stats = {}
    _, high, _ = _levels(cap=3, stats=stats)  # neighbours 30..35 -> 6 > cap 3 -> ids 30, 31, 32 kept
    assert stats["cap_hit"] is True and stats["n_neighbours"] == 6 and stats["n_scored"] == 3 and stats["cap"] == 3
    assert {c["via_question_id"] for c in high} <= {30, 31, 32}
    stats = {}
    _levels(stats=stats)
    assert stats["cap_hit"] is False and stats["cap"] == 100


def test_no_leakage_of_the_evaluation_question_or_its_thread():
    low, high, _ = _levels()
    fused = dl.fuse_dual_level(low, high, 5, 400)
    assert_no_leakage(fused, EXCL)  # raises LeakageError otherwise
    assert not {1, 9001, 9002} & ({c["via_question_id"] for c in low + high} | {c["answer_id"] for c in low + high})
    with pytest.raises(LeakageError):  # the check itself is live: an item from the thread is caught
        assert_no_leakage([{"question_id": 1, "answer_id": 9002}], EXCL)


def test_no_trust_accepted_bonus_or_expansion():
    low, high, _ = _levels()
    fused = dl.fuse_dual_level(low, high, 5, 400)
    assert fused
    for item in fused:
        assert item["trust_weight"] is None and item["source_stage"] in ("low_level", "high_level")
        assert not {"combined_score", "final_score", "trust", "sim_norm", "alpha"} & set(item)
        assert item.get("seed_similarity") is None
    # an accepted answer does not outrank a same-question non-accepted one: scores per question are equal
    q31 = {i["answer_id"]: i["relevance_score"] for i in fused if i["question_id"] == 31}
    assert len(set(q31.values())) <= 1
    src = open(dl.__file__).read()
    v4 = src[src.index("def retrieve_low_level_v4"):src.index("# FUSION -- dedup")]
    assert "trustScore" not in v4 and "semantic" not in v4.lower().replace("semantic expansion", "")


def test_config_hash_changes_with_the_retrieval_version():
    args = ("openai", "gpt-4o-mini", 384, 42, 1536, 5, 3, 3, True)
    v3 = dl.build_config(*args, d_retrieval_version="v3")
    v4 = dl.build_config(*args, d_retrieval_version="v4")
    assert v3["d_retrieval_version"] == "v3" and v4["d_retrieval_version"] == "v4"
    assert compute_config_hash(v3) != compute_config_hash(v4)
    v5 = dl.build_config(*args, d_retrieval_version="v5")
    assert len({compute_config_hash(c) for c in (v3, v4, v5)}) == 3
    assert dl.build_config(*args) == v5 and dl.D_RETRIEVAL_VERSION == "v5"  # v5 (Amendment 17) is the default now
    assert dl.build_output_path("r", "openai", "gpt-4o-mini", 384, 42, True, d_retrieval_version="v3") != \
        dl.build_output_path("r", "openai", "gpt-4o-mini", 384, 42, True, d_retrieval_version="v4")


def test_v3_stays_reproducible_and_unknown_versions_are_refused():
    assert callable(dl.retrieve_high_level) and callable(dl.retrieve_low_level)
    assert dl.D_RETRIEVAL_VERSIONS == ("v3", "v4", "v5")
    with pytest.raises(ValueError):
        dl.process_sample(None, None, None, None, None, "g", None, None, None, {}, {}, 5, 3, 3, 400, "m", None,
                          d_retrieval_version="v9")


def test_official_matrix_pins_the_same_version():
    from llm import official_run_matrix as orm

    assert orm.D_RETRIEVAL_VERSION == dl.D_RETRIEVAL_VERSION
    d = [r for r in orm.build_matrix(None) if r["condition"] == "D"][0]
    assert d["argv"][d["argv"].index("--d-retrieval-version") + 1] == "v5"
