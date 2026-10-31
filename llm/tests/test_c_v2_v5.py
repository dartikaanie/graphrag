"""C-v2 (c_retrieval_version "v5", Amendment 19). Offline: fake KG answer index, fake Neo4j driver, fake embedder."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_graphrag"))
import c_graphrag as cg  # noqa: E402

from llm.common import anchors as an  # noqa: E402
from llm.common.leakage import build_exclusion  # noqa: E402
from llm.manifest import compute_config_hash  # noqa: E402
from llm.sample_split import FROZEN_DIR, ids_sha256  # noqa: E402


class FakeKG:
    """Answer rows 0..n-1 in rank order (score descending); honours the exclusion like KGAnswerIndex.search."""

    def __init__(self, rows):  # rows: [(answer_id, question_id, score)]
        self.answer_ids = np.array([r[0] for r in rows])
        self.question_ids = np.array([r[1] for r in rows])
        self.scores = [r[2] for r in rows]
        self.n_asked = []

    def search(self, q, top_k, exclusion=None, stats=None):
        self.n_asked.append(top_k)
        out = []
        for row in range(len(self.scores)):
            if exclusion is not None and exclusion.excludes(self.question_ids[row], self.answer_ids[row]):
                continue
            out.append((row, self.scores[row]))
            if len(out) >= top_k:
                break
        return out


def _kg(pairs):  # [(question_id, score)] one answer each, answer id = 1000 + position
    return FakeKG([(1000 + i, q, s) for i, (q, s) in enumerate(pairs)])


def test_anchor_score_is_best_answer_and_ties_go_to_the_smaller_question_id():
    kg = _kg([(7, 0.9), (5, 0.8), (7, 0.7), (3, 0.8), (9, 0.6), (2, 0.5)])
    stats = {}
    anchors, hits = an.select_answer_level_anchors(kg, None, 3, None, stats=stats)
    assert anchors == [(7, 0.9), (3, 0.8), (5, 0.8)]  # 3 before 5 on the 0.8 tie; 7 scored by its best answer
    assert len(hits) == 6 and (stats["anchors_found"], stats["anchors_requested"]) == (3, 3)
    assert kg.n_asked == [10]  # no deep scan needed


def test_deep_scan_to_50_when_fewer_than_three_distinct_questions():
    pairs = [(1, 0.9 - 0.001 * i) for i in range(12)] + [(2, 0.5), (3, 0.4)]  # top 10 answers: question 1 only
    kg = _kg(pairs)
    stats = {}
    anchors, hits = an.select_answer_level_anchors(kg, None, 3, None, stats=stats)
    assert [q for q, _ in anchors] == [1, 2, 3] and len(hits) == 10  # pool stays the top-10 answers
    assert kg.n_asked == [10, 50] and stats["answers_scanned"] == 14
    short = _kg([(1, 0.9), (1, 0.8)])
    stats = {}
    anchors, _ = an.select_answer_level_anchors(short, None, 3, None, stats=stats)
    assert len(anchors) == 1 and (stats["anchors_found"], stats["anchors_requested"]) == (1, 3)


def test_answer_search_excludes_the_thread_before_ranking():
    kg = _kg([(100, 0.99), (100, 0.98), (5, 0.9), (6, 0.8), (7, 0.7)])
    excl = build_exclusion(100, [1000, 1001], None)
    anchors, hits = an.select_answer_level_anchors(kg, None, 3, excl)
    assert [q for q, _ in anchors] == [5, 6, 7]
    assert all(int(kg.question_ids[r]) != 100 and int(kg.answer_ids[r]) not in (1000, 1001) for r, _ in hits)


class FakeDriver:
    """Records the Cypher it is given; returns canned rows."""

    def __init__(self, rows=()):
        self.queries, self.params, self.rows = [], [], list(rows)

    def session(self, database=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        return [dict(r) for r in self.rows]


def test_weight_rule_in_every_query_and_v4_queries_unchanged():
    d4, d5 = FakeDriver(), FakeDriver()
    cg.traverse_graph(d4, "db", [1], {9}, {99})
    cg.traverse_graph(d5, "db", [1], {9}, {99}, trust_weights=True)
    assert len(d4.queries) == 2 and len(d5.queries) == 2
    assert all("coalesce(a.trustScore" not in q and "answer_score" not in q for q in d4.queries)
    assert "r.weight AS edge_weight" in d4.queries[0] and "(r2.weight * r3.weight) AS edge_weight" in d4.queries[1]
    one, two = d5.queries
    rule1 = "CASE WHEN type(r) = 'HAS_ACCEPTED_ANSWER' THEN 1.0 ELSE coalesce(a.trustScore, 0.0) END"
    assert rule1 in one and "r.weight" not in one and "a.score AS answer_score" in one
    assert "(r2.weight * CASE WHEN type(r3) = 'HAS_ACCEPTED_ANSWER' THEN 1.0 ELSE coalesce(a.trustScore, 0.0) END)" in two
    assert "a.score AS answer_score" in two and "LIMIT 50" in two
    pool = FakeDriver()
    cg.fetch_answer_pool(pool, "db", [5], {9}, {99})
    assert rule1 in pool.queries[0] and pool.params[0]["exclude_q"] == [9] and pool.params[0]["exclude_a"] == [99]
    assert cg.fetch_answer_pool(pool, "db", [], set(), set()) == []


def test_leakage_exclusion_reaches_the_traversal_and_pool_queries():
    d = FakeDriver()
    cg.traverse_graph(d, "db", [1], {100}, {1000, 1001}, trust_weights=True)
    assert all(p["exclude_a"] == [1000, 1001] or set(p["exclude_a"]) == {1000, 1001} for p in d.params)
    assert set(d.params[1]["exclude_q"]) == {100}
    assert all("NOT a.id IN $exclude_a" in q for q in d.queries) and "NOT q2.id IN $exclude_q" in d.queries[1]


class FakeEmbed:
    pass


def _cand(aid, q, edge, sim_vec, accepted=False, score=3, hop=1):
    return {"via_question_id": q, "via_question_title": f"t{q}", "answer_id": aid, "answer_body": f"body {aid}",
            "answer_trust_score": 0.1, "is_accepted": accepted, "answer_score": score, "edge_weight": edge,
            "hop": hop, "rel_type": "HAS_ANSWER", "anchor_id": q, "_sim": sim_vec}


@pytest.fixture
def fuse(monkeypatch):
    """fuse_and_rank with a similarity table keyed by the answer body."""
    table = {}
    monkeypatch.setattr(cg, "compute_candidate_similarities",
                        lambda m, qe, texts, ids, cache=None: [table[int(a)] for a in ids])

    def run(graph, expansion=(), top_k=5, alpha=0.5, sims=None):
        table.clear()
        table.update(sims)
        return cg.fuse_and_rank(graph, list(expansion), top_k, 400, c_retrieval_version="v5", alpha=alpha,
                                w_path_trust=1.0, w_intrinsic=0.0, expansion_trust_cap=0.4,
                                embed_model=FakeEmbed(), query_emb=np.ones(4, dtype="float32"))
    return run


def test_fusion_formula_dedup_highest_w_path_and_expansion_cap(fuse):
    g = [_cand(1, 10, 1.0, 0, True), _cand(1, 10, 0.3, 0, True, hop=2), _cand(2, 11, 0.5, 0), _cand(3, 12, 0.2, 0)]
    e = [_cand(4, 13, 0.9, 0)]
    final, meta = fuse(g, e, alpha=0.25, sims={1: 0.2, 2: 0.6, 3: 0.4, 4: 0.8})
    assert meta["candidate_pool_size"] == 4 and meta["n_filtered"] == 0
    by = {f["answer_id"]: f for f in final}
    assert by[1]["trust"] == 1.0  # duplicate kept at the highest w_path
    assert by[4]["trust"] == pytest.approx(0.36)  # expansion: 0.9 * 0.4
    lo, hi = 0.2, 0.8
    for aid, w, sim in ((1, 1.0, 0.2), (2, 0.5, 0.6), (3, 0.2, 0.4), (4, 0.36, 0.8)):
        sim_norm = (sim - lo) / (hi - lo)
        assert by[aid]["final_score"] == pytest.approx(0.75 * sim_norm + 0.25 * w, abs=1e-4)
    assert [f["answer_id"] for f in final] == sorted(by, key=lambda a: (-by[a]["final_score"], a))


def test_filter_drops_negative_unaccepted_and_refills_by_similarity(fuse):
    g = [_cand(1, 10, 0.9, 0, score=-2),             # dropped
         _cand(2, 11, 0.9, 0, accepted=True, score=-5),  # accepted: kept whatever the score
         _cand(3, 12, 0.5, 0, score=0),               # kept (not below 0)
         _cand(4, 13, 0.9, 0, score=-1),              # dropped
         _cand(5, 14, 0.1, 0, score=None)]            # missing score: kept
    sims = {1: 0.9, 2: 0.1, 3: 0.2, 4: 0.5, 5: 0.3}
    final, meta = fuse(g, top_k=3, sims=sims)
    assert {f["answer_id"] for f in final} == {2, 3, 5} and meta["n_filtered"] == 2 and meta["n_refilled"] == 0
    final, meta = fuse(g, top_k=5, sims=sims)  # only 3 kept: refill both dropped, by similarity (1 then 4)
    assert [f["answer_id"] for f in final][3:] == [1, 4] and meta == {"candidate_pool_size": 5, "n_filtered": 2, "n_refilled": 2}
    final, meta = fuse(g, top_k=4, sims=sims)
    assert [f["answer_id"] for f in final][3:] == [1] and meta["n_refilled"] == 1


def test_v4_has_no_filter_keys(monkeypatch):
    monkeypatch.setattr(cg, "compute_candidate_similarities", lambda m, qe, texts, ids, cache=None: [0.5] * len(ids))
    final, meta = cg.fuse_and_rank([_cand(1, 10, 0.5, 0, score=-9)], [], 5, 400, c_retrieval_version="v4", alpha=0.5,
                                   embed_model=FakeEmbed(), query_emb=np.ones(4, dtype="float32"))
    assert meta == {"candidate_pool_size": 1} and len(final) == 1


def _config(version, **kw):
    return cg.build_config("openai", "gpt-4o-mini", 150, 42, 0, 5, 3, 3, "trust_weighted", kw.pop("w", 0.7),
                           kw.pop("wi", 0.3), kw.pop("cap", 0.4), True, True, c_retrieval_version=version, alpha=0.5,
                           leakage_mode="thread", **kw)


def test_v5_config_hash_differs_pins_fusion_and_v4_hash_is_unchanged():
    c4, c5 = _config("v4"), _config("v5")
    assert compute_config_hash(c4) != compute_config_hash(c5)
    assert c5["c_retrieval_version"] == "v5" and "v5" in c5 and "v5" not in c4
    assert (c5["fusion_w_path_trust"], c5["fusion_w_intrinsic"], c5["semantic_expansion_trust_cap"]) == (1.0, 0.0, 0.4)
    assert c5 == _config("v5", w=0.1, wi=0.9, cap=0.9)  # flags cannot move the v5 pins
    assert (c4["fusion_w_path_trust"], c4["fusion_w_intrinsic"]) == (0.7, 0.3)
    assert "v5" not in cg.C_RETRIEVAL_VERSION and cg.C_RETRIEVAL_VERSION == "v4"  # default untouched


def test_devlarge_ids_150_no_overlap_with_test_and_dev():
    def load(n):
        d = json.loads((FROZEN_DIR / n).read_text())
        assert ids_sha256(d["ids"]) == d["ids_sha256"]
        return d
    big, test, dev = load("devlarge_ids_v1.json"), load("test_ids_v1.json"), load("dev_ids_v1.json")
    assert len(big["ids"]) == 150 == len(set(big["ids"])) and big["dev_offset"] == 435
    assert not set(big["ids"]) & (set(test["ids"]) | set(dev["ids"]))


def test_devlarge_slice_is_positions_435_to_584_in_both_samplers():
    import pandas as pd

    from llm.sample_split import sample_questions_split
    df = pd.DataFrame({"Id": range(700)})
    perm = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    for fn in (sample_questions_split, cg.sample_questions_split):
        got = fn(df, 150, 42, "devlarge")
        assert list(got["Id"]) == list(perm.iloc[434:584]["Id"])


def test_retrieval_only_v5_writes_one_record_per_question(tmp_path, monkeypatch):
    import pandas as pd

    kg = _kg([(7, 0.9), (5, 0.8), (3, 0.7), (9, 0.6)])
    cand = lambda aid, q, score: {**_cand(aid, q, 0.5, 0, score=score), "hop": 1}  # noqa: E731
    monkeypatch.setattr(cg, "traverse_graph", lambda *a, **k: [cand(2001, 7, 3), cand(2002, 5, -4)])
    monkeypatch.setattr(cg, "semantic_expansion", lambda *a, **k: [])
    monkeypatch.setattr(cg, "fetch_answer_pool", lambda *a, **k: [cand(3001, 9, 1)])
    monkeypatch.setattr(cg, "compute_candidate_similarities",
                        lambda m, qe, texts, ids, cache=None: [0.1 * (i + 1) for i in range(len(ids))])
    monkeypatch.setattr(cg, "log", lambda *a, **k: None)
    qids = [101, 102]
    df = pd.DataFrame({"Id": qids, "Title": ["a", "b"], "Body": ["x", "y"], "Tags": ["t", "t"],
                       "AcceptedAnswerId": [1, 2], "AcceptedAnswerBody": ["g", "g"], "n_tokens": [5, 5]})
    emb = np.ones((2, 4), dtype="float32")
    out = tmp_path / "o.jsonl"
    cfg = _config("v5")
    results, stats = cg.process_sample(
        df, None, None, FakeEmbed(), FakeDriver(), "db", None, None, emb, {101: 0, 102: 1}, {101: [], 102: []},
        3, 3, 3, 400, "m", out, c_retrieval_version="v5", alpha=0.5, retrieval_only=True, kg_index=kg,
        config=cfg, config_hash=compute_config_hash(cfg))
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == len(results) == 2 and [r["question_id"] for r in lines] == qids
    for r in lines:
        assert r["retrieved_context"] and r["anchors_found"] == 3 and r["anchors_requested"] == 3
        assert r["v5_n_filtered"] == 1 and r["v5_n_refilled"] == 1 and "retrieval_latency_sec" in r
    assert stats["c_v2"]["n_filtered"] == 2


def test_failed_empty_run_is_cleared_but_nothing_else(tmp_path):
    out = tmp_path / "r.jsonl"
    man = tmp_path / "r.jsonl.manifest.json"
    out.write_text("")
    man.write_text(json.dumps({"status": "failed"}))
    assert cg.clear_failed_empty_run(out, lambda *a: None) and not out.exists() and not man.exists()
    out.write_text('{"question_id": 1}\n')
    man.write_text(json.dumps({"status": "failed"}))
    assert not cg.clear_failed_empty_run(out, lambda *a: None) and out.exists() and man.exists()
    out.write_text("")
    man.write_text(json.dumps({"status": "completed"}))
    assert not cg.clear_failed_empty_run(out, lambda *a: None) and out.exists()
