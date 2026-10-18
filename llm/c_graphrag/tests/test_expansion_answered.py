"""Amendment 9 -- answerable-only semantic expansion, offline with a fake
FAISS index, embedding model and Neo4j driver."""

import numpy as np
import pytest

import c_graphrag as cg


class FakeModel:
    """encode([title]) -> a 1-dim vector carrying the seed's number (title "s<N>")."""

    def encode(self, texts, convert_to_numpy=True, **kw):
        return np.array([[float(texts[0][1:])]], dtype="float32")


class FakeIndex:
    """search(): per seed number, a fixed ranked list of (row, score)."""

    def __init__(self, ranked: dict):
        self.ranked = ranked
        self.calls = []

    def search(self, q_emb, n):
        key = self._key
        self.calls.append((key, n))
        rows = self.ranked[key][:n]
        scores = np.array([[s for _, s in rows] + [-1.0] * (n - len(rows))], dtype="float32")
        idxs = np.array([[r for r, _ in rows] + [-1] * (n - len(rows))], dtype="int64")
        return scores, idxs


class Result(list):
    pass


class FakeSession:
    def __init__(self, driver):
        self.d = driver

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, **params):
        self.d.queries.append((query, params))
        if "EXISTS" in query:
            return [{"qid": q} for q in params["qids"] if q in self.d.answerable]
        out = []
        for q in params["qids"]:
            for a in self.d.answers.get(q, []):
                if a not in params["exclude_a"]:
                    out.append({"via_question_id": q, "via_question_title": f"t{q}", "answer_id": a,
                                "answer_body": "b", "answer_trust_score": 0.5, "is_accepted": False,
                                "edge_weight": 0.5, "hop": "semantic", "rel_type": "HAS_ANSWER"})
        return out


class FakeDriver:
    def __init__(self, answerable, answers):
        self.answerable, self.answers, self.queries = set(answerable), answers, []

    def session(self, database=None):
        return FakeSession(self)


IDS = np.arange(1000, 1100)  # row r -> question id 1000 + r


def _run(monkeypatch, ranked_by_seed, answerable, require_answer, exclude_q=frozenset(), exclude_a=frozenset(),
         seeds=("s1", "s2")):
    index = FakeIndex({})
    # embed_query -> remember which seed is being searched
    def fake_embed(title, model):
        index._key = title
        return np.ones((1, 1), dtype="float32")
    monkeypatch.setattr(cg, "embed_query", fake_embed)
    index.ranked = ranked_by_seed
    answers = {q: [q * 10] for q in answerable}
    driver = FakeDriver(answerable, answers)
    trav = [{"via_question_id": 900 + i, "via_question_title": s} for i, s in enumerate(seeds)]
    stats = {}
    cands = cg.semantic_expansion(driver, "db", trav, index, IDS, FakeModel(), 3, set(), set(exclude_q),
                                  set(exclude_a), stats=stats, require_answer=require_answer)
    return cands, stats, index, driver


def _ranked(qids_scores):
    return [(q - 1000, s) for q, s in qids_scores]


def test_selects_only_answerable_in_similarity_order_with_id_tiebreak(monkeypatch):
    ranked = {"s1": _ranked([(1001, .9), (1002, .8), (1004, .7), (1003, .7), (1005, .6), (1006, .5)]),
              "s2": _ranked([(1004, .95), (1007, .9), (1008, .8), (1009, .7)])}
    answerable = {1002, 1003, 1004, 1005, 1007, 1009}
    cands, stats, index, driver = _run(monkeypatch, ranked, answerable, True)
    # s1: 1001 (no answer) skipped; 1002; then tie 1003/1004 at .7 -> id asc 1003, 1004
    # s2: 1004 already taken by s1 -> 1007, 1009 (1008 has no answer)
    assert stats["expansion_qids"] == [1002, 1003, 1004, 1007, 1009]
    assert stats["n_expansion_requested"] == 6 and stats["n_expansion_answerable"] == 5
    assert stats["expansion_qids_empty"] == []
    assert {c["via_question_id"] for c in cands} == {1002, 1003, 1004, 1007, 1009}
    # ONE EXISTS round-trip covering every seed's candidates, then the unchanged fetch query
    exists = [q for q, _ in driver.queries if "EXISTS" in q]
    assert len(exists) == 1 and len(driver.queries) == 2
    assert driver.queries[0][1]["qids"] == sorted({1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009})


def test_search_n_is_20x_top_n_plus_excluded(monkeypatch):
    ranked = {"s1": _ranked([(1001, .9)]), "s2": _ranked([(1002, .9)])}
    _, _, index, _ = _run(monkeypatch, ranked, {1001, 1002}, True, exclude_q={1050, 1051})
    # seed i: 20*3 + |{already_seen} ∪ exclude_q ∪ {seed}| + 3*i
    assert index.calls == [("s1", 60 + 3 + 0), ("s2", 60 + 3 + 3)]


def test_excluded_answers_make_a_question_unanswerable(monkeypatch):
    ranked = {"s1": _ranked([(1001, .9), (1002, .8)])}
    # 1001's only answer (10010) is excluded -> EXISTS fails for it in the real Cypher; the fake
    # driver models that by leaving it out of `answerable`
    cands, stats, _, driver = _run(monkeypatch, ranked, {1002}, True, exclude_a={10010}, seeds=("s1",))
    assert stats["expansion_qids"] == [1002]
    assert driver.queries[0][1]["exclude_a"] == [10010]


def test_require_answer_false_is_the_previous_expansion(monkeypatch):
    ranked = {"s1": _ranked([(1001, .9), (1002, .8), (1003, .7), (1004, .6)]),
              "s2": _ranked([(1005, .9), (1006, .8), (1007, .7)])}
    cands, stats, index, driver = _run(monkeypatch, ranked, {1002}, False)
    assert stats["expansion_qids"] == [1001, 1002, 1003, 1005, 1006, 1007]  # top-3 regardless of answers
    assert stats["expansion_qids_empty"] == [1001, 1003, 1005, 1006, 1007]
    assert "n_expansion_requested" not in stats
    assert not any("EXISTS" in q for q, _ in driver.queries)
    assert index.calls[0][1] == 3 + 1 + 5  # anchor_via_vector_search's own buffer (top_n + |{seed}| + 5)


def test_fewer_than_top_n_uses_what_is_there(monkeypatch):
    ranked = {"s1": _ranked([(1001, .9), (1002, .8)])}
    _, stats, _, _ = _run(monkeypatch, ranked, {1002}, True, seeds=("s1",))
    assert stats["expansion_qids"] == [1002]
    assert (stats["n_expansion_requested"], stats["n_expansion_answerable"]) == (3, 1)


def test_config_keys_only_when_on_and_filename_matches():
    from llm.manifest import compute_config_hash

    args = ("openai", "gpt-4o-mini", 50, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4, True, True)
    off = cg.build_config(*args, c_retrieval_version="v3", alpha=0.5)
    on = cg.build_config(*args, c_retrieval_version="v3", alpha=0.5, expansion_require_answer=True)
    assert "expansion_require_answer" not in off
    assert on["expansion_require_answer"] is True and on["expansion_overfetch"] == cg.EXPANSION_OVERFETCH == 20
    path = cg.build_output_path("r", "openai", "gpt-4o-mini", 50, 42, oversample_pool=1536, c_retrieval_version="v3",
                                alpha=0.5, expansion_require_answer=True)
    assert path.name.endswith(f"_{compute_config_hash(on)}.jsonl")


def test_setting_default_true_and_env_override(monkeypatch):
    monkeypatch.delenv("EXPANSION_REQUIRE_ANSWER", raising=False)
    assert cg.expansion_require_answer_default() is True
    monkeypatch.setenv("EXPANSION_REQUIRE_ANSWER", "false")
    assert cg.expansion_require_answer_default() is False
