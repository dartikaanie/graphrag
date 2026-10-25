"""Answerable entity anchoring (llm/common/anchors.py, Amendment 17), shared by C v4 and D v5. Offline: a fake FAISS
list and a fake Neo4j driver that honours the exclusion parameter like the real query."""

import numpy as np

from llm.common import anchors as an

# FAISS list, best first: 1 is the evaluation question itself
LIST = [1, 10, 11, 12, 13, 14, 15, 16]
ANSWERS = {1: [9001, 9002], 10: [], 11: [], 12: [1201], 13: [1301, 1302], 14: [1401], 15: [1501], 16: [1601]}


class FakeIndex:
    def __init__(self, ids, scores=None):
        self.ids = ids
        self.scores = scores or [0.9 - 0.01 * i for i in range(len(ids))]

    def search(self, q, n):
        ids = self.ids[:n] + [-1] * max(0, n - len(self.ids))
        sc = self.scores[:n] + [0.0] * max(0, n - len(self.scores))
        return np.array([sc]), np.array([list(range(len(self.ids)))[:n] + [-1] * max(0, n - len(self.ids))])


class FakeDriver:
    def __init__(self, answers=ANSWERS):
        self.answers, self.calls = answers, 0

    def session(self, database=None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, query, qids, exclude_a):
        self.calls += 1
        ok = [q for q in qids if any(a not in set(exclude_a) for a in self.answers.get(q, []))]
        return type("R", (), {"data": lambda self_: [{"qid": q} for q in ok]})()


IDS_ARR = np.arange(0, 100)  # row i -> question id i (rows map to ids through ids_arr)


def _select(ids=LIST, exclude_q=(1,), exclude_a=(9001, 9002), n=3, depth=an.ANCHOR_SCAN_DEPTH, answers=ANSWERS, stats=None):
    idx = FakeIndex(ids)
    return an.select_answerable_anchors(np.zeros((1, 4), dtype="float32"), idx, np.array(ids + [0]), n, set(exclude_q),
                                        set(exclude_a), FakeDriver(answers), "db", depth=depth, stats=stats)


def test_unanswered_questions_are_skipped_and_the_scan_continues_down_the_same_list():
    stats = {}
    assert [q for q, _ in _select(stats=stats)] == [12, 13, 14]
    assert (stats["anchors_found"], stats["anchors_requested"]) == (3, 3)
    assert stats["anchors_skipped_unanswered"] == 2 and stats["anchor_scan_depth"] == 400


def test_no_unanswered_seed_ever():
    for n in (1, 2, 3, 5):
        for q, _ in _select(n=n):
            assert ANSWERS[q]


def test_evaluation_question_and_its_answers_are_excluded():
    ids = [1] + LIST[1:]
    out = [q for q, _ in _select(ids=ids)]
    assert 1 not in out  # the evaluation question has answers, but it is excluded
    # a question whose ONLY answers are in the exclusion set is not answerable
    answers = {**ANSWERS, 12: [9001]}
    assert 12 not in [q for q, _ in _select(answers=answers)]


def test_fewer_than_n_seeds_uses_what_was_found_and_records_it():
    answers = {q: [] for q in LIST} | {13: [1301], 14: [1401]}
    stats = {}
    out = _select(answers=answers, stats=stats)
    assert [q for q, _ in out] == [13, 14] and stats["anchors_found"] == 2 and stats["anchors_requested"] == 3
    assert stats["anchor_candidates_scanned"] == stats["anchor_candidates_scanned"]  # recorded
    none = {}
    assert _select(answers={q: [] for q in LIST}, stats=none) == [] and none["anchors_found"] == 0


def test_scan_depth_limits_the_search():
    assert [q for q, _ in _select(depth=3)] == [12]  # candidates (without the excluded 1): 10, 11, 12 only
    assert [q for q, _ in _select(depth=400)] == [12, 13, 14]


def test_deterministic_and_rank_ordered_with_cosines():
    a, b = _select(), _select()
    assert a == b and [q for q, _ in a] == [12, 13, 14]
    assert [s for _, s in a] == sorted((s for _, s in a), reverse=True)  # FAISS order, best first


def test_one_batched_existence_check_per_question():
    d = FakeDriver()
    idx = FakeIndex(LIST)
    an.select_answerable_anchors(np.zeros((1, 4), dtype="float32"), idx, np.array(LIST + [0]), 3, {1}, {9001, 9002}, d, "db")
    assert d.calls == 1


def test_c_and_d_use_the_one_shared_function():
    import c_graphrag as cg
    from llm.d_lightrag import d_lightrag as dl

    assert cg.select_answerable_anchors is dl.select_answerable_anchors is an.select_answerable_anchors
    assert an.ANCHOR_SCAN_DEPTH == 400
    import inspect
    assert "select_answerable_anchors(" in inspect.getsource(cg.process_sample)
    assert "select_answerable_anchors(" in inspect.getsource(dl.process_sample)
