"""Step 5: exploratory switches (max_hops, edge_types) -- defaults
reproduce current behavior; max_hops=1 never returns hop-2 items;
edge_types filtering works for both the 1-hop fetch types and the
2-hop relation types. Uses a fake Neo4j driver/session (no real
database dependency) that records which Cypher was run and returns
canned rows, so these are pure unit tests of the query-construction
logic, not integration tests against the real graph.
"""

from c_graphrag import DEFAULT_EDGE_TYPES, semantic_expansion, traverse_graph


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, queries_run, rows_by_call):
        self.queries_run = queries_run
        self.rows_by_call = rows_by_call
        self._call_index = 0

    def run(self, query, **kwargs):
        self.queries_run.append(query)
        rows = self.rows_by_call[self._call_index] if self._call_index < len(self.rows_by_call) else []
        self._call_index += 1
        return _FakeResult(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeDriver:
    def __init__(self, rows_by_call):
        self.queries_run = []
        self.rows_by_call = rows_by_call

    def session(self, database=None):
        return _FakeSession(self.queries_run, self.rows_by_call)


def _row(answer_id=1, via_question_id=10, hop=1):
    return {"via_question_id": via_question_id, "via_question_title": "t", "answer_id": answer_id,
            "answer_body": "b", "answer_trust_score": 0.1, "is_accepted": False,
            "edge_weight": 0.5, "hop": hop, "rel_type": "HAS_ANSWER"}


def test_default_max_hops_runs_both_queries():
    driver = _FakeDriver(rows_by_call=[[_row(1, hop=1)], [_row(2, hop=2)]])
    candidates = traverse_graph(driver, "neo4j", [10], set(), set())
    assert len(driver.queries_run) == 2  # 1-hop AND 2-hop query both ran
    assert len(candidates) == 2


def test_max_hops_1_never_runs_2hop_query_or_returns_hop2_items():
    driver = _FakeDriver(rows_by_call=[[_row(1, hop=1)]])
    candidates = traverse_graph(driver, "neo4j", [10], set(), set(), max_hops=1)
    assert len(driver.queries_run) == 1  # only the 1-hop query ran
    assert all(c["hop"] == 1 for c in candidates)


def test_edge_types_restricting_to_one_fetch_type_excludes_2hop_too():
    """edge_types=("HAS_ACCEPTED_ANSWER",) leaves no 2-hop relation types
    -- the 2-hop query must not run at all."""
    driver = _FakeDriver(rows_by_call=[[_row(1, hop=1)]])
    candidates = traverse_graph(driver, "neo4j", [10], set(), set(),
                                 edge_types=("HAS_ACCEPTED_ANSWER",))
    assert len(driver.queries_run) == 1
    assert "HAS_ACCEPTED_ANSWER" in driver.queries_run[0]
    assert "HAS_ANSWER]" not in driver.queries_run[0].replace("HAS_ACCEPTED_ANSWER", "")


def test_edge_types_restricting_2hop_relation_type_only():
    driver = _FakeDriver(rows_by_call=[[_row(1, hop=1)], [_row(2, hop=2)]])
    candidates = traverse_graph(
        driver, "neo4j", [10], set(), set(),
        edge_types=("HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "TAG_COOCCUR"),
    )
    assert len(driver.queries_run) == 2
    second_query = driver.queries_run[1]
    assert "TAG_COOCCUR" in second_query
    assert "IS_RELATED_TO" not in second_query
    assert "EMBED_SIM" not in second_query


def test_no_fetch_types_returns_nothing_no_query_run():
    driver = _FakeDriver(rows_by_call=[])
    candidates = traverse_graph(driver, "neo4j", [10], set(), set(), edge_types=("TAG_COOCCUR",))
    assert candidates == []
    assert driver.queries_run == []


def test_default_edge_types_constant_covers_every_type_used():
    assert set(DEFAULT_EDGE_TYPES) == {
        "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR", "EMBED_SIM",
    }


def test_semantic_expansion_default_edge_types_runs_query():
    class _FakeIndex:
        def search(self, q, n):
            import numpy as np
            return np.array([[0.9]]), np.array([[0]])

    driver = _FakeDriver(rows_by_call=[[_row(5, via_question_id=20)]])
    traversal_candidates = [{"via_question_id": 10, "via_question_title": "seed"}]

    class _FakeEmbedModel:
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np
            return np.array([[1.0]])

    import numpy as np
    candidates = semantic_expansion(
        driver, "neo4j", traversal_candidates, _FakeIndex(), np.array([20]),
        _FakeEmbedModel(), n_expansion=1, already_seen_qids={10}, exclude_question_ids=set(),
        exclude_answer_ids=set(),
    )
    assert len(candidates) == 1
    assert len(driver.queries_run) == 1


def test_semantic_expansion_no_fetch_types_returns_empty():
    import numpy as np

    driver = _FakeDriver(rows_by_call=[])
    candidates = semantic_expansion(
        driver, "neo4j", [{"via_question_id": 10, "via_question_title": "x"}],
        None, np.array([]), None, n_expansion=1, already_seen_qids=set(),
        exclude_question_ids=set(), exclude_answer_ids=set(), edge_types=("TAG_COOCCUR",),
    )
    assert candidates == []
    assert driver.queries_run == []


def test_fetch_author_trust_batched_lookup_missing_answers_absent():
    from c_graphrag import fetch_author_trust

    class _Row(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    driver = _FakeDriver(rows_by_call=[[_Row(answer_id=1, weight=0.7)]])
    weights = fetch_author_trust(driver, "neo4j", [1, 2])
    assert weights == {1: 0.7}  # answer_id=2 has no AUTHOR_TRUST edge -- simply absent
    assert 2 not in weights


def test_fetch_author_trust_empty_list_no_query():
    from c_graphrag import fetch_author_trust

    driver = _FakeDriver(rows_by_call=[])
    assert fetch_author_trust(driver, "neo4j", []) == {}
    assert driver.queries_run == []
