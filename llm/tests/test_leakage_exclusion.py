"""Expanded per-query leakage exclusion (llm/common/leakage.py,
llm/leakage/build_exclusion_sets.py) -- offline, synthetic ids."""

import hashlib

import pytest

from llm.common import leakage as lk
from llm.leakage import build_exclusion_sets as bes

NEIGHBORS = {1: [(2, "Duplicate"), (3, "Linked")], 2: [(1, "Duplicate")], 4: [(5, "Linked")]}
NEAR = {1: [(6, 0.97), (7, 0.94)], 4: [(8, 0.951)]}
ANSWERS = {1: [10, 11], 2: [20], 3: [30], 5: [50], 6: [60], 7: [70], 8: [80]}
THREAD = {1: [10, 11, 12], 4: []}


def _rows(modes=bes.DEFAULT_MODES):
    return bes.compute_rows([4, 1], modes, THREAD, NEIGHBORS, NEAR, ANSWERS)


def _row(rows, mode, q):
    return next(r for r in rows if r["mode"] == mode and r["question_id"] == q)


def test_modes_and_settings_strings():
    assert lk.leakage_mode() == "dup+neardup0.95"  # the function's own defaults: the Amendment 8 mode
    assert lk.leakage_mode(True, False, None, exclude_dup=False) == "thread"
    assert lk.leakage_mode(True, True, 0.95) == "dup+linked+neardup0.95"
    assert lk.leakage_mode(True, False, None) == "dup"
    assert lk.leakage_mode(False, True, 0.9) == "off"
    for m in bes.DEFAULT_MODES[1:]:
        spec = bes.parse_mode(m)
        assert lk.leakage_mode(True, spec["linked"], spec["neardup"]) == m
    with pytest.raises(ValueError):
        bes.parse_mode("linked")


def test_settings_from_env(monkeypatch):
    for k in ("LEAKAGE_FILTER", "LEAKAGE_EXCLUDE_DUP", "LEAKAGE_EXCLUDE_LINKED", "LEAKAGE_NEAR_DUP_COSINE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(lk.LeakageSettings, "model_config", {**lk.LeakageSettings.model_config, "env_file": None})
    assert lk.LeakageSettings().mode == "thread" == lk.default_mode()  # Amendment 15: the settings default
    monkeypatch.setenv("LEAKAGE_EXCLUDE_DUP", "true")
    assert lk.LeakageSettings().mode == "dup"
    monkeypatch.setenv("LEAKAGE_NEAR_DUP_COSINE", "0.95")
    assert lk.LeakageSettings().mode == "dup+neardup0.95"
    monkeypatch.delenv("LEAKAGE_EXCLUDE_DUP")
    monkeypatch.delenv("LEAKAGE_NEAR_DUP_COSINE")
    monkeypatch.setenv("LEAKAGE_FILTER", "true")
    monkeypatch.setenv("LEAKAGE_EXCLUDE_LINKED", "true")
    monkeypatch.setenv("LEAKAGE_NEAR_DUP_COSINE", "none")
    assert lk.LeakageSettings().mode == "dup+linked"
    monkeypatch.setenv("LEAKAGE_FILTER", "false")
    assert lk.LeakageSettings().mode == "off"


def test_config_key_absent_only_for_none():
    """Amendment 15: None (a run before Amendment 8) has no key; "thread" is written."""
    assert lk.config_leakage_key(None) == {}
    assert lk.config_leakage_key(lk.MODE_THREAD) == {"leakage": "thread"}
    assert lk.config_leakage_key("dup+neardup0.95") == {"leakage": "dup+neardup0.95"}


def test_thread_exclusion_set_is_only_q_and_its_answers_even_with_a_table_full_of_neighbours(tmp_path):
    """Duplicate (1<->2), near-dup (1~6) and Linked (1-3) neighbours exist in the table for q=1,
    yet the thread context ignores them: exclusion = {q} / thread answers + accepted answer."""
    path = tmp_path / "ex.parquet"
    bes.write_rows(_rows(), path)
    assert _row(_rows(), "dup+linked+neardup0.95", 1)["excluded_q_ids"] == [1, 2, 3, 6]  # the table does have them
    ex = lk.LeakageContext.from_mode(lk.MODE_THREAD, path).exclusion_for(1, [10, 11], 12)
    assert ex.question_ids == {1} and ex.answer_ids == {10, 11, 12}
    for neighbour_q, neighbour_a in ((2, 20), (3, 30), (6, 60)):  # Duplicate, Linked, near-dup stay retrievable
        assert not ex.excludes(neighbour_q, neighbour_a)
    assert ex.excludes(1, 999) and ex.excludes(999, 12)


def test_assert_no_leakage_still_fails_on_q_or_its_answers_in_thread_mode():
    ex = lk.LEGACY_CONTEXT.exclusion_for(1, [10, 11], 12)
    lk.assert_no_leakage([{"question_id": 2, "answer_id": 20}, {"question_id": 3, "answer_id": 30}], ex)  # neighbours: ok
    with pytest.raises(lk.LeakageError):
        lk.assert_no_leakage([{"question_id": 1, "answer_id": 999}], ex)   # q itself
    with pytest.raises(lk.LeakageError):
        lk.assert_no_leakage([{"question_id": 77, "answer_id": 11}], ex)   # one of q's answers
    with pytest.raises(lk.LeakageError):
        lk.assert_no_leakage([{"question_id": 77, "answer_id": 12}], ex)   # q's accepted answer


def test_thread_mode_is_q_and_its_answers_only():
    r = _row(_rows(), lk.MODE_THREAD, 1)
    assert r["excluded_q_ids"] == [1] and r["excluded_a_ids"] == [10, 11, 12]
    assert (r["n_dup"], r["n_linked"], r["n_neardup"]) == (0, 0, 0)


def test_duplicate_both_directions_with_answers_linked_off_by_default():
    rows = _rows()
    assert _row(rows, "dup", 1)["excluded_q_ids"] == [1, 2]
    assert _row(rows, "dup", 1)["excluded_a_ids"] == [10, 11, 12, 20]
    # neighbours are fetched undirected: 2 -> 1 also listed for q=2 (not in this question set)
    assert _row(rows, "dup", 4)["excluded_q_ids"] == [4]  # Linked only -> not excluded
    d = _row(rows, "dup+neardup0.95", 1)
    assert d["excluded_q_ids"] == [1, 2, 6] and 60 in d["excluded_a_ids"] and 70 not in d["excluded_a_ids"]
    assert (d["n_dup"], d["n_linked"], d["n_neardup"]) == (1, 0, 1)
    l4 = _row(rows, "dup+linked+neardup0.95", 4)
    assert l4["excluded_q_ids"] == [4, 5, 8] and l4["excluded_a_ids"] == [50, 80]


def test_rows_are_sorted_and_byte_deterministic(tmp_path):
    rows = _rows()
    assert rows == sorted(rows, key=lambda r: (r["mode"], r["question_id"]))
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    bes.write_rows(rows, a)
    bes.write_rows(bes.compute_rows([1, 4], bes.DEFAULT_MODES, THREAD, NEIGHBORS, NEAR, ANSWERS), b)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_context_reads_table_and_builds_exclusion(tmp_path):
    path = tmp_path / "ex.parquet"
    bes.write_rows(_rows(), path)
    ctx = lk.LeakageContext.from_mode("dup+neardup0.95", path)
    ex = ctx.exclusion_for(1, thread_answer_ids=[99], accepted_answer_id=12)
    assert ex.question_ids == {1, 2, 6} and {10, 11, 12, 20, 60, 99} <= ex.answer_ids
    assert ex.excludes(2, 1) and ex.excludes(123, 60) and not ex.excludes(3, 30)
    with pytest.raises(lk.LeakageError, match="no precomputed exclusion set"):
        ctx.exclusion_for(777)
    assert lk.LeakageContext.from_mode(lk.MODE_OFF).exclusion_for(1, [10], 12) == lk.NO_EXCLUSION
    assert lk.LEGACY_CONTEXT.exclusion_for(1, [10], 12) == lk.build_exclusion(1, [10], 12)
    with pytest.raises(lk.LeakageError, match="not in"):
        lk.ExclusionTable.load("dup+linked", path)


def test_query_log_fields():
    ex = lk.build_exclusion(1, [10, 11], 12, [2], [20])
    assert lk.LEGACY_CONTEXT.query_log(ex, 3, 5) == {"mode": "thread", "n_excluded_q": 2, "n_excluded_a": 4,
                                                     "n_hits_filtered": 3, "final_k": 5}


def test_build_exclusion_never_uses_answer_content():
    import inspect

    params = inspect.signature(lk.build_exclusion).parameters
    assert set(params) == {"question_id", "thread_answer_ids", "accepted_answer_id",
                           "related_question_ids", "related_answer_ids"}
