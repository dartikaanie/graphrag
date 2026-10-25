"""Condition D retrieval v5 (Amendment 17): answerable anchors + the high level fills up to k. Offline (fake driver of
test_d_retrieval_v4)."""

import pytest

import d_lightrag as dl
import test_d_retrieval_v4 as t4
from llm.common.leakage import assert_no_leakage


def test_fill_takes_three_then_adds_until_k_distinct_answers():
    ranked = [30, 31, 32, 33, 34, 35]
    one = {q: [q * 10] for q in ranked}
    assert dl.select_high_level_fill(ranked, one, set(), 3, 5) == [30, 31, 32, 33, 34]   # 3, then 2 more for 5 answers
    assert dl.select_high_level_fill(ranked, one, {1, 2}, 3, 5) == [30, 31, 32]            # low-level answers count
    assert dl.select_high_level_fill(ranked, one, set(), 3, 2) == [30, 31, 32]             # never fewer than n_high
    assert dl.select_high_level_fill(ranked[:2], one, set(), 3, 5) == [30, 31]            # candidates run out
    two = {**one, 31: [310, 311]}
    assert dl.select_high_level_fill(ranked, two, set(), 3, 5) == [30, 31, 32, 33]         # a 2-answer question counts twice
    assert dl.select_high_level_fill([], {}, set(), 3, 5) == []


def _v5(anchors=t4.ANCHORS, low=frozenset(), cap=dl.HIGH_LEVEL_CANDIDATE_CAP, stats=None, driver=None):
    driver = driver or t4.FakeDriver()
    out = dl.retrieve_high_level_v5(driver, "g", anchors, set(t4.EXCL.question_ids), set(t4.EXCL.answer_ids), 3,
                                    t4.QUERY, t4.MATRIX, t4.ID_TO_ROW, 5, set(low), cap=cap, stats=stats)
    return out, driver


def test_v5_fills_up_to_five_answers_when_enough_candidates_exist():
    stats = {}
    high, _ = _v5(stats=stats)
    answers = {c["answer_id"] for c in high}
    assert len(answers) >= 5 and stats["high_level_extended"] is True and stats["n_high_level_taken"] > 3
    fused = dl.fuse_dual_level([], high, 5, 400)
    assert len(fused) == 5                                  # the top 5 answers by score are kept
    scores = [i["relevance_score"] for i in fused]
    assert scores == sorted(scores, reverse=True)


def test_v5_stops_as_soon_as_k_is_reached_and_keeps_n_high_when_enough():
    stats = {}
    _v5(low={1, 2}, stats=stats)      # two low-level answers already: the three best neighbours bring the total to 5
    assert stats["n_high_level_taken"] == 3 and stats["high_level_extended"] is False


def test_v5_same_score_as_the_low_level_and_no_edge_weights():
    high, driver = _v5()
    scores = dl.question_scores(t4.QUERY, t4.MATRIX, t4.ID_TO_ROW, {c["via_question_id"] for c in high})
    assert all(c["relevance_score"] == scores[c["via_question_id"]] for c in high)
    assert not any("weight" in q.lower() for q in driver.queries)


def test_v5_deterministic_and_leak_free():
    a, _ = _v5()
    b, _ = _v5()
    assert [(c["answer_id"], c["relevance_score"]) for c in a] == [(c["answer_id"], c["relevance_score"]) for c in b]
    fused = dl.fuse_dual_level([], a, 5, 400)
    assert_no_leakage(fused, t4.EXCL)
    assert not {1, 9001, 9002} & ({c["via_question_id"] for c in a} | {c["answer_id"] for c in a})


def test_v5_cap_still_applies_and_is_reported():
    stats = {}
    high, _ = _v5(cap=3, stats=stats)
    assert stats["cap_hit"] is True and stats["n_neighbours"] == 6 and {c["via_question_id"] for c in high} <= {30, 31, 32}


def test_v5_has_no_trust_or_expansion_fields():
    high, _ = _v5()
    fused = dl.fuse_dual_level([], high, 5, 400)
    for item in fused:
        assert item["trust_weight"] is None and item["source_stage"] == "high_level"
        assert not {"combined_score", "final_score", "trust", "sim_norm", "alpha"} & set(item)


def test_versions_hash_and_pin():
    args = ("openai", "gpt-4o-mini", 384, 42, 1536, 5, 3, 3, True)
    from llm.manifest import compute_config_hash

    hashes = {v: compute_config_hash(dl.build_config(*args, d_retrieval_version=v)) for v in ("v3", "v4", "v5")}
    assert len(set(hashes.values())) == 3 and dl.D_RETRIEVAL_VERSION == "v5"
    from llm import official_run_matrix as orm

    assert orm.D_RETRIEVAL_VERSION == "v5"
    with pytest.raises(ValueError):
        dl.process_sample(None, None, None, None, None, "g", None, None, None, {}, {}, 5, 3, 3, 400, "m", None,
                          d_retrieval_version="v9")
