"""Step 5: selection helper on fixtures for every rule path (clear
winner, primary tie -> tie-breaker -> larger alpha, stage-1
keep-current/skip-stage-2, pooled stage-2 with a count-based tie
margin), applying docs/DECISION_C_SCORING.md's rules exactly."""

import pytest

from ctxrel_sweep import jaccard_overlap, select_stage1_candidates, select_stage2_winner


def _stage1(pcts: dict[float, float], scores: dict[float, float] | None = None) -> dict:
    scores = scores or {}
    return {
        alpha: {"pct_questions_with_relevant": pct, "mean_relevance_score": scores.get(alpha, 0.0)}
        for alpha, pct in pcts.items()
    }


def test_stage1_clear_winner_no_tie():
    report = _stage1({0.0: 40.0, 0.25: 60.0, 0.5: 90.0, 0.75: 70.0, 1.0: 30.0})
    result = select_stage1_candidates(report)
    assert result["keep_current"] is False
    assert result["top_two"] == [0.5, 0.75]


def test_stage1_primary_tie_resolved_by_tiebreaker_mean_relevance_score():
    # 0.5 and 0.75 tie on the primary criterion (89 vs 90, within 2pp)
    # but 0.75 has a clearly higher mean relevance score -- the
    # tie-breaker must put 0.75 first, NOT the larger-alpha fallback
    # (which would also pick 0.75 here, so use scores where larger-alpha
    # and the tie-breaker disagree to prove the tie-breaker is actually
    # applied).
    report = _stage1(
        {0.0: 40.0, 0.25: 60.0, 0.5: 90.0, 0.75: 89.0, 1.0: 30.0},
        {0.5: 0.9, 0.75: 0.3},
    )
    result = select_stage1_candidates(report)
    assert result["top_two"] == [0.5, 0.75]


def test_stage1_primary_and_tiebreaker_tie_resolved_by_larger_alpha():
    # 0.5 and 0.75 tie on primary (within 2pp) AND on mean_relevance_score
    # (within 0.02) -- only then does larger alpha decide.
    report = _stage1(
        {0.0: 40.0, 0.25: 60.0, 0.5: 90.0, 0.75: 89.0, 1.0: 30.0},
        {0.5: 0.60, 0.75: 0.61},
    )
    result = select_stage1_candidates(report)
    assert result["top_two"] == [0.75, 0.5]


def test_stage1_tiebreaker_margin_boundary_just_over_is_not_tied():
    # 0.03 difference in mean_relevance_score is just OUTSIDE the 0.02
    # tie margin -- the higher score must win outright, not fall through
    # to the larger-alpha tiebreak.
    report = _stage1(
        {0.5: 90.0, 0.75: 89.0},
        {0.5: 0.80, 0.75: 0.77},
    )
    result = select_stage1_candidates(report)
    assert result["ranked"][0][0] == 0.5


def test_stage1_keep_current_when_alpha_1_ranks_first_outright():
    report = _stage1({0.0: 10.0, 0.25: 20.0, 0.5: 30.0, 0.75: 40.0, 1.0: 90.0})
    result = select_stage1_candidates(report)
    assert result["keep_current"] is True
    assert result["top_two"] == []


def test_stage1_keep_current_when_all_tied_with_alpha_1_on_both_primary_and_tiebreaker():
    # Every alpha is within 2pp of alpha=1's primary score AND within
    # 0.02 of alpha=1's tie-breaker score -- a genuine global tie, so
    # keep-current applies even though alpha=1 doesn't rank strictly
    # first (0.5 edges it out by 0.4pp, well inside the tie margin).
    report = _stage1(
        {0.0: 50.0, 0.25: 50.0, 0.5: 50.4, 0.75: 50.0, 1.0: 50.0},
        {0.0: 0.60, 0.25: 0.60, 0.5: 0.61, 0.75: 0.60, 1.0: 0.60},
    )
    result = select_stage1_candidates(report)
    assert result["keep_current"] is True
    assert result["top_two"] == []


def test_stage1_not_keep_current_when_alpha_1_ties_on_primary_but_not_tiebreaker():
    # alpha=1 ties with the best on the PRIMARY criterion alone, but its
    # tie-breaker score is far worse and it does NOT rank first overall
    # -- under the new rule this must NOT trigger keep-current (unlike
    # the earlier, looser rule that only checked the primary criterion).
    report = _stage1(
        {0.0: 50.0, 0.25: 50.0, 0.5: 50.0, 0.75: 50.0, 1.0: 49.0},
        {0.0: 0.9, 0.25: 0.9, 0.5: 0.9, 0.75: 0.9, 1.0: 0.1},
    )
    result = select_stage1_candidates(report)
    assert result["keep_current"] is False
    assert result["top_two"] == [0.75, 0.5]


def test_stage1_not_keep_current_when_alpha_1_clearly_loses():
    report = _stage1({0.0: 90.0, 0.25: 85.0, 0.5: 80.0, 0.75: 75.0, 1.0: 10.0})
    result = select_stage1_candidates(report)
    assert result["keep_current"] is False


def test_stage1_all_alpha_at_zero_percent_relevant_and_tied_score_triggers_keep_current():
    """Degenerate case: if EVERY alpha ties on the primary criterion
    (e.g. all at 0% RELEVANT) AND their tie-breaker scores also agree,
    the whole field is tied with alpha=1 on both measures, so
    keep-current applies -- there is no retrieval-side relevance signal
    at all to prefer a different alpha on."""
    report = _stage1({0.0: 0.0, 0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 1.0: 0.0})
    result = select_stage1_candidates(report)
    assert result["keep_current"] is True
    assert result["top_two"] == []


def test_stage1_all_alpha_at_zero_percent_relevant_but_tiebreaker_disagrees_does_not_keep_current():
    """All alpha tie at 0% RELEVANT (primary), but their tie-breaker
    scores differ enough that it's NOT a global tie, and alpha=1 is not
    the tie-breaker winner -- keep-current must NOT apply; the top two
    by tie-breaker score advance normally."""
    report = _stage1(
        {0.0: 0.0, 0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 1.0: 0.0},
        {0.0: 0.5, 0.25: 0.4, 0.5: 0.2, 0.75: 0.1, 1.0: 0.0},
    )
    result = select_stage1_candidates(report)
    assert result["keep_current"] is False
    assert result["top_two"] == [0.0, 0.25]


def test_stage1_empty_report_raises():
    with pytest.raises(ValueError):
        select_stage1_candidates({})


def _stage2(counts: dict[float, tuple[int, int, int, int]]) -> dict:
    """counts[alpha] = (hall_count_excl, n_excl, hall_count_incl, n_incl)."""
    return {
        alpha: {
            "hallucination_count_excl_abstain": c_excl, "n_excl_abstain": n_excl,
            "hallucination_count_incl_abstain": c_incl, "n_incl_abstain": n_incl,
        }
        for alpha, (c_excl, n_excl, c_incl, n_incl) in counts.items()
    }


def test_stage2_clear_winner_lowest_hallucination_rate():
    # 0.75: 10/100 (10%) vs 0.5: 30/100 (30%) -- far more than the
    # 2-answer tie margin apart.
    metrics = _stage2({0.5: (30, 100, 40, 100), 0.75: (10, 100, 20, 100)})
    result = select_stage2_winner(metrics)
    assert result["winner"] == 0.75
    assert result["keep_current"] is False


def test_stage2_tie_within_2_answers_resolved_by_larger_alpha():
    # 20 vs 21 hallucinations out of 100 -- within the 2-answer tie
    # margin, so larger alpha wins despite 0.75 having a (negligibly)
    # higher raw count.
    metrics = _stage2({0.5: (20, 100, 30, 100), 0.75: (21, 100, 30, 100)})
    result = select_stage2_winner(metrics)
    assert result["winner"] == 0.75


def test_stage2_tie_excl_abstain_resolved_by_incl_abstain_count():
    # Tied (within margin) on excl.-ABSTAIN count -- the incl.-ABSTAIN
    # count then decides, even though it also needs its own tie-margin
    # check (35 vs 30 is outside the margin, so 0.75 wins outright here).
    metrics = _stage2({0.5: (20, 100, 35, 100), 0.75: (21, 100, 30, 100)})
    result = select_stage2_winner(metrics)
    assert result["winner"] == 0.75


def test_stage2_keep_current_when_winner_is_alpha_1():
    metrics = _stage2({1.0: (10, 100, 15, 100), 0.75: (20, 100, 25, 100)})
    result = select_stage2_winner(metrics)
    assert result["winner"] == 1.0
    assert result["keep_current"] is True


def test_stage2_empty_raises():
    with pytest.raises(ValueError):
        select_stage2_winner({})


def test_jaccard_overlap_identical_sets():
    assert jaccard_overlap({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_overlap_disjoint_sets():
    assert jaccard_overlap({1, 2}, {3, 4}) == 0.0


def test_jaccard_overlap_partial():
    assert jaccard_overlap({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)


def test_jaccard_overlap_both_empty_is_one():
    assert jaccard_overlap(set(), set()) == 1.0


def test_build_stage1_report_raises_on_missing_metrics():
    from ctxrel_sweep import build_stage1_report

    with pytest.raises(ValueError):
        build_stage1_report({0.5: {"pct_questions_with_relevant": 50.0}})


def test_build_stage1_report_passes_through_complete_metrics():
    from ctxrel_sweep import build_stage1_report

    metrics = {
        "pct_questions_with_relevant": 50.0, "pct_relevant": 10.0, "pct_partial": 20.0,
        "pct_irrelevant": 70.0, "mean_relevance_score": 0.3, "mean_sim_selected": 0.4,
        "mean_trust_selected": 0.5, "pct_accepted_selected": 60.0,
        "jaccard_vs_alpha1": 0.8, "jaccard_vs_alpha0": 0.2,
    }
    report = build_stage1_report({0.5: metrics})
    assert report[0.5] == metrics
