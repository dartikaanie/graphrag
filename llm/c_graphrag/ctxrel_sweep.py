"""
ctxrel_sweep.py
=====================================
Pure, testable logic for the α sweep (Step 4 of docs/agent_prompt_
c_retrieval_v3_devset.md): stage-1 report math, the selection helper
(applying docs/DECISION_C_SCORING.md's rules EXACTLY, no extra
heuristics), and stage-2 report math. Deliberately separated from any
I/O (reading ctxrel-v1 output files, calling the judge, launching
retrieval) so the DECISION LOGIC itself can be unit-tested against
fixtures without any real run -- see llm/c_graphrag/tests/
test_selection_helper.py.

Jaccard overlap and the stage-1/stage-2 report builders operate on
plain Python dicts/lists the caller assembles from real ctxrel-v1/
judge-v2 output records -- this module never reads a file itself.
"""

from typing import Callable


def jaccard_overlap(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def build_stage1_report(per_alpha_metrics: dict[float, dict]) -> dict[float, dict]:
    """`per_alpha_metrics[alpha]` is expected to already contain:
    pct_questions_with_relevant, pct_relevant, pct_partial, pct_irrelevant,
    mean_relevance_score, mean_sim_selected, mean_trust_selected,
    pct_accepted_selected, jaccard_vs_alpha1, jaccard_vs_alpha0 (the
    caller computes these from real ctxrel-v1 + retrieval records --
    this function is a pass-through/validation point, kept as its own
    function so the report's EXACT key set is documented and testable
    without needing real data)."""
    required = {
        "pct_questions_with_relevant", "pct_relevant", "pct_partial", "pct_irrelevant",
        "mean_relevance_score", "mean_sim_selected", "mean_trust_selected",
        "pct_accepted_selected", "jaccard_vs_alpha1", "jaccard_vs_alpha0",
    }
    report = {}
    for alpha, metrics in per_alpha_metrics.items():
        missing = required - set(metrics)
        if missing:
            raise ValueError(f"alpha={alpha}: missing stage-1 metrics {missing}")
        report[alpha] = dict(metrics)
    return report


# ---------------------------------------------------------------------
# Selection helper -- implements docs/DECISION_C_SCORING.md's Selection
# procedure EXACTLY (primary criterion -> tie-breaker -> larger-alpha
# tiebreak; stage-1 keep-current/skip-stage-2 rule; pooled stage-2
# criteria with a count-based tie margin). No extra heuristics beyond
# what's written there.
# ---------------------------------------------------------------------

TIE_THRESHOLD_PERCENTAGE_POINTS = 2.0  # "differences of <=1 question (2 percentage points) are ties"
RELEVANCE_SCORE_TIE_THRESHOLD = 0.02  # stage-1 tie-breaker: mean relevance score, tie <=0.02
STAGE2_ANSWER_COUNT_TIE_MARGIN = 2  # stage-2: hallucination counts tied if they differ by <=2 answers


def _tiered_rank(pool: dict[float, dict], tiers: list[tuple[Callable[[dict], float], float]]) -> list[float]:
    """Lexicographic ranking with a tolerance ("tie margin") at each
    tier: the best-scoring tier-0 cluster (everyone within `margin` of
    the single best tier-0 score) is ranked ahead of the rest, and
    WITHIN that cluster the same process repeats using the remaining
    tiers; once tiers are exhausted (or a cluster has only one member),
    ties are broken by alpha descending. Each `score_fn` must return a
    value where HIGHER is better (callers negate counts/rates where
    lower is better).

    This is not a plain `sorted(key=...)` call: a plain sort would
    tie-break EVERY pair by the next criterion regardless of whether
    their current-tier scores are actually close, silently overriding
    a real, non-tied difference at an earlier tier.
    """
    if not tiers:
        return sorted(pool.keys(), reverse=True)

    score_fn, margin = tiers[0]
    remaining = dict(pool)
    ranked: list[float] = []
    while remaining:
        best = max(score_fn(m) for m in remaining.values())
        cluster = {a: m for a, m in remaining.items() if best - score_fn(m) <= margin}
        ranked.extend(_tiered_rank(cluster, tiers[1:]))
        for a in cluster:
            del remaining[a]
    return ranked


def select_stage1_candidates(stage1_report: dict[float, dict]) -> dict:
    """Ranking order: primary criterion (pct_questions_with_relevant,
    descending) -> tie-breaker (mean_relevance_score, descending, only
    among primary ties) -> larger alpha (only among ties on both).

    Keep-current rule: skip stage 2 and keep the current score (C
    retrieval v2) if EITHER (a) alpha=1 ranks first in the full ranking
    order above, OR (b) every alpha is tied WITH alpha=1 on BOTH the
    primary criterion and the tie-breaker (a global tie, not merely a
    tie on the primary criterion alone -- e.g. if every alpha scores 0%
    on the primary criterion but their tie-breaker scores differ and
    alpha=1 does not have the best one, this does NOT trigger
    keep-current; alpha=1 must actually win outright or the whole field
    must agree with it on both measures).

    Returns {"ranked": [(alpha, pct), ...] sorted by the full ranking
             order above, "top_two": [alpha_a, alpha_b] for stage 2
             (empty if keep-current), "keep_current": bool,
             "reasoning": str}.
    """
    if not stage1_report:
        raise ValueError("stage1_report must not be empty")

    tiers = [
        (lambda m: m["pct_questions_with_relevant"], TIE_THRESHOLD_PERCENTAGE_POINTS),
        (lambda m: m["mean_relevance_score"], RELEVANCE_SCORE_TIE_THRESHOLD),
    ]
    ranked_alphas = _tiered_rank(stage1_report, tiers)
    ranked = [(a, stage1_report[a]["pct_questions_with_relevant"]) for a in ranked_alphas]

    if ranked[0][0] == 1.0:
        return {
            "ranked": ranked, "top_two": [], "keep_current": True,
            "reasoning": "alpha=1 ranks first -- keep-current rule applies, stage 2 skipped, C retrieval v2 kept.",
        }

    one = stage1_report.get(1.0)
    all_tied_with_one = one is not None and all(
        abs(m["pct_questions_with_relevant"] - one["pct_questions_with_relevant"]) <= TIE_THRESHOLD_PERCENTAGE_POINTS
        and abs(m["mean_relevance_score"] - one["mean_relevance_score"]) <= RELEVANCE_SCORE_TIE_THRESHOLD
        for m in stage1_report.values()
    )
    if all_tied_with_one:
        return {
            "ranked": ranked, "top_two": [], "keep_current": True,
            "reasoning": (
                "Every alpha is tied with alpha=1 on both the primary criterion and the tie-breaker -- "
                "keep-current rule applies, stage 2 skipped, C retrieval v2 kept."
            ),
        }

    top_two = [a for a, _ in ranked[:2]]
    return {
        "ranked": ranked, "top_two": top_two, "keep_current": False,
        "reasoning": (
            f"Top two by primary criterion (% questions with >=1 RELEVANT item), tie-broken by mean relevance "
            f"score then larger alpha: alpha={ranked[0][0]} ({ranked[0][1]}%), "
            f"alpha={ranked[1][0] if len(ranked) > 1 else '-'} "
            f"({ranked[1][1] if len(ranked) > 1 else '-'}%) advance to stage 2."
        ),
    }


def select_stage2_winner(stage2_metrics: dict[float, dict]) -> dict:
    """Stage 2 is run on the POOLED dev answers (grounding on + off
    combined into one set per alpha, not reported/selected separately).

    Secondary criterion: hallucination rate excl. ABSTAIN (ascending),
    then hallucination rate incl. ABSTAIN (ascending) as the tie-break,
    then larger alpha. The tie margin is on the hallucination COUNT (not
    the rate): two alpha are tied on a criterion if their hallucination
    counts, out of that criterion's pooled n, differ by at most
    STAGE2_ANSWER_COUNT_TIE_MARGIN answers.

    `stage2_metrics[alpha]` must contain `hallucination_count_excl_abstain`,
    `n_excl_abstain`, `hallucination_count_incl_abstain`, `n_incl_abstain`.

    Keep-current rule: if the winner is alpha=1, keep current (C
    retrieval v2).

    Returns {"winner": alpha, "keep_current": bool, "reasoning": str}.
    """
    if not stage2_metrics:
        raise ValueError("stage2_metrics must not be empty")

    tiers = [
        (lambda m: -m["hallucination_count_excl_abstain"], STAGE2_ANSWER_COUNT_TIE_MARGIN),
        (lambda m: -m["hallucination_count_incl_abstain"], STAGE2_ANSWER_COUNT_TIE_MARGIN),
    ]
    ranked_alphas = _tiered_rank(stage2_metrics, tiers)
    winner = ranked_alphas[0]
    keep_current = winner == 1.0

    m = stage2_metrics[winner]
    rate_excl = round(m["hallucination_count_excl_abstain"] / m["n_excl_abstain"], 4) if m["n_excl_abstain"] else None
    reasoning = (
        f"alpha={winner} has the lowest pooled hallucination rate (excl. ABSTAIN)="
        f"{rate_excl} ({m['hallucination_count_excl_abstain']}/{m['n_excl_abstain']}) among the stage-1 finalists "
        f"(tie margin: {STAGE2_ANSWER_COUNT_TIE_MARGIN} answers)."
    )
    if keep_current:
        reasoning += " Winner is alpha=1 -- keep-current rule applies, C retrieval v2 kept."
    return {"winner": winner, "keep_current": keep_current, "reasoning": reasoning}
