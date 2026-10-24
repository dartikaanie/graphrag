"""Synthetic low-/high-level candidates for fuse_dual_level() provenance tests,
shaped like retrieve_low_level()/retrieve_high_level() output (incl. the
provenance keys). fixtures/fuse_dual_level_golden_pre_provenance.json holds what
fuse_dual_level() returned for them BEFORE provenance capture existed (produced
from the committed HEAD d_lightrag.py)."""


def _c(via, aid, rel, score, hop, accepted, **extra):
    return {"via_question_id": via, "via_question_title": f"title {via}", "answer_id": aid,
            "answer_body": f"body of {aid}", "hop": hop, "rel_type": rel, "relevance_score": score,
            "is_accepted": accepted, **extra}


def low_candidates():
    return [
        _c(100, 1001, "HAS_ACCEPTED_ANSWER", 0.91, 1, True, anchor_id=100, edge_weight=1.0),
        _c(100, 1002, "HAS_ANSWER", 0.91, 1, False, anchor_id=100, edge_weight=0.5),
        _c(200, 2001, "HAS_ANSWER", 0.85, 1, False, anchor_id=200, edge_weight=0.8),
    ]


def high_candidates():
    return [
        # q2=300 reached by anchors 100 and 200 with the same (weight, type) -> one row, alt_anchor_ids=[200]
        _c(300, 3001, "TAG_COOCCUR", 0.7, 2, False, anchor_id=100, alt_anchor_ids=[200], r2_weight=0.7,
           r3_type="HAS_ANSWER", r3_weight=0.8, edge_weight=0.7 * 0.8),
        # answer 1001 again, lower score than its low-level hit -> loses the dedup
        _c(301, 1001, "EMBED_SIM", 0.4, 2, True, anchor_id=200, alt_anchor_ids=[], r2_weight=0.4,
           r3_type="HAS_ACCEPTED_ANSWER", r3_weight=1.0, edge_weight=0.4),
        _c(302, 3002, "IS_RELATED_TO", 0.95, 2, None, anchor_id=100, alt_anchor_ids=[], r2_weight=0.95,
           r3_type="HAS_ANSWER", r3_weight=0.9, edge_weight=0.95 * 0.9),
    ]
