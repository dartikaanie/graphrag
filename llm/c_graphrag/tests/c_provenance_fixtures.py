"""Synthetic traversal/expansion candidates for fuse_and_rank() provenance tests,
shaped like traverse_graph()/semantic_expansion() output (including the
provenance keys: anchor_id, r2_weight, r3_weight, seed_id, seed_similarity).

The set is built so that dedup matters: answer 1001 is reachable as a 1-hop
item AND as a lower-scoring 2-hop item, answer 3001 is reachable from two
anchors and from semantic expansion, answer 1002 has two 1-hop edge types.
fixtures/fuse_and_rank_golden_pre_provenance.json holds what fuse_and_rank()
returned for them (see CONFIGS / run_config()) BEFORE provenance capture existed:
it was produced by running run_config() on the committed HEAD (6af872f)
c_graphrag.py.
"""

import numpy as np


class FakeEmbedModel:
    """Deterministic stand-in for SentenceTransformer (same shape as the one in
    test_fuse_and_rank_v3.py)."""

    def encode(self, texts, convert_to_numpy=True, device="cpu", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([np.array([len(t) % 7 + 1, (len(t) % 3) + 1, 1.0], dtype=np.float32) for t in texts])


QUERY_EMB = np.array([1.0, 0.0, 0.0])


def _c(via, title, aid, body, edge_weight, trust, accepted, hop, rel, **extra):
    return {"via_question_id": via, "via_question_title": title, "answer_id": aid, "answer_body": body,
            "answer_trust_score": trust, "is_accepted": accepted, "edge_weight": edge_weight, "hop": hop,
            "rel_type": rel, **extra}


def graph_candidates():
    return [
        # 1-hop (anchor == via_question_id)
        _c(100, "anchor one", 1001, "accepted answer of 100", 1.0, 0.9, True, 1, "HAS_ACCEPTED_ANSWER", anchor_id=100),
        _c(100, "anchor one", 1002, "second answer of 100", 0.5, 0.3, False, 1, "HAS_ANSWER", anchor_id=100),
        _c(100, "anchor one", 1002, "second answer of 100", 0.5, 0.3, True, 1, "HAS_ACCEPTED_ANSWER", anchor_id=100),
        _c(200, "anchor two", 2001, "answer of 200", 0.8, 0.5, False, 1, "HAS_ANSWER", anchor_id=200),
        # 2-hop: edge_weight == r2_weight * r3_weight
        _c(300, "related q 300", 3001, "answer of 300", 0.5 * 0.8, 0.6, False, 2, "TAG_COOCCUR->HAS_ANSWER",
           anchor_id=100, r2_weight=0.5, r3_weight=0.8),
        _c(300, "related q 300", 3001, "answer of 300", 0.4 * 0.8, 0.6, False, 2, "EMBED_SIM->HAS_ANSWER",
           anchor_id=200, r2_weight=0.4, r3_weight=0.8),
        _c(301, "related q 301", 1001, "accepted answer of 100", 0.4 * 1.0, 0.9, True, 2, "EMBED_SIM->HAS_ACCEPTED_ANSWER",
           anchor_id=200, r2_weight=0.4, r3_weight=1.0),
        _c(302, "related q 302", 3002, "answer of 302", 0.9 * 0.9, 0.2, False, 2, "IS_RELATED_TO->HAS_ANSWER",
           anchor_id=100, r2_weight=0.9, r3_weight=0.9),
    ]


def expansion_candidates():
    # seed 300 (a hop-2 q2) -> expanded Questions 400 / 401 ; 401 owns answer 3001 again
    return [
        _c(400, "expanded q 400", 4001, "answer of 400", 0.7, 0.5, False, "semantic", "HAS_ANSWER",
           seed_id=300, seed_similarity=0.8123),
        _c(401, "expanded q 401", 3001, "answer of 300", 0.9, 0.6, False, "semantic", "HAS_ANSWER",
           seed_id=300, seed_similarity=0.7),
        _c(402, "expanded q 402", 4002, "answer of 402", 0.6, 0.4, True, "semantic", "HAS_ACCEPTED_ANSWER",
           seed_id=100, seed_similarity=0.65),
    ]


# (label, fuse_and_rank kwargs). v3 configs also need embed_model / query_emb (added by the caller).
CONFIGS = {
    "v2_trust_weighted": dict(fusion_mode="trust_weighted"),
    "v2_uniform": dict(fusion_mode="uniform"),
    "v3_alpha_0.5": dict(c_retrieval_version="v3", alpha=0.5),
    "v3_alpha_1_accepted_only": dict(c_retrieval_version="v3", alpha=1.0, accepted_only=True),
}


def run_config(fuse_and_rank, label, top_k=5):
    kwargs = dict(CONFIGS[label])
    if kwargs.get("c_retrieval_version") == "v3":
        kwargs.update(embed_model=FakeEmbedModel(), query_emb=QUERY_EMB)
    items, meta = fuse_and_rank(graph_candidates(), expansion_candidates(), top_k, 400, **kwargs)
    return {"items": items, "meta": meta}
