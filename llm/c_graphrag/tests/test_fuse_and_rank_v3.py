"""Step 5 tests for fuse_and_rank()'s v3 scoring: alpha=1 matches v2
trust_weighted exactly, alpha=0 ranks purely by similarity, min-max
normalization edge cases, and v2's own behavior staying byte-identical
when c_retrieval_version is omitted."""

import numpy as np
import pytest

from c_graphrag import fuse_and_rank, min_max_normalize


class _FakeEmbedModel:
    """Deterministic fake: encode(text) -> a vector derived from the
    text's own length, so similarity differs across texts without any
    real model/network dependency. convert_to_numpy/device/batch_size
    kwargs accepted and ignored, matching SentenceTransformer's own
    signature shape."""

    def encode(self, texts, convert_to_numpy=True, device="cpu", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            # 2-D "embedding": [len(t), 1] then padded -- just needs to be
            # distinguishable and give a well-defined cosine similarity.
            v = np.array([len(t) % 7 + 1, (len(t) % 3) + 1, 1.0], dtype=np.float32)
            out.append(v)
        return np.stack(out)


def _candidate(answer_id, via_question_id=1, title="Q title", body="answer body",
                edge_weight=0.5, trust_score=0.2, is_accepted=False, hop=1, rel_type="HAS_ANSWER"):
    return {
        "via_question_id": via_question_id, "via_question_title": title,
        "answer_id": answer_id, "answer_body": body, "answer_trust_score": trust_score,
        "is_accepted": is_accepted, "edge_weight": edge_weight, "hop": hop, "rel_type": rel_type,
    }


def test_v2_fusion_unaffected_returns_tuple_with_empty_meta():
    graph = [_candidate(1), _candidate(2, edge_weight=0.9)]
    final, meta = fuse_and_rank(graph, [], top_k=5, token_chunk_limit=400)
    assert meta == {}
    assert len(final) == 2
    assert "sim" not in final[0]  # v2 records never carry v3-only fields


def test_v3_alpha_1_matches_v2_trust_weighted_ranking():
    """alpha=1 -> final_score = trust only -> SAME top-5 order as v2
    trust_weighted (same combined_score, same answer_id tie-break)."""
    graph = [
        _candidate(1, edge_weight=0.9, trust_score=0.1),
        _candidate(2, edge_weight=0.2, trust_score=0.9),
        _candidate(3, edge_weight=0.5, trust_score=0.5),
    ]
    v2_final, _ = fuse_and_rank(graph, [], top_k=5, token_chunk_limit=400, fusion_mode="trust_weighted")
    v3_final, meta = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=1.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]),
    )
    assert [c["answer_id"] for c in v2_final] == [c["answer_id"] for c in v3_final]
    assert meta["candidate_pool_size"] == 3


def test_v3_alpha_0_ranks_purely_by_similarity():
    graph = [
        _candidate(1, body="short", edge_weight=0.9, trust_score=0.9),  # high trust, but we check sim-only ranking
        _candidate(2, body="a" * 200, edge_weight=0.1, trust_score=0.1),  # low trust, long body -> different sim
    ]
    final, _ = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=0.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]),
    )
    # alpha=0 -> final_score == sim_norm only; trust is irrelevant to ranking.
    assert final[0]["final_score"] == final[0]["sim_norm"]
    assert final[1]["final_score"] == final[1]["sim_norm"]


def test_v3_records_carry_sim_sim_norm_trust_final_score_alpha_rank():
    graph = [_candidate(1), _candidate(2, edge_weight=0.8)]
    final, _ = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=0.5,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]),
    )
    for i, item in enumerate(final, start=1):
        for key in ("sim", "sim_norm", "trust", "final_score", "alpha", "rank"):
            assert key in item
        assert item["rank"] == i
        assert item["alpha"] == 0.5


def test_v3_accepted_only_filters_pool():
    graph = [_candidate(1, is_accepted=True), _candidate(2, is_accepted=False)]
    final, meta = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=1.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]), accepted_only=True,
    )
    assert meta["candidate_pool_size"] == 1
    assert final[0]["answer_id"] == 1


def test_v3_use_author_trust_blends_author_weight_and_stays_bounded():
    graph = [_candidate(1, edge_weight=1.0, trust_score=1.0)]
    graph[0]["author_weight"] = 1.0
    final, _ = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=1.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]), use_author_trust=True,
    )
    # trust combined_score was 1.0 (max), author_weight 1.0 -> blended trust stays 1.0 (both ends of [0,1])
    assert 0.0 <= final[0]["trust"] <= 1.0
    assert final[0]["trust"] == pytest.approx(1.0)


def test_v3_use_author_trust_defaults_to_zero_when_missing():
    graph = [_candidate(1, edge_weight=1.0, trust_score=1.0)]  # no author_weight key at all
    final, _ = fuse_and_rank(
        graph, [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=1.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]), use_author_trust=True,
    )
    combined = graph[0]["edge_weight"] * 0.7 + graph[0]["answer_trust_score"] * 0.3
    expected_trust = (1 - 0.3) * combined + 0.3 * 0.0
    assert final[0]["trust"] == pytest.approx(round(expected_trust, 4), abs=1e-6)


def test_v3_empty_pool_returns_empty_with_zero_pool_size():
    final, meta = fuse_and_rank(
        [], [], top_k=5, token_chunk_limit=400, c_retrieval_version="v3", alpha=1.0,
        embed_model=_FakeEmbedModel(), query_emb=np.array([1.0, 0.0, 0.0]),
    )
    assert final == []
    assert meta == {"candidate_pool_size": 0}


# ---------------------------------------------------------------------
# min_max_normalize edge cases
# ---------------------------------------------------------------------

def test_min_max_normalize_all_equal_returns_ones():
    assert min_max_normalize([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_min_max_normalize_single_candidate_returns_one():
    assert min_max_normalize([0.3]) == [1.0]


def test_min_max_normalize_empty_returns_empty():
    assert min_max_normalize([]) == []


def test_min_max_normalize_spreads_0_to_1():
    result = min_max_normalize([2.0, 4.0, 6.0])
    assert result == [0.0, 0.5, 1.0]
