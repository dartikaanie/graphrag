"""Tests for reputation_ablation_check.py's pure logic (no Neo4j/FAISS/LLM):
the reputation fix formula, that alpha=0 is never affected, that the fix
can reorder candidates when trust matters, and the verdict rule."""

import numpy as np
import pytest

import reputation_ablation_check as rac


class _FakeEnc:
    """tiktoken stand-in (no network): one token per whitespace word."""

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


@pytest.fixture(autouse=True)
def _no_tiktoken_download(monkeypatch):
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _FakeEnc())


class _FakeEmbedModel:
    def encode(self, texts, convert_to_numpy=True, device="cpu", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([np.array([len(t) % 7 + 1, (len(t) % 3) + 1, 1.0], dtype=np.float32) for t in texts])


def _cand(aid, trust, edge=1.0, accepted=False, body="same body text"):
    return {"via_question_id": 10, "via_question_title": "Q", "answer_id": aid, "answer_body": body,
            "answer_trust_score": trust, "is_accepted": accepted, "edge_weight": edge, "hop": 1,
            "rel_type": "HAS_ANSWER"}


def test_fix_replaces_constant_with_author_weight():
    out = rac.apply_reputation_fix([_cand(1, 0.40), _cand(2, 0.40)], {1: 1.0})
    assert out[0]["answer_trust_score"] == pytest.approx(0.40 - 0.1 + 0.2)  # strong author
    assert out[1]["answer_trust_score"] == pytest.approx(0.40 - 0.1)        # no AUTHOR_TRUST edge -> 0


def test_fix_does_not_mutate_input():
    cands = [_cand(1, 0.4)]
    rac.apply_reputation_fix(cands, {1: 1.0})
    assert cands[0]["answer_trust_score"] == 0.4


def test_alpha_zero_unaffected_by_any_variant():
    graph = [_cand(i, 0.4, body="x " * i) for i in range(1, 9)]
    weights = {i: (i % 3) / 2 for i in range(1, 9)}
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    base = rac.rank_variant("base", 0.0, graph, [], weights, _FakeEmbedModel(), q, None)
    for v in ("rep_in_trustscore", "author_switch"):
        other = rac.rank_variant(v, 0.0, graph, [], weights, _FakeEmbedModel(), q, None)
        assert not rac.compare_topk(base, other, weights)["order_changed"]


def test_fix_can_reorder_at_alpha_one():
    # Identical stored trust (reputation was a constant), different authors:
    # with the fix, the high-reputation author's answer must move to rank 1.
    graph = [_cand(1, 0.4), _cand(2, 0.4)]
    weights = {2: 1.0}
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    base = rac.rank_variant("base", 1.0, graph, [], weights, _FakeEmbedModel(), q, None, top_k=1)
    fixed = rac.rank_variant("rep_in_trustscore", 1.0, graph, [], weights, _FakeEmbedModel(), q, None, top_k=1)
    assert base[0]["answer_id"] == 1  # tie broken by answer_id
    assert fixed[0]["answer_id"] == 2
    cmp = rac.compare_topk(base, fixed, weights)
    assert cmp["set_changed"] and cmp["rank1_changed"] and cmp["entered_ids"] == [2]


def test_verdict_threshold():
    def s(pct):
        return {"rep_in_trustscore@0.5": {"variant": "rep_in_trustscore", "alpha": 0.5, "pct_set_changed": pct},
                "rep_in_trustscore@0.0": {"variant": "rep_in_trustscore", "alpha": 0.0, "pct_set_changed": 99.0}}
    assert rac.verdict(s(4.0))["label"] == "NETRAL"        # alpha=0 ignored
    assert rac.verdict(s(5.0))["label"] == "NETRAL"        # boundary inclusive
    assert rac.verdict(s(6.0))["label"] == "BERPENGARUH"
