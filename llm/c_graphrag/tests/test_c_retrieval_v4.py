"""Condition C retrieval v4 (Amendment 17) = v3 ranking + answerable anchors."""

import c_graphrag as cg
from llm import official_run_matrix as orm
from llm.c_graphrag import official_params as op
from llm.manifest import compute_config_hash

ARGS = ("openai", "gpt-4o-mini", 384, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4, True, True)


def test_v4_is_the_default_official_version_and_ranks_like_v3():
    assert cg.C_RETRIEVAL_VERSION == orm.C_RETRIEVAL_VERSION == op.OFFICIAL_C_RETRIEVAL_VERSION == "v4"
    assert cg.alpha_ranking("v3") and cg.alpha_ranking("v4") and not cg.alpha_ranking("v2")
    assert cg.C_RETRIEVAL_VERSIONS == ("v2", "v3", "v4")


def test_config_hash_changes_with_the_version_and_carries_alpha():
    cfgs = {v: cg.build_config(*ARGS, c_retrieval_version=v, alpha=0.25, sample_split="test") for v in ("v3", "v4")}
    assert cfgs["v4"]["c_retrieval_version"] == "v4" and cfgs["v4"]["alpha"] == 0.25
    assert compute_config_hash(cfgs["v3"]) != compute_config_hash(cfgs["v4"])


def test_v4_output_name_and_matrix_cell():
    path = cg.build_output_path("r", "openai", "gpt-4o-mini", 384, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4,
                                True, True, c_retrieval_version="v4", alpha=0.25)
    assert "_v4-alpha0-25" in str(path)
    row = next(r for r in orm.build_matrix(None) if r["condition"] == "C")
    assert row["argv"][row["argv"].index("--c-retrieval-version") + 1] == "v4"


def test_the_same_fusion_runs_for_v3_and_v4():
    import numpy as np

    cands = [{"answer_id": a, "via_question_id": 1, "via_question_title": "t", "answer_body": "b", "edge_weight": w,
              "answer_trust_score": 0.5, "hop": 1, "rel_type": "HAS_ANSWER", "anchor_id": 1, "is_accepted": False}
             for a, w in ((1, 1.0), (2, 0.5), (3, 0.2))]

    class M:
        def encode(self, texts, **k):
            return np.stack([np.eye(8)[i % 8] for i, _ in enumerate(texts)]).astype("float32")

    outs = []
    for v in ("v3", "v4"):
        items, meta = cg.fuse_and_rank(cands, [], 5, 400, c_retrieval_version=v, alpha=0.5, embed_model=M(),
                                       query_emb=np.eye(8)[:1].astype("float32"))
        outs.append([(i["answer_id"], i["final_score"]) for i in items])
    assert outs[0] == outs[1] and len(outs[0]) == 3
