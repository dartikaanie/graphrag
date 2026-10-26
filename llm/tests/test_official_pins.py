"""Pre-run pins (2026-10-26): EMBED_MODEL and the C fusion / expansion parameters are in the official commands and are
checked at start-up, so a changed .env value cannot reach a test-sample run. Offline."""
import inspect
import re
from pathlib import Path

import pytest

from llm import official_run_matrix as orm
from llm import run_registry as rr

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = {"status": "selected", "c_retrieval_version": "v4", "alpha_primary": 0.5, "alpha_ablation": 0.25,
            "max_hops": 2, "edge_types": [], "use_author_trust": False, "accepted_only": False}
GOOD_C = dict(embed_model="all-MiniLM-L6-v2", fusion_w_path_trust=0.7, fusion_w_intrinsic=0.3,
              semantic_expansion_trust_cap=0.4)


def test_frozen_values_are_the_bab_iii_values():
    assert (orm.EMBED_MODEL, orm.FUSION_W_PATH_TRUST, orm.FUSION_W_INTRINSIC, orm.SEMANTIC_EXPANSION_TRUST_CAP) == \
        ("all-MiniLM-L6-v2", 0.7, 0.3, 0.4)
    from llm.c_graphrag import c_graphrag as cg
    assert (cg.DEFAULT_FUSION_W_PATH_TRUST, cg.DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
            cg.DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP) == (orm.FUSION_W_PATH_TRUST, orm.FUSION_W_INTRINSIC,
                                                         orm.SEMANTIC_EXPANSION_TRUST_CAP)


def test_good_values_pass_and_dev_is_never_checked():
    assert orm.pin_error("C", **GOOD_C) is None
    assert orm.pin_error("B", embed_model="all-MiniLM-L6-v2") is None
    assert orm.pin_error("D", embed_model="all-MiniLM-L6-v2") is None
    assert orm.pin_error("C", sample_split="dev", **{**GOOD_C, "fusion_w_path_trust": 0.3}) is None
    assert orm.pin_error("A") is None


@pytest.mark.parametrize("key, bad", [("embed_model", "other-model"), ("fusion_w_path_trust", 0.6),
                                      ("fusion_w_intrinsic", 0.4), ("semantic_expansion_trust_cap", 0.5)])
def test_a_changed_value_refuses_a_test_run(key, bad):
    msg = orm.pin_error("C", **{**GOOD_C, key: bad})
    assert msg and "refused" in msg and repr(bad) in msg


def test_b_and_d_refuse_a_changed_embedding_model():
    for cond in ("B", "D"):
        assert orm.pin_error(cond, embed_model="x")


def test_commands_carry_the_pins():
    for r in orm.build_matrix(OFFICIAL):
        a = r["argv"]
        if r["condition"] in ("B", "C", "D"):
            assert a[a.index("--embed-model") + 1] == orm.EMBED_MODEL
        if r["condition"] == "C":
            assert a[a.index("--fusion-w-path-trust") + 1] == "0.7"
            assert a[a.index("--fusion-w-intrinsic") + 1] == "0.3"
            assert a[a.index("--semantic-expansion-trust-cap") + 1] == "0.4"


@pytest.mark.parametrize("script", ["llm/b_rag/b_condition_b_rag.py", "llm/c_graphrag/c_graphrag.py",
                                    "llm/d_lightrag/d_lightrag.py"])
def test_each_script_calls_the_check_before_any_work(script):
    src = (REPO_ROOT / script).read_text()
    main = src[src.index("def main("):]
    assert "pin_error(" in main
    for later in ("get_candidate_questions(", "connect_neo4j(", "load_faiss_cache(", "build_or_load_index("):
        if later in main:
            assert main.index("pin_error(") < main.index(later), later


def test_registry_expects_the_c_fusion_values():
    exp = rr.expected_generation_config(rr.STATUS_OFFICIAL, "C")
    assert (exp["fusion_w_path_trust"], exp["fusion_w_intrinsic"], exp["semantic_expansion_trust_cap"]) == (0.7, 0.3, 0.4)
