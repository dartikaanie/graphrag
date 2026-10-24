"""Dev split for conditions A and B (docs/DECISION_C_SCORING.md): same permutation rule as C, existing test
runs untouched, and B's retrieval index re-checked for leakage."""

import pandas as pd
import pytest

from llm.a_pure_llm import a_baseline_replication as a
from llm.b_rag import b_condition_b_rag as b
from llm.manifest import compute_config_hash
from llm.sample_split import sample_questions_split


def _pool(n=600):
    return pd.DataFrame({"Id": list(range(n)), "val": list(range(n))})


@pytest.mark.parametrize("n", [10, 384])
def test_test_split_identical_to_each_conditions_own_sampler(n):
    df = _pool()
    for mod in (a, b):
        pd.testing.assert_frame_equal(mod.sample_questions(df, n, seed=42), sample_questions_split(df, n, 42, "test"))


def test_dev_split_is_zero_based_384_to_433_and_disjoint_from_test():
    df = _pool()
    permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dev = sample_questions_split(df, 50, 42, "dev")
    assert list(dev["Id"]) == list(permuted.iloc[384:434]["Id"])
    assert set(dev["Id"]).isdisjoint(set(sample_questions_split(df, 384, 42, "test")["Id"]))
    with pytest.raises(ValueError):
        sample_questions_split(df, 5, 42, "bogus")


def test_same_dev_questions_as_condition_c():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_graphrag"))
    import c_graphrag

    df = _pool()
    pd.testing.assert_frame_equal(sample_questions_split(df, 50, 42, "dev"),
                                  c_graphrag.sample_questions_split(df, 50, 42, split="dev"))


def test_test_runs_keep_config_and_filename_dev_runs_get_their_own():
    a_test = a.build_config("openai", "gpt-4o-mini", 10, 42, 1536)
    assert "sample_split" not in a_test and a_test == a.build_config("openai", "gpt-4o-mini", 10, 42, 1536, "test")
    a_dev = a.build_config("openai", "gpt-4o-mini", 50, 42, 1536, "dev")
    assert a_dev["sample_split"] == "dev" and compute_config_hash(a_dev) != compute_config_hash(a_test)
    # the committed pilot files keep their exact names
    assert a.build_output_path("r", "openai", "gpt-4o-mini", 10, 42, oversample_pool=1536).name == \
        "condition_a_openai_gpt-4o-mini_n10_seed42_v3_caf85ba006.jsonl"
    assert b.build_output_path("r", "openai", "gpt-4o-mini", 10, 42, True, False, 1536, 5,
                               corpus="legacy_sample", provenance_version=None).name == \
        "condition_b_openai_gpt-4o-mini_n10_seed42_plain_v3_37e6072d55.jsonl"
    assert b.build_output_path("r", "openai", "gpt-4o-mini", 10, 42, True, True, 1536, 5,
                               corpus="legacy_sample", provenance_version=None).name == \
        "condition_b_openai_gpt-4o-mini_n10_seed42_grounded_v3_b0f24482bd.jsonl"
    assert "_dev_" in a.build_output_path("r", "openai", "gpt-4o-mini", 50, 42, oversample_pool=1536,
                                          sample_split="dev").name
    names = {b.build_output_path("r", "openai", "gpt-4o-mini", 50, 42, True, g, 1536, 5, "dev").name for g in (True, False)}
    assert len(names) == 2 and all("_dev_" in n for n in names)
    assert "sample_split" not in b.build_config("openai", "gpt-4o-mini", 10, 42, 1536, 5, True, True)
    # the KG corpus gets its own config_hash / filename; legacy keeps the old ones
    legacy = b.build_config("openai", "gpt-4o-mini", 10, 42, 1536, 5, True, True, corpus="legacy_sample")
    kg = b.build_config("openai", "gpt-4o-mini", 10, 42, 1536, 5, True, True)
    assert "corpus" not in legacy and kg["corpus"] == "kg_answers_v1_tok400"
    assert compute_config_hash(kg) != compute_config_hash(legacy)


def test_b_per_query_leakage_check():
    """B's index-level check was replaced by a per-query assert (llm/common/leakage.py)."""
    from llm.common.leakage import assert_no_leakage, build_exclusion

    retrieved = [{"question_id": 1, "answer_id": 11}, {"question_id": 2, "answer_id": 22}]
    assert_no_leakage(retrieved, build_exclusion(7, [77], 78))  # clean
    with pytest.raises(b.LeakageError, match="1 item"):
        assert_no_leakage(retrieved, build_exclusion(2, [99]))  # own question
    with pytest.raises(b.LeakageError, match="1 item"):
        assert_no_leakage(retrieved, build_exclusion(9, [], accepted_answer_id=11))  # own accepted answer


def test_repo_state_parsing_and_exemptions(monkeypatch):
    import subprocess

    from llm import repo_state as rs

    assert rs.parse_porcelain_z("R  new name.md\0old name.md\0?? a b.jsonl\0 M x.py\0") == \
        [("R ", "new name.md"), ("??", "a b.jsonl"), (" M", "x.py")]

    def fake_run(cmd, **kw):
        out = "abc123\n" if "rev-parse" in cmd else "?? llm/out.jsonl\0"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert rs.repo_state()["git_tree_clean"] is False
    state = rs.repo_state([rs.REPO_ROOT / "llm" / "out.jsonl"])
    assert state["git_tree_clean"] is True and state["git_commit"] == "abc123" and state["git_exempted"] == ["llm/out.jsonl"]
