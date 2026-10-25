"""BLOCKING requirement (docs/agent_prompt_c_retrieval_v3_devset.md): adding
alpha/sample_split/max_hops/edge_types/use_author_trust/accepted_only, and
making c_retrieval_version a real parameter instead of a hardcoded module
constant, must NOT change the config_hash of any existing run.

This test recomputes config_hash for all 9 real pilot runs (A/B/C/D,
prompt_version v3, the 2026-10-05 pilot) from their ACTUAL stored
run_history.jsonl parameters and asserts an EXACT match against their
ACTUAL stored config_hash -- not a synthetic fixture, the real recorded
values, read directly from the real run_history files.
"""

import json
from pathlib import Path

from llm import archive_paths
from llm.manifest import compute_config_hash

import c_graphrag
from llm.a_pure_llm import a_baseline_replication
from llm.b_rag import b_condition_b_rag
from llm.d_lightrag import d_lightrag

REPO_ROOT = Path(__file__).resolve().parents[3]


def _pilot_v3_records(path: Path) -> list[dict]:
    records = []
    # The 9 pilot runs live in the pre-official ARCHIVE. A live llm/<cond>/logs/run_history.jsonl (written by any
    # later run, e.g. a retrieval-only smoke test) must not shadow it: read the archived copy explicitly.
    with open(archive_paths.archived(path) or path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Pilot generation runs only -- stage-1 retrieval-only dev runs
            # (whose filenames also contain "v3_") are a different record kind.
            # ... and dev-split runs (calibration / stage 2) carry sample_split in their config; the
            # 9 pilot runs are test-split runs.
            if (r.get("config_hash") and "v3_" in str(r.get("output_path", "")) and not r.get("retrieval_only")
                    and r.get("sample_split", "test") == "test"):
                records.append(r)
    return records


def test_all_9_real_pilot_run_config_hashes_unchanged():
    checked = 0

    for r in _pilot_v3_records(REPO_ROOT / "llm" / "a_pure_llm" / "logs" / "run_history.jsonl"):
        config = a_baseline_replication.build_config(
            r["provider"], r["model"], r["n_sample_target"], r["seed"], r["oversample_pool"],
        )
        assert compute_config_hash(config) == r["config_hash"], f"A run {r['output_path']} hash changed"
        checked += 1

    for r in _pilot_v3_records(REPO_ROOT / "llm" / "b_rag" / "logs" / "run_history.jsonl"):
        config = b_condition_b_rag.build_config(
            r["provider"], r["model"], r["n_sample_target"], r["seed"], r["oversample_pool"],
            r["top_k"], r["require_citation"], r["require_grounding"],
            corpus="legacy_sample",  # every pilot run predates the KG corpus
            provenance_version=None,  # ... and item provenance (part of config_hash since v3)
        )
        assert compute_config_hash(config) == r["config_hash"], f"B run {r['output_path']} hash changed"
        checked += 1

    for r in _pilot_v3_records(REPO_ROOT / "llm" / "c_graphrag" / "logs" / "run_history.jsonl"):
        config = c_graphrag.build_config(
            r["provider"], r["model"], r["n_sample_target"], r["seed"], r["oversample_pool"],
            r["top_k"], r["n_anchor"], r["n_semantic_expansion"], r["fusion_mode"],
            r["fusion_w_path_trust"], r["fusion_w_answer_intrinsic_trust"],
            r["semantic_expansion_trust_cap"], r["require_grounding"], r["enable_semantic_expansion"],
            # c_retrieval_version was ALWAYS "v2" for these runs (the global
            # constant's only value before this feature existed) -- no
            # alpha/sample_split/exploratory-switch overrides, so the new
            # optional params are left at their defaults deliberately.
            c_retrieval_version=r["c_retrieval_version"], provenance_version=None,
        )
        assert compute_config_hash(config) == r["config_hash"], f"C run {r['output_path']} hash changed"
        checked += 1

    for r in _pilot_v3_records(REPO_ROOT / "llm" / "d_lightrag" / "logs" / "run_history.jsonl"):
        config = d_lightrag.build_config(
            r["provider"], r["model"], r["n_sample_target"], r["seed"], r["oversample_pool"],
            r["top_k"], r["n_low_level"], r["n_high_level"], r["require_grounding"], provenance_version=None,
            d_retrieval_version="v3",  # the version these pilot runs were made with (v4 is the current default)
        )
        assert compute_config_hash(config) == r["config_hash"], f"D run {r['output_path']} hash changed"
        checked += 1

    assert checked == 9, f"expected to check all 9 real pilot runs, checked {checked}"


def test_build_config_v2_default_dict_is_byte_identical_to_pre_feature_shape():
    """Direct proof that build_config() with no v3/exploratory overrides
    produces EXACTLY the dict shape it always did -- no new keys appear
    at all (not even with default values), which is what makes the hash
    above stable. c_retrieval_version is passed explicitly as "v2" here
    (the caller-visible default), matching every real run checked above."""
    config = c_graphrag.build_config(
        "openai", "gpt-4o-mini", 10, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4, True, True,
        c_retrieval_version="v2", provenance_version=None,
    )
    assert set(config.keys()) == {
        "condition", "provider", "model", "n_sample", "seed", "oversample_pool", "top_k",
        "n_anchor", "n_semantic_expansion", "fusion_mode", "fusion_w_path_trust", "fusion_w_intrinsic",
        "semantic_expansion_trust_cap", "require_grounding", "enable_semantic_expansion",
        "c_retrieval_version",
    }


def test_provenance_version_is_part_of_the_config_hash_of_new_runs():
    """Since provenance v3 a B/C/D config carries provenance_version (so its config_hash, and the
    output filename, differ from a pre-provenance run's); None reproduces the legacy config."""
    from llm.common.item_provenance import PROVENANCE_VERSION

    new = c_graphrag.build_config("openai", "gpt-4o-mini", 10, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4,
                                  True, True, c_retrieval_version="v2")
    legacy = c_graphrag.build_config("openai", "gpt-4o-mini", 10, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4,
                                     True, True, c_retrieval_version="v2", provenance_version=None)
    assert new["provenance_version"] == PROVENANCE_VERSION == "v3" and "provenance_version" not in legacy
    assert {k: v for k, v in new.items() if k != "provenance_version"} == legacy
    assert compute_config_hash(new) != compute_config_hash(legacy)
    for mod, args in ((b_condition_b_rag, ("openai", "m", 10, 42, 1536, 5, True, False)),
                      (d_lightrag, ("openai", "m", 10, 42, 1536, 5, 3, 3, True))):
        assert mod.build_config(*args)["provenance_version"] == "v3"
        assert "provenance_version" not in mod.build_config(*args, provenance_version=None)


def test_each_exploratory_switch_changes_config_hash_when_non_default():
    from llm.manifest import compute_config_hash

    def _build(**overrides):
        kwargs = dict(c_retrieval_version="v3", alpha=1.0)
        kwargs.update(overrides)
        return c_graphrag.build_config(
            "openai", "gpt-4o-mini", 10, 42, 1536, 5, 3, 3, "trust_weighted", 0.7, 0.3, 0.4, True, True,
            **kwargs,
        )

    base_hash = compute_config_hash(_build())

    variants = [
        dict(sample_split="dev"),
        dict(max_hops=1),
        dict(edge_types=("HAS_ACCEPTED_ANSWER",)),
        dict(use_author_trust=True),
        dict(accepted_only=True),
        dict(alpha=0.5),
    ]
    for overrides in variants:
        cfg = _build(**overrides)
        assert compute_config_hash(cfg) != base_hash, f"{overrides} did not change config_hash"


def test_batch_id_never_affects_config_hash():
    """Unrelated regression guard already established elsewhere in this
    codebase (batch_id/batch_launched_at are metadata only) -- build_config()
    never takes a batch_id argument at all, so there's nothing to assert
    beyond: adding this feature didn't accidentally introduce one."""
    import inspect

    sig = inspect.signature(c_graphrag.build_config)
    assert "batch_id" not in sig.parameters
    assert "batch_launched_at" not in sig.parameters


def test_thread_leakage_run_never_shares_a_hash_or_filename_with_a_pre_amendment_8_run():
    """Amendment 15: leakage_mode=None is the legacy (pre-Amendment-8) config -- no "leakage" key, hash
    unchanged (asserted by the real-run test above). A "thread" run writes "leakage": "thread", so its
    config_hash -- and therefore its output filename, which is what resume/append keys on -- differs."""
    b_args = ("openai", "gpt-4o-mini", 5, 42, 20, 5, True, True)
    legacy_b, thread_b = (b_condition_b_rag.build_config(*b_args, leakage_mode=m) for m in (None, "thread"))
    assert "leakage" not in legacy_b and thread_b["leakage"] == "thread"
    assert compute_config_hash(legacy_b) != compute_config_hash(thread_b)
    assert (b_condition_b_rag.build_output_path("out", "openai", "gpt-4o-mini", 5, 42, True, True, 20, 5)
            != b_condition_b_rag.build_output_path("out", "openai", "gpt-4o-mini", 5, 42, True, True, 20, 5,
                                                   leakage_mode="thread"))

    d_args = ("openai", "gpt-4o-mini", 5, 42, 20, 5, 3, 3, True)
    legacy_d, thread_d = (d_lightrag.build_config(*d_args, leakage_mode=m) for m in (None, "thread"))
    assert "leakage" not in legacy_d and thread_d["leakage"] == "thread"
    assert compute_config_hash(legacy_d) != compute_config_hash(thread_d)

    c_args = ("openai", "gpt-4o-mini", 5, 42, 20, 5, 3, 3, "trust_weighted", 0.7, 0.3, 1.0, True, True)
    legacy_c, thread_c = (c_graphrag.build_config(*c_args, c_retrieval_version="v3", leakage_mode=m)
                          for m in (None, "thread"))
    assert "leakage" not in legacy_c and thread_c["leakage"] == "thread"
    assert compute_config_hash(legacy_c) != compute_config_hash(thread_c)
