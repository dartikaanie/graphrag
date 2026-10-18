"""Frozen sample-ID lists (llm/sample_ids/*_v1.json) and the check every
condition runs right after sampling (llm/sample_split.verify_sample).
Fake pools and fake frozen lists in tmp_path; the real files are only read."""

import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import llm.sample_split as ss
from llm.a_pure_llm import a_baseline_replication as a
from llm.b_rag import b_condition_b_rag as b
from llm.d_lightrag import d_lightrag as d

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llm" / "c_graphrag"))
import c_graphrag as cg  # noqa: E402

for _m in (a, b, cg, d):
    _m.log = lambda *x, **k: None


def _pool(n=600, offset=0):
    return pd.DataFrame({"Id": [1000 + offset + i for i in range(n)], "val": range(n)})


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Frozen lists built from a fake pool with the shared sampler, exactly
    as build_sample_ids.py builds the real ones."""
    pool = _pool()
    files = {}
    for split, n, offset in (("test", 384, None), ("dev", 50, 385)):
        ids = [int(i) for i in ss.sample_questions_split(pool, n, 42, split)["Id"]]
        files[split] = tmp_path / f"{split}_ids_v1.json"
        files[split].write_text(json.dumps({"split": split, "dev_offset": offset, "ids": ids,
                                            "ids_sha256": ss.ids_sha256(ids)}))
    monkeypatch.setattr(ss, "FROZEN_FILES", files)
    return pool, files


@pytest.mark.parametrize("n", [1, 10, 384])
def test_every_conditions_test_sampler_matches_the_frozen_list(frozen, n):
    pool, _ = frozen
    samples = {
        "A": a.sample_questions(pool, n, 42),
        "B": b.sample_questions(pool, n, 42),
        "C": cg.sample_questions_split(pool, n, 42, split="test"),
        "D": d.sample_questions(pool, n, 42),
        "shared": ss.sample_questions_split(pool, n, 42, "test"),
    }
    for cond, df in samples.items():
        info = ss.verify_sample(df["Id"], "test", n)
        assert info["n_checked"] == n, cond


@pytest.mark.parametrize("n", [35, 50])
def test_every_conditions_dev_sampler_matches_the_frozen_list(frozen, n):
    pool, _ = frozen
    for df in (ss.sample_questions_split(pool, n, 42, "dev"), cg.sample_questions_split(pool, n, 42, split="dev")):
        assert ss.verify_sample(df["Id"], "dev", n)["n_checked"] == n


def test_mismatch_raises(frozen):
    pool, _ = frozen
    good = list(ss.sample_questions_split(pool, 10, 42, "test")["Id"])
    with pytest.raises(ss.SampleMismatchError, match="position 1"):
        ss.verify_sample(list(reversed(good)), "test", 10)
    with pytest.raises(ss.SampleMismatchError):  # other seed
        ss.verify_sample(ss.sample_questions_split(pool, 10, 7, "test")["Id"], "test", 10)
    with pytest.raises(ss.SampleMismatchError):  # other pool (e.g. a different oversample_pool)
        ss.verify_sample(ss.sample_questions_split(_pool(offset=1), 10, 42, "test")["Id"], "test", 10)
    with pytest.raises(ss.SampleMismatchError):  # dev questions passed as test
        ss.verify_sample(ss.sample_questions_split(pool, 10, 42, "dev")["Id"], "test", 10)


def test_runtime_bounds(frozen):
    pool, _ = frozen
    with pytest.raises(ss.SampleMismatchError, match="n_sample=385"):
        ss.assert_split_bounds("test", 385)
    for offset in (1, 300, 384):
        with pytest.raises(ss.SampleMismatchError, match="must be > 384"):
            ss.assert_split_bounds("dev", 50, offset)
    ss.assert_split_bounds("test", 384)
    ss.assert_split_bounds("dev", 50, 385)
    with pytest.raises(ss.SampleMismatchError, match="frozen dev list starts at position 385"):
        ss.verify_sample(ss.sample_questions_split(pool, 5, 42, "dev", 386)["Id"], "dev", 5, 386)


def test_edited_frozen_file_is_rejected(frozen):
    _, files = frozen
    data = json.loads(files["test"].read_text())
    data["ids"][0], data["ids"][1] = data["ids"][1], data["ids"][0]
    files["test"].write_text(json.dumps(data))
    with pytest.raises(ss.SampleMismatchError, match="edited"):
        ss.load_frozen_ids("test")


def test_real_frozen_files():
    test, dev = ss.load_frozen_ids("test"), ss.load_frozen_ids("dev")
    assert (test["n"], len(test["ids"]), test["positions"]) == (384, 384, "1..384")
    assert (dev["n"], len(dev["ids"]), dev["dev_offset"], dev["positions"]) == (50, 50, 385, "385..434")
    assert not set(test["ids"]) & set(dev["ids"])
    assert test["seed"] == dev["seed"] == 42 and test["oversample_pool"] == 1536
    assert test["n_pool_after_token_filter"] == dev["n_pool_after_token_filter"] == 1471
    assert test["pool_ids_sha256"] == dev["pool_ids_sha256"]


@pytest.mark.parametrize("mod", [a, b, cg, d])
def test_each_main_verifies_the_sample_before_using_it(mod):
    src = inspect.getsource(mod.main)
    check = src.index("verify_sample(sample_df[\"Id\"], args.sample_split, args.n_sample, args.dev_offset)")
    assert check < src.index("get_accepted_answers(")
    assert check < src.index("process_sample(")
    assert "sys.exit(2)" in src[check:check + 300]
