"""Dev/test disjointness and unchanged test-set sampling (Step 5)."""

import pandas as pd

from c_graphrag import sample_questions, sample_questions_split


def _fake_pool(n=500):
    return pd.DataFrame({"Id": list(range(n)), "val": list(range(n))})


def test_test_split_byte_identical_to_sample_questions():
    df = _fake_pool()
    old = sample_questions(df, 384, seed=42)
    new = sample_questions_split(df, 384, seed=42, split="test")
    pd.testing.assert_frame_equal(old, new)


def test_test_split_byte_identical_for_smaller_n_too():
    df = _fake_pool()
    old = sample_questions(df, 10, seed=42)
    new = sample_questions_split(df, 10, seed=42, split="test")
    pd.testing.assert_frame_equal(old, new)


def test_dev_split_disjoint_from_test_regardless_of_n():
    df = _fake_pool()
    test_ids = set(sample_questions_split(df, 384, seed=42, split="test")["Id"])
    for n in (10, 30, 50, 100):
        dev_ids = set(sample_questions_split(df, n, seed=42, split="dev")["Id"])
        assert test_ids.isdisjoint(dev_ids), f"dev n={n} overlaps test positions"


def test_dev_split_positions_385_to_434_for_n_50():
    df = _fake_pool()
    full_permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    expected_ids = list(full_permuted.iloc[384:434]["Id"])
    dev = sample_questions_split(df, 50, seed=42, split="dev", dev_offset=385)
    assert list(dev["Id"]) == expected_ids


def test_dev_offset_is_configurable():
    df = _fake_pool()
    full_permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dev = sample_questions_split(df, 10, seed=42, split="dev", dev_offset=100)
    assert list(dev["Id"]) == list(full_permuted.iloc[99:109]["Id"])


def test_unknown_split_raises():
    import pytest

    df = _fake_pool()
    with pytest.raises(ValueError):
        sample_questions_split(df, 10, seed=42, split="bogus")


def test_dev_and_test_use_same_permutation_same_seed():
    """Disjointness must come from SLICING the same permutation, not a
    different shuffle -- verified by checking position 0 of test equals
    position 0 of the full permutation, and dev's first row equals the
    full permutation's row at dev_offset-1."""
    df = _fake_pool()
    full_permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test = sample_questions_split(df, 5, seed=42, split="test")
    dev = sample_questions_split(df, 5, seed=42, split="dev", dev_offset=385)
    assert test.iloc[0]["Id"] == full_permuted.iloc[0]["Id"]
    assert dev.iloc[0]["Id"] == full_permuted.iloc[384]["Id"]


def test_default_dev_offset_is_zero_based_384_to_433():
    """Pins docs/DECISION_C_SCORING.md's dev set: with the DEFAULT
    dev_offset (385, a 1-based position), dev must be exactly 0-based
    positions 384..433 of the permutation (iloc[384:434]) -- the first
    row right after the test sample's last row (0-based 383)."""
    df = _fake_pool()
    full_permuted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dev = sample_questions_split(df, 50, seed=42, split="dev")
    positions = [int(full_permuted.index[full_permuted["Id"] == i][0]) for i in dev["Id"]]
    assert positions == list(range(384, 434))
    test = sample_questions_split(df, 384, seed=42, split="test")
    assert int(full_permuted.index[full_permuted["Id"] == test["Id"].iloc[-1]][0]) == 383
