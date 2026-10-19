"""Panel aggregation (Amendment 12 (d)): all 4^3 = 64 label combinations, the missing-judge cases and the
severity-unspecified label. Offline."""

import statistics
from itertools import product

import pytest

from judge_panel_aggregate import (ABSTAIN, FAKTUAL, JUDGE_LABELS, PANEL_LABELS, PENUH, SEBAGIAN, UNRESOLVED,
                                   UNSPECIFIED, aggregate_labels, panel_category)

JIDS = ("llama", "deepseek", "qwen")
ORD = {ABSTAIN: 0, FAKTUAL: 0, SEBAGIAN: 1, PENUH: 2}
FROM_ORD = {0: FAKTUAL, 1: SEBAGIAN, 2: PENUH}
ALL_64 = list(product(JUDGE_LABELS, repeat=3))


def _agg(*labels):
    return aggregate_labels(dict(zip(JIDS, labels)))


def _expected(labels):
    """Independent oracle written from the rule text."""
    if labels.count(ABSTAIN) >= 2:
        return ABSTAIN
    return FROM_ORD[int(statistics.median(ORD[x] for x in labels))]


def test_there_are_64_combinations():
    assert len(ALL_64) == 64 == len(set(ALL_64))


@pytest.mark.parametrize("labels", ALL_64, ids=["-".join(x[:4] for x in c) for c in ALL_64])
def test_all_64_combinations(labels):
    rec = _agg(*labels)
    assert rec["panel_label"] == _expected(labels)
    assert rec["panel_label"] in PANEL_LABELS and rec["panel_label"] not in (UNRESOLVED, UNSPECIFIED)
    assert rec["panel_incomplete"] is False
    assert rec["unanimous"] == (len(set(labels)) == 1)
    assert rec["vote_breakdown"]["n_valid"] == 3 and rec["vote_breakdown"]["n_missing"] == 0
    assert sum(rec["vote_breakdown"]["labels"].values()) == 3
    assert rec["panel_id"] == "panel-v1"
    # binary = 2-of-3 majority vote on hallucinated yes/no (ABSTAIN counts as "no")
    majority = sum(1 for x in labels if x in (SEBAGIAN, PENUH)) >= 2
    if rec["panel_label"] == ABSTAIN:
        assert rec["panel_hallucinated"] is None and rec["panel_abstain"] is True
        assert majority is False
    else:
        assert rec["panel_hallucinated"] is majority
        assert rec["panel_abstain"] is False


def test_order_invariant():
    for labels in ALL_64:
        out = {_agg(*perm)["panel_label"] for perm in {labels, labels[::-1], labels[1:] + labels[:1]}}
        assert len(out) == 1


@pytest.mark.parametrize("labels,expected", [
    ((ABSTAIN, ABSTAIN, PENUH), ABSTAIN),
    ((ABSTAIN, PENUH, PENUH), PENUH),          # median(0, 2, 2) = 2
    ((ABSTAIN, SEBAGIAN, PENUH), SEBAGIAN),    # median(0, 1, 2) = 1
    ((ABSTAIN, FAKTUAL, PENUH), FAKTUAL),      # one ABSTAIN only: scored as 0
    ((FAKTUAL, SEBAGIAN, PENUH), SEBAGIAN),
    ((FAKTUAL, FAKTUAL, PENUH), FAKTUAL),
    ((SEBAGIAN, PENUH, PENUH), PENUH),
])
def test_hand_cases(labels, expected):
    assert _agg(*labels)["panel_label"] == expected


# --- missing judges ---------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    (FAKTUAL, FAKTUAL, FAKTUAL),
    (ABSTAIN, ABSTAIN, ABSTAIN),
    (PENUH, PENUH, PENUH),
    (SEBAGIAN, SEBAGIAN, SEBAGIAN),
    (SEBAGIAN, PENUH, SEBAGIAN),   # agree on hallucinated; lower severity
    (PENUH, SEBAGIAN, SEBAGIAN),
])
def test_two_valid_agreeing(a, b, expected):
    rec = aggregate_labels({"llama": a, "deepseek": None, "qwen": b})
    assert rec["panel_label"] == expected
    assert rec["panel_incomplete"] is True
    assert rec["unanimous"] is False
    assert rec["vote_breakdown"]["n_valid"] == 2 and rec["vote_breakdown"]["n_missing"] == 1


@pytest.mark.parametrize("a,b", [
    (FAKTUAL, PENUH), (FAKTUAL, SEBAGIAN),     # disagree on hallucinated
    (ABSTAIN, FAKTUAL), (ABSTAIN, PENUH),      # disagree on ABSTAIN status
])
def test_two_valid_disagreeing(a, b):
    for x, y in ((a, b), (b, a)):
        rec = aggregate_labels({"llama": None, "deepseek": x, "qwen": y})
        assert rec["panel_label"] == UNRESOLVED
        assert rec["panel_hallucinated"] is None and rec["panel_abstain"] is None
        assert rec["panel_incomplete"] is False


@pytest.mark.parametrize("label", JUDGE_LABELS)
def test_one_valid(label):
    rec = aggregate_labels({"llama": label, "deepseek": None, "qwen": None})
    assert rec["panel_label"] == UNRESOLVED and rec["panel_hallucinated"] is None
    assert rec["vote_breakdown"]["n_valid"] == 1


def test_zero_valid():
    rec = aggregate_labels({"llama": None, "deepseek": None, "qwen": None})
    assert rec["panel_label"] == UNRESOLVED and rec["panel_hallucinated"] is None
    assert rec["vote_breakdown"]["n_valid"] == 0 and rec["vote_breakdown"]["n_missing"] == 3
    assert rec["unanimous"] is False


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        _agg(FAKTUAL, "HALU", PENUH)


# --- severity unspecified (never imputed) ------------------------------------

@pytest.mark.parametrize("labels,expected", [
    ((UNSPECIFIED, PENUH, PENUH), PENUH),            # median is 2 either way
    ((UNSPECIFIED, SEBAGIAN, SEBAGIAN), SEBAGIAN),
    ((UNSPECIFIED, FAKTUAL, FAKTUAL), FAKTUAL),
    ((UNSPECIFIED, SEBAGIAN, PENUH), UNSPECIFIED),   # 1 or 2 depending on the unknown severity
    ((UNSPECIFIED, FAKTUAL, PENUH), UNSPECIFIED),
    ((UNSPECIFIED, FAKTUAL, SEBAGIAN), SEBAGIAN),    # median 1 either way
    ((UNSPECIFIED, ABSTAIN, ABSTAIN), ABSTAIN),
    ((UNSPECIFIED, UNSPECIFIED, FAKTUAL), UNSPECIFIED),
])
def test_unspecified_three(labels, expected):
    rec = _agg(*labels)
    assert rec["panel_label"] == expected
    assert rec["panel_hallucinated"] is (None if expected == ABSTAIN else expected != FAKTUAL)


@pytest.mark.parametrize("a,b,expected", [
    (UNSPECIFIED, SEBAGIAN, SEBAGIAN),   # lower of {1|2, 1} is 1
    (UNSPECIFIED, PENUH, UNSPECIFIED),   # lower of {1|2, 2} unknown
    (UNSPECIFIED, FAKTUAL, UNRESOLVED),
])
def test_unspecified_two(a, b, expected):
    assert aggregate_labels({"llama": a, "deepseek": b, "qwen": None})["panel_label"] == expected


def test_binary_never_depends_on_unspecified():
    for labels in product(JUDGE_LABELS + (UNSPECIFIED,), repeat=3):
        rec = _agg(*labels)
        if rec["panel_label"] == ABSTAIN:
            continue
        assert rec["panel_hallucinated"] is (sum(1 for x in labels if x in (SEBAGIAN, PENUH, UNSPECIFIED)) >= 2)


def test_panel_category():
    assert panel_category(ABSTAIN) == "ABSTAIN"
    assert panel_category(FAKTUAL) == "NOT_HALLUCINATED"
    assert panel_category(PENUH) == panel_category(SEBAGIAN) == panel_category(UNSPECIFIED) == "HALLUCINATED"
    assert panel_category(UNRESOLVED) is None
