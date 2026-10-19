"""Tests for judge_interjudge_analysis.py (no API, no annotation files)."""

import pytest

from judge_binary import CATEGORY_ABSTAIN as AB, CATEGORY_HALLUCINATED as H, CATEGORY_NOT_HALLUCINATED as N
from judge_interjudge_analysis import (BINARY, categorize, fleiss, fleiss_kappa, pairwise, prevalence,
                                       severity_bucket, to_mode)


def test_fleiss_kappa_textbook_example():
    # Fleiss (1971) style check: 10 items, 3 raters, values computed by hand.
    rows = [[H, H, H], [N, N, N], [H, H, N], [N, N, H], [H, H, H],
            [N, N, N], [N, N, N], [H, N, N], [H, H, H], [N, N, N]]
    # p_H = 13/30, p_N = 17/30; P_e = (169 + 289) / 900; P_bar = (7 * 1 + 3 * 1/3) / 10 = 0.8
    p_e = (169 + 289) / 900
    assert fleiss_kappa(rows, BINARY) == pytest.approx((0.8 - p_e) / (1 - p_e))


def test_fleiss_kappa_undefined_and_perfect():
    assert fleiss_kappa([[N, N, N], [N, N, N]], BINARY) is None
    assert fleiss_kappa([], BINARY) is None
    assert fleiss_kappa([[N, N, N], [H, H, H]], BINARY) == pytest.approx(1.0)


def test_to_mode():
    assert to_mode(AB, "abstain_excluded") is None
    assert to_mode(AB, "abstain_counted") == N
    assert to_mode(AB, "three_category") == AB
    assert to_mode(None, "abstain_counted") is None


def test_categorize():
    assert categorize([N, N, N], N) == "A"
    assert categorize([N, N, N], H) == "B"
    assert categorize([N, H, N], N) == "C"
    assert categorize([N, None, N], N) == "D"
    assert categorize([AB, AB, AB], AB) == "A"


def test_fleiss_uses_complete_items_only_and_pairwise_counts_n():
    raters = {"llama": {"a": N, "b": H, "c": AB}, "deepseek": {"a": N, "b": None, "c": N},
              "qwen": {"a": N, "b": H, "c": N}}
    items = ["a", "b", "c"]
    assert fleiss(raters, items, "abstain_excluded")["n"] == 1      # b missing, c has an ABSTAIN
    assert fleiss(raters, items, "abstain_counted")["n"] == 2       # b missing
    pw = pairwise(raters, items, "abstain_excluded")
    assert pw["llama_vs_qwen"]["n"] == 2 and pw["llama_vs_deepseek"]["n"] == 1


def test_prevalence_and_severity_bucket():
    p = prevalence({"a": H, "b": N, "c": AB, "d": None}, ["a", "b", "c", "d"])
    assert (p["abstain_excluded"]["x"], p["abstain_excluded"]["n"]) == (1, 2)
    assert (p["abstain_counted"]["x"], p["abstain_counted"]["n"]) == (1, 3)
    assert severity_bucket(None, None) == "missing"
    assert severity_bucket(H, "SEBAGIAN") == "SEBAGIAN"
    assert severity_bucket(H, None) == "UNSPECIFIED"
    assert severity_bucket(N, None) == "FAKTUAL"
