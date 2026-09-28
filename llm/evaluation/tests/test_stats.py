"""Test offline untuk _stats_common.py (Cliff's delta, koreksi Holm) --
murni fungsi matematis atas angka diketahui, tanpa I/O apa pun."""

import pytest

from _stats_common import cliffs_delta, holm_correction


# ---------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------

def test_cliffs_delta_identical_distributions():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    delta, band = cliffs_delta(x, y)
    assert delta == pytest.approx(0.0)
    assert band == "negligible"


def test_cliffs_delta_complete_dominance():
    x = [10, 11, 12]
    y = [1, 2, 3]
    delta, band = cliffs_delta(x, y)
    assert delta == pytest.approx(1.0)
    assert band == "large"


def test_cliffs_delta_complete_dominance_reversed():
    x = [1, 2, 3]
    y = [10, 11, 12]
    delta, band = cliffs_delta(x, y)
    assert delta == pytest.approx(-1.0)
    assert band == "large"


def test_cliffs_delta_known_value_with_ties():
    # x=[1,2,3] vs y=[2,3,4]: 9 pasangan total.
    # x=1: <2,<3,<4            -> 0 gt, 3 lt
    # x=2: =2(tie),<3,<4       -> 0 gt, 2 lt
    # x=3: >2,=3(tie),<4       -> 1 gt, 1 lt
    # gt=1, lt=6 -> delta=(1-6)/9
    x = [1, 2, 3]
    y = [2, 3, 4]
    delta, band = cliffs_delta(x, y)
    assert delta == pytest.approx(-5 / 9)
    assert band == "large"  # |-0.5556| > 0.474


def test_cliffs_delta_empty_input():
    assert cliffs_delta([], [1, 2]) == (None, None)
    assert cliffs_delta([1, 2], []) == (None, None)


# ---------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------

def test_holm_correction_classic_example():
    # p=[0.01,0.02,0.03,0.04], n=4, multipliers (n-rank)=[4,3,2,1] sesuai
    # urutan ascending -> raw adjusted [0.04,0.06,0.06,0.04], step-down
    # monotonic enforcement -> [0.04,0.06,0.06,0.06].
    p = [0.01, 0.02, 0.03, 0.04]
    adj = holm_correction(p)
    assert adj[0] == pytest.approx(0.04)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.06)
    assert adj[3] == pytest.approx(0.06)


def test_holm_correction_preserves_input_order():
    # urutan TIDAK ascending -- adjusted harus tetap sejajar index input asal.
    p = [0.04, 0.01, 0.03, 0.02]
    adj = holm_correction(p)
    # p[1]=0.01 adalah yang terkecil (rank0, multiplier4) -> 0.04
    assert adj[1] == pytest.approx(0.04)
    # p[3]=0.02 rank1, multiplier3 -> 0.06
    assert adj[3] == pytest.approx(0.06)
    # p[2]=0.03 rank2, multiplier2 -> 0.06 (running max dgn 0.06 sebelumnya)
    assert adj[2] == pytest.approx(0.06)
    # p[0]=0.04 rank3, multiplier1 -> raw 0.04, tapi running max 0.06 -> 0.06
    assert adj[0] == pytest.approx(0.06)


def test_holm_correction_caps_at_one():
    p = [0.9, 0.95]
    adj = holm_correction(p)
    assert all(a <= 1.0 for a in adj)


def test_holm_correction_empty():
    assert holm_correction([]) == []


def test_holm_correction_single_value_unchanged():
    assert holm_correction([0.03]) == [pytest.approx(0.03)]
