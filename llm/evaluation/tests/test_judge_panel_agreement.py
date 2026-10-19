"""judge_panel_agreement (Amendment 12 (e)) on synthetic labels. Offline; no human or judge data."""

import pytest

import judge_panel_agreement as ag

A, N, H = "ABSTAIN", "NOT_HALLUCINATED", "HALLUCINATED"

HUMAN = {"i1": H, "i2": H, "i3": N, "i4": N, "i5": A, "i6": H, "i7": None}
COND = {"i1": "A", "i2": "A", "i3": "B-plain", "i4": "B-plain", "i5": "B-grounded", "i6": "B-grounded", "i7": "A"}


def test_binary_metrics_counts():
    # rater: hits i1, misses i2 (unsafe pass), false alarm on i3, agrees on i4; i6 missing
    rater = {"i1": H, "i2": N, "i3": H, "i4": N, "i5": N, "i6": None, "i7": H}
    a, b, counts = ag.binary_pairs(rater, HUMAN, "abstain_excluded")
    assert counts == {"n_missing": 2, "n_abstain_excluded": 1}
    m = ag.binary_metrics(a, b, n_resamples=0)
    assert m["n"] == 4
    assert (m["raw_agreement"]["x"], m["raw_agreement"]["n"]) == (2, 4)
    assert (m["hallucination_recall"]["x"], m["hallucination_recall"]["n"]) == (1, 2)
    assert (m["unsafe_pass_rate"]["x"], m["unsafe_pass_rate"]["n"]) == (1, 2)
    assert (m["false_alarm_rate"]["x"], m["false_alarm_rate"]["n"]) == (1, 2)
    assert m["hallucination_recall"]["rate"] + m["unsafe_pass_rate"]["rate"] == 1
    assert m["confusion_human_rows"] == {N: {N: 1, H: 1}, H: {N: 1, H: 1}}
    assert m["kappa"] == 0.0


def test_abstain_counted_maps_to_not_hallucinated():
    rater = {"i1": H, "i2": H, "i3": N, "i4": A, "i5": N, "i6": H, "i7": None}
    a, b, counts = ag.binary_pairs(rater, HUMAN, "abstain_counted")
    assert counts == {"n_missing": 1}
    m = ag.binary_metrics(a, b, n_resamples=0)
    assert m["n"] == 6 and m["raw_agreement"]["x"] == 6 and m["kappa"] == 1.0


def test_kappa_matches_hand_computation():
    # 10 items: po = 0.8, pe = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa 0.6
    human = [H] * 5 + [N] * 5
    rater = [H] * 4 + [N] + [H] + [N] * 4
    assert ag.binary_metrics(rater, human, n_resamples=0)["kappa"] == pytest.approx(0.6)


def test_report_structure_and_acceptance():
    raters = {r: dict(HUMAN) for r in ag.RATERS}
    raters["qwen"]["i1"] = N
    rep = ag.agreement_report(HUMAN, raters, COND, n_resamples=50)
    assert set(rep["vs_human"]) == set(ag.RATERS)
    assert set(rep["vs_human"]["panel"]) == set(ag.MODES)
    assert rep["vs_human"]["panel"]["abstain_excluded"]["kappa"] == 1.0
    assert rep["three_category"]["panel"]["kappa"] == 1.0 and rep["three_category"]["panel"]["n"] == 6
    assert rep["acceptance"]["panel_kappa_3cat"] == 1.0 and rep["acceptance"]["meets_target"] is True
    assert rep["three_category"]["qwen"]["kappa"] < 1.0
    assert set(rep["pairwise_judges"]) == {"llama_vs_deepseek", "llama_vs_qwen", "deepseek_vs_qwen"}
    assert rep["abstain_agreement"]["panel"]["n_abstain_human"] == 1
    assert rep["n_per_condition"]["A"]["n_items"] == 3 and rep["n_per_condition"]["A"]["n_human_labelled"] == 2
    md = ag.report_markdown("calibration", rep)
    assert "ABSTAIN excluded" in md and "ABSTAIN counted" in md and "unsafe pass" in md
    assert md.index("3-category kappa") < md.index("Supporting statistics")


def test_three_category_kappa_and_acceptance_decided_by_it():
    # binary (ABSTAIN excluded) agreement is perfect, but the panel calls the human ABSTAIN item NOT_HALLUCINATED
    human = {"i1": H, "i2": N, "i3": A, "i4": H, "i5": N}
    panel = {"i1": H, "i2": N, "i3": N, "i4": H, "i5": N}
    rep = ag.agreement_report(human, {"panel": panel}, {i: "A" for i in human}, n_resamples=0)
    assert rep["vs_human"]["panel"]["abstain_excluded"]["kappa"] == 1.0
    t = rep["three_category"]["panel"]
    assert t["raw_agreement"]["x"] == 4 and t["n"] == 5
    assert t["confusion_human_rows"][A][N] == 1
    from judge_binary import cohen_kappa, CATEGORIES
    expected = cohen_kappa([panel[i] for i in sorted(human)], [human[i] for i in sorted(human)], CATEGORIES)
    assert t["kappa"] == pytest.approx(expected, abs=1e-4) and t["kappa"] < 1.0
    assert rep["acceptance"]["panel_kappa_3cat"] == t["kappa"]


def test_raters_from_records_join_through_key():
    key_items = [{"item_id": "CAL-001", "run_id": "r1", "question_id": 1},
                 {"item_id": "CAL-002", "run_id": "r1", "question_id": 2}]
    jrec = {"panel_id": "panel-v1", "run_id": "r1", "question_id": 1, "judge_model": "Qwen/Qwen2.5-72B-Instruct",
            "call_failed": False, "parse_error": False, "answer_attempted": True, "hallucinated": True}
    prec = {"panel_id": "panel-v1", "run_id": "r1", "question_id": 2, "panel_label": "UNRESOLVED"}
    raters = ag.raters_from_records(key_items, [jrec], [prec])
    assert raters["qwen"] == {"CAL-001": H, "CAL-002": None}
    assert raters["panel"] == {"CAL-001": None, "CAL-002": None}
