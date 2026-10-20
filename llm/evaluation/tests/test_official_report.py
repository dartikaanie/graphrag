"""official_report.py: agreement statistics, three numbers, rule (e), mock walkthrough. Offline."""

import random

import official_report as orp
from judge_interjudge_analysis import fleiss_kappa as reference_fleiss


def test_krippendorff_alpha_matches_the_published_nominal_example():
    # Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", nominal example: alpha = 0.743
    a = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    b = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    c = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    d = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    rows = [[str(x) if x is not None else None for x in unit] for unit in zip(a, b, c, d)]
    assert round(orp.krippendorff_alpha_nominal(rows), 3) == 0.743


def test_fleiss_matches_the_existing_implementation():
    rng = random.Random(1)
    cats = orp.CATS
    rows = [[rng.choice(cats) for _ in range(3)] for _ in range(80)]
    assert orp.fleiss_kappa(rows) == round(reference_fleiss(rows, cats), 4)


def test_three_numbers():
    t = orp.three_numbers(["HALLUCINATED", "NOT_HALLUCINATED", "NOT_HALLUCINATED", "ABSTAIN", None])
    assert t["rate_excl_abstain"] == round(1 / 3, 4) and t["abstain_rate"] == 0.25 and t["rate_all_answers"] == 0.25
    assert t["n_missing"] == 1


def test_rule_e_flags():
    def run(label, panel, ds):
        rates = {"rate_excl_abstain": panel}
        return {"label": label, "results": {"panel": rates, "judges": {
            "deepseek": {"rate_excl_abstain": ds}, "llama": {"rate_excl_abstain": panel},
            "qwen3-235b": {"rate_excl_abstain": panel}}}}
    flags = {c["comparison"]: c.get("flag") for c in orp.comparisons(
        [run("R5", 0.1, 0.3), run("R1", 0.2, 0.4), run("R3", 0.05, 0.35), run("R9", 0.1, 0.2), run("R7", 0.2, 0.3)])}
    assert flags["R5 vs R1"] == "consistent"      # both lower
    assert flags["R5 vs R3"] == "inconclusive"    # panel higher, DeepSeek lower
    assert flags["R5 vs R9"] == "inconclusive"    # panel difference exactly 0
    assert flags["R5 vs R7"] == "inconclusive"    # DeepSeek difference exactly 0
    assert flags["R4 vs R2"] is None              # not available


def test_mock_report_and_item_drilldown_are_labelled():
    store = orp.MockStore(n=6)
    rep = orp.build(store)
    assert rep["meta"]["mock"] is True and len(rep["runs"]) == 5
    assert "MOCK DATA" in orp.markdown(rep)
    item = orp.item_detail(store, "MOCK-R5", 1000)
    assert item["mock"] and set(item["judges"]) == set(orp.JUDGES) and "panel_label" in item["panel"]


def test_nf3_is_p95_of_end_to_end_latency_for_condition_c():
    answers = [{"end_to_end_latency_sec": float(x)} for x in range(1, 21)]  # 1..20 s
    r = orp.nf3(answers, 99.0, "C")
    assert not r["fallback"] and r["value_sec"] == 19.0 and r["meets"] is False and r["nf3_applies"]
    assert orp.nf3([{"end_to_end_latency_sec": 2.0}] * 20, None, "C")["meets"] is True
    assert orp.nf3(answers, None, "B")["meets"] is None  # NF3 is a Condition C target


def test_nf3_falls_back_to_run_duration_for_records_without_r3_timing():
    r = orp.nf3([{"retrieval_latency_sec": 1.0}, {"end_to_end_latency_sec": 3.0}], 7.5, "C")
    assert r["fallback"] and r["value_sec"] == 7.5 and r["meets"] is None and r["method"].startswith("FALLBACK")
