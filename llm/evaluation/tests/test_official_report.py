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


def test_three_numbers_both_versions():
    t = orp.three_numbers(["HALLUCINATED", "NOT_HALLUCINATED", "NOT_HALLUCINATED", "ABSTAIN", None])
    assert t["rate_excl_abstain"] == round(1 / 3, 4)            # (a) primary: ABSTAIN leaves the denominator
    assert t["rate_abstain_as_failed"] == 0.5                    # (b) (1 hallucinated + 1 ABSTAIN) / 4 usable
    assert t["abstain_rate"] == 0.25 and t["n_missing"] == 1 and "rate_all_answers" not in t


def test_severity_split():
    sp = orp.severity_split(["PENUH", "PENUH", "SEBAGIAN", "UNSPECIFIED"])
    assert (sp["full"], sp["partial"], sp["unspecified"], sp["n_hallucinated"]) == (2, 1, 1, 4)
    assert sp["full_share"] == 0.5 and orp.severity_split([])["full_share"] is None


def _rec(claims, attempted=True):
    return {"answer_attempted": attempted, "claims": claims}


def test_claim_stats_denominators():
    cl = lambda v, sup, cs: {"verdict": v, "supported_by_context": sup, "cited_item_supports": cs}  # noqa: E731
    recs = [_rec([cl("SUPPORTED", "YES", "YES"), cl("UNVERIFIABLE", "NO", None), cl("SUPPORTED", None, None)]),
            _rec([cl("CONTRADICTED", "NO", "NO"), cl("SUPPORTED", "YES", "YES")]),
            _rec([], attempted=False),                                   # ABSTAIN: no claims, left out
            {"call_failed": True}]                                       # unusable: left out
    s = orp.claim_stats(recs)
    assert (s["n_claims"], s["n_unverifiable"]) == (5, 1) and s["unverifiable_share"] == 0.2
    f, c = s["faithfulness"], s["citation_support"]
    assert (f["n_claims_rated"], f["micro"]) == (4, 0.5)                # YES 2 of 4 rated; the null claim is left out
    assert f["macro"] == round((0.5 + 0.5) / 2, 4) and f["n_answers"] == 2
    assert (c["n_claims_cited"], c["micro"]) == (3, round(2 / 3, 4))   # cited claims: YES, NO, YES


def test_paired_counts_primary_and_abstain_as_failed():
    H, N, A = "HALLUCINATED", "NOT_HALLUCINATED", "ABSTAIN"
    c = {"1": N, "2": N, "3": H, "4": H, "5": A, "6": N, "7": None}
    x = {"1": H, "2": N, "3": N, "4": H, "5": H, "6": A, "7": N}
    p = orp.paired_counts(c, x)
    assert (p["c_better"], p["equal"], p["c_worse"]) == (1, 2, 1) and p["n_excluded"] == 3
    assert p["cells"] == {"only_c": 1, "only_other": 1, "both": 1, "neither": 1}
    f = orp.paired_counts(c, x, abstain_as_failed=True)  # ABSTAIN = failed (hallucinated); the unusable pair is left out
    assert (f["c_better"], f["equal"], f["c_worse"]) == (2, 3, 1) and f["n_excluded"] == 1


def _run(label, panel, ds, pq=None):
    rates = {"rate_excl_abstain": panel, "rate_abstain_as_failed": panel}
    return {"label": label, "per_question_panel": pq or {}, "results": {"panel": rates, "judges": {
        "deepseek": {"rate_excl_abstain": ds}, "llama": {"rate_excl_abstain": panel},
        "qwen3-235b": {"rate_excl_abstain": panel}}}}


def test_rule_e_flags_for_the_three_official_comparisons():
    flags = {c["name"]: c.get("flag") for c in orp.comparisons(
        [_run("R5", 0.1, 0.3), _run("R1", 0.2, 0.4), _run("R3", 0.05, 0.35), _run("R9", 0.1, 0.2)])}
    assert flags == {"C vs A": "consistent",       # both lower
                     "C vs B": "inconclusive",     # panel higher, DeepSeek lower
                     "C vs D": "inconclusive"}     # panel difference exactly 0


def test_legacy_comparisons_are_skipped_quietly_without_legacy_runs():
    runs = [_run("R5", 0.1, 0.3), _run("R1", 0.2, 0.4)]
    names = [c["name"] for c in orp.comparisons(runs)]
    assert names == ["C vs A", "C vs B", "C vs D"]                     # B and D missing -> listed as not available
    assert [c["available"] for c in orp.comparisons(runs)] == [True, False, False]
    with_legacy = orp.comparisons(runs + [_run("R7", 0.2, 0.3)])
    assert [c["comparison"] for c in with_legacy][-1] == "R5 vs R7"   # appears only when both legacy runs exist


def test_mock_report_and_item_drilldown_are_labelled():
    store = orp.MockStore(n=6)
    rep = orp.build(store)
    assert rep["meta"]["mock"] is True and [r["label"] for r in rep["runs"]] == ["R1", "R3", "R5", "R9"]
    md = orp.markdown(rep)
    assert "MOCK DATA" in md and "C vs A" in md and "all 3 judges same label" in md
    run = next(r for r in rep["runs"] if r["label"] == "R5")
    assert run["results"]["faithfulness_citation"] is not None
    assert next(r for r in rep["runs"] if r["label"] == "R1")["results"]["faithfulness_citation"] is None  # A: no context
    assert 0 <= run["reliability"]["all_three_judges_same"]["same_label"] <= 1
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
