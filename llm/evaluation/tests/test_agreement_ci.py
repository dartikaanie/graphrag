"""Amendment 11 agreement: kappa with bootstrap 95% CI, every rater pair, faithfulness, familiarity split --
synthetic data with known kappa."""

import pytest

import annotation_tool as at
import judge_binary as jb

H, N, A = jb.CATEGORY_HALLUCINATED, jb.CATEGORY_NOT_HALLUCINATED, jb.CATEGORY_ABSTAIN


def test_known_kappa_and_reproducible_ci():
    a, b = [H, H, N, N] * 10, [H, N, N, N] * 10          # kappa = (0.75 - 0.5) / 0.5 = 0.5
    r1, r2 = jb.agreement_report(a, b), jb.agreement_report(a, b)
    assert r1["kappa_3cat"] == 0.5 and r1["kappa_3cat_ci95"] == r2["kappa_3cat_ci95"]
    lo, hi = r1["kappa_3cat_ci95"]
    assert lo < 0.5 < hi and r1["bootstrap"]["n_resamples"] == 2000 and r1["bootstrap"]["seed"] == jb.BOOTSTRAP_SEED
    assert r1["kappa_binary_non_abstain"] == 0.5 and r1["kappa_binary_non_abstain_ci95"] == r1["kappa_3cat_ci95"]
    assert jb.agreement_report(a, b, n_resamples=0)["kappa_3cat_ci95"] is None
    assert jb.agreement_report(a, b, seed=1)["kappa_3cat_ci95"] != r1["kappa_3cat_ci95"]


def test_perfect_agreement_and_undefined_resamples():
    same = [H, N, A, N] * 5
    r = jb.agreement_report(same, same)
    assert r["kappa_3cat"] == 1.0 and r["kappa_3cat_ci95"] == [1.0, 1.0]
    assert jb.bootstrap_kappa_ci([H, H], [H, H], jb.CATEGORIES)["ci95"] is None   # kappa undefined everywhere


def test_kappa_report_on_another_scale():
    levels = jb.FAITHFULNESS_LEVELS
    a = ["all_supported", "not_supported", "partly_supported", "all_supported"] * 5
    r = jb.kappa_report(a, a, levels)
    assert r["kappa"] == 1.0 and r["confusion"]["all_supported"]["all_supported"] == 10


@pytest.mark.parametrize("ratio, level", [(1.0, "all_supported"), (0.0, "not_supported"), (0.5, "partly_supported"),
                                          (0.99, "partly_supported"), (None, None)])
def test_faithfulness_category(ratio, level):
    assert jb.faithfulness_category(ratio) == level


def _row(cat, familiar=True, context=True, faith="all_supported"):
    return {"category": cat, "domain_familiarity": "familiar" if familiar else "not_familiar",
            "has_context": context, "faithfulness_level": faith if context and cat != A else None}


def _judge(cat, faithfulness=1.0):
    attempted = cat != A
    claims = [{"verdict": "FABRICATED" if cat == H else "SUPPORTED"}] if attempted else []
    return {"answer_attempted": attempted, "claims": claims, "parse_error": False, "call_failed": False,
            "faithfulness": faithfulness if attempted else None}


def test_agreement_pairs_on_synthetic_raters():
    cats = [H, H, N, N] * 10
    researcher = {f"I{i:02d}": _row(c, familiar=i < 20, context=i % 2 == 0) for i, c in enumerate(cats)}
    primary = {i: _judge(r["category"]) for i, r in researcher.items()}                       # identical
    other = [H, N, N, N] * 10
    secondary = {f"I{i:02d}": _judge(c, faithfulness=0.5) for i, c in enumerate(other)}       # kappa 0.5 vs researcher
    secondary.pop("I39")                                                                      # one missing record
    annotator2 = {i: dict(r) for i, r in researcher.items() if int(i[1:]) < 20}
    intra = {"first": {i: r["category"] for i, r in researcher.items()},
             "second": {i: (N if i == "I00" else r["category"]) for i, r in researcher.items()}}
    rep = at.agreement_pairs(researcher, {"primary": primary, "secondary": secondary}, annotator2, intra)
    p = rep["pairs"]
    assert p["primary_vs_researcher"]["kappa_3cat"] == 1.0 and p["primary_vs_researcher"]["n"] == 40
    assert p["secondary_vs_researcher"]["n"] == 39 and p["secondary_vs_researcher"]["n_excluded"] == 1
    assert p["annotator2_vs_researcher"]["n"] == 20 and p["annotator2_vs_researcher"]["kappa_3cat"] == 1.0
    assert p["intra_researcher"]["n"] == 40 and p["intra_researcher"]["kappa_3cat"] < 1.0
    assert set(p) == {"primary_vs_researcher", "secondary_vs_researcher", "primary_vs_secondary",
                      "annotator2_vs_researcher", "intra_researcher"}
    f = rep["familiarity"]
    assert f["primary_vs_researcher|familiar"]["n"] == 20 and f["primary_vs_researcher|not_familiar"]["n"] == 20
    fa = rep["faithfulness"]
    assert fa["primary_vs_researcher"]["n"] == 20 and fa["primary_vs_researcher"]["observed_agreement"] == 1.0
    assert fa["secondary_vs_researcher"]["observed_agreement"] == 0.0          # judge "partly" vs human "all"
    md = at.agreement_markdown("calibration", "judge-v2.2", rep)
    assert "| primary_vs_researcher | 40 | 0 | 1.0 | [1.0, 1.0] |" in md and "## Faithfulness" in md


def test_judges_by_item_joins_one_version_through_the_key():
    key = {"items": [{"item_id": "CAL-001", "run_id": "r", "question_id": 1},
                     {"item_id": "CAL-002", "run_id": "r", "question_id": 2}]}
    recs = [{"prompt_version": "judge-v2.2", "judge_id": "primary", "run_id": "r", "question_id": 1},
            {"prompt_version": "judge-v2.1", "judge_id": "primary", "run_id": "r", "question_id": 2},
            {"prompt_version": "judge-v2.2", "judge_id": "secondary", "run_id": "x", "question_id": 1}]
    out = at.judges_by_item(key, recs, "judge-v2.2")
    assert list(out) == ["primary"] and list(out["primary"]) == ["CAL-001"]
