"""Amendment 14 screening: eligibility criteria, measured-input projection, runtime budget guard. Offline."""

import judge_panel as jp
import judge_screening as js
from judge_panel_config import SCREENING_BY_ID


def _rec(label="FAKTUAL", attempted=True, claims=None, **kw):
    r = {"parse_ok": True, "call_failed": False, "parse_error": False, "answer_attempted": attempted, "label": label,
         "derived_label": label, "claims": claims or [], "reasoning_present": False, "raw_response": "{}",
         "attempts": [{"error": None}], "served_model_mismatch": False}
    r.update(kw)
    return r


def _halluc(evidence=""):
    return _rec("HALUSINASI_PENUH", claims=[{"verdict": "CONTRADICTED", "severity": "CORE", "evidence": evidence}])


def test_quote_check_flags_candidate_text_presented_as_reference():
    ref, cand = "Use datetime.fromisoformat for this.", "You should call parse_date() [SO-1] on the string."
    rec = {"claims": [{"evidence": '[R] "use datetime.fromisoformat"'},
                      {"evidence": '[R] “call parse_date() on the string”'},
                      {"evidence": '[R] "something invented"'}]}
    q = js.quote_check(rec, ref, cand)
    assert q["n_quotes"] == 3 and q["violations"] == ["call parse_date() on the string"]
    assert q["in_neither"] == ["something invented"]


def test_criteria_all_three():
    items = [f"I{i}" for i in range(60)]
    texts = {i: ("reference text", "candidate text") for i in items}
    recs = {i: _rec() for i in items}
    recs["I0"], recs["I1"], recs["I2"] = _halluc(), _halluc(), _halluc()
    recs["I3"] = _rec("ABSTAIN", attempted=False)
    c = js.criteria(recs, texts)
    assert c["c1_not_degenerate"]["hallucinated"] == 3 and c["c1_not_degenerate"]["non_abstain"] == 59
    assert c["eligible"]
    recs["I2"] = _rec()                                              # 2/59 < 5%
    assert not js.criteria(recs, texts)["c1_not_degenerate"]["pass"]
    recs["I2"] = _halluc('[R] "candidate text"')                     # quote taken from the candidate
    assert js.criteria(recs, texts)["c2_no_candidate_as_reference_quote"]["violations"] == 1
    recs["I2"] = _halluc()
    recs["I5"] = _rec(reasoning_present=True)
    assert not js.criteria(recs, texts)["c3_clean_parse_no_reasoning"]["pass"]
    recs["I5"] = None                                                # missing record -> 59/60 usable
    assert not js.criteria(recs, texts)["c3_clean_parse_no_reasoning"]["pass"]
    recs["I5"] = _rec(served_model_mismatch=True)
    c = js.criteria(recs, texts)
    assert not c["valid"] and not c["eligible"]


def test_projection_uses_measured_prompt_tokens_for_identical_messages():
    judge = SCREENING_BY_ID["qwen3-235b"]
    msgs = [{"role": "user", "content": "x" * 3000}]
    sha = jp._messages_sha256(msgs)
    basis = [{"messages_sha256": sha, "prompt_tokens": 400}, {"messages_sha256": sha, "prompt_tokens": 500}]
    p = jp.project_cost([({}, judge, msgs)], [], 2000, (judge,), basis)
    assert p["per_judge"]["qwen3-235b"]["input_tokens"] == 500       # largest measured count
    assert p["input_token_basis"].startswith("1/1")
    p = jp.project_cost([({}, judge, msgs)], [], 2000, (judge,), [])
    assert p["per_judge"]["qwen3-235b"]["input_tokens"] == 1000      # 3000 chars / 3.0


def test_budget_guard_refuses_a_call_that_could_exceed_the_cap():
    judge = SCREENING_BY_ID["qwen3-235b"]
    g = jp.BudgetGuard(0.006, 2000, lambda m: 5000)  # room for one worst-case call (~$0.00465)
    worst = g.worst_case(judge, [])
    assert round(worst, 6) == round(3 * (5000 / 1e6 * 0.09 + 2000 / 1e6 * 0.55), 6)
    held = g.reserve(judge, [])
    assert held == worst and g.reserve(judge, []) is None and g.n_skipped == 1
    g.settle(held, 0.002)
    assert g.spent == 0.002 and g.reserved == 0
    assert jp.BudgetGuard(None, 2000, lambda m: 0).reserve(judge, []) == 0.0


def test_screening_requires_calibration_set_and_rule(tmp_path):
    rc = jp.main(["--run-files", "x.jsonl", "--out", str(tmp_path / "o.jsonl"), "--max-cost-usd", "0.1",
                  "--screen-judge", "qwen3-235b"])
    assert rc == 1


def test_qwen3_32b_sends_both_thinking_switches():
    assert SCREENING_BY_ID["qwen3-32b"].extra_body == {"reasoning_effort": "none",
                                                      "chat_template_kwargs": {"enable_thinking": False}}


def test_fairness_report_uses_the_same_quote_check_per_judge():
    texts = {"I1": ("reference sentence", "candidate sentence"), "I2": ("ref", "cand")}
    recs = {"a": {"I1": _halluc('[R] "candidate sentence"'), "I2": _rec()},
            "b": {"I1": _halluc('[R] "reference sentence"'), "I2": _rec()}}
    for r in (*recs["a"].values(), *recs["b"].values()):
        r["judge_model"] = "m"
    rep = js.fairness_report(recs, texts, "x.jsonl")
    assert rep["judges"]["a"]["violations"] == 1 and rep["judges"]["a"]["violating_items"] == {
        "I1": ["candidate sentence"]}
    assert rep["judges"]["b"]["violations"] == 0 and rep["judges"]["b"]["n_quotes"] == 1
