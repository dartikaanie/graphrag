"""judge-v2.3 (iteration 2): prompt changes #1-#5 on top of the unchanged v2.2 prompt, JSON mode request, and the
supported_by_context "N/A" -> null normalization. Offline."""

import json

import pytest

import judge_panel as jp
import judge_prompt_v2_2 as v22
import judge_prompt_v2_3 as v23
from judge_panel_config import JUDGES_BY_ID


def _claim(sbc="YES", verdict="SUPPORTED", severity=None, cited=None, cis=None):
    return {"claim": "c", "verdict": verdict, "severity": severity, "reference_conflict": False,
            "supported_by_context": sbc, "cited_items": cited or [], "cited_item_supports": cis, "evidence": "e"}


def _raw(claims, attempted=True):
    return json.dumps({"answer_attempted": attempted, "completeness": "FULL",
                       "approach_match": "SAME" if attempted else None, "claims": claims, "reasoning": "r",
                       "label": "FAKTUAL"})


# --- prompt -------------------------------------------------------------------

def test_versions_and_request_params():
    assert v23.PROMPT_VERSION == "judge-v2.3" and v23.PARSE_VERSION == "pv2.3-1"
    assert v23.REQUEST_TEMPERATURE == 0.0 and v23.REQUEST_MAX_TOKENS == 2000
    assert v23.REQUEST_RESPONSE_FORMAT == {"type": "json_object"}
    assert v23.BLINDING_VERSION == v22.BLINDING_VERSION == "blind-v3"
    assert v22.PROMPT_VERSION == "judge-v2.2" and v22.PARSE_VERSION == "pv2.2-1"


@pytest.mark.parametrize("needle", [
    "Explicit conflict with the reference",                 # #1
    "A different choice is not a contradiction",            # #2
    "List each identifier the answer relies on",            # #3
    "chosen in this order: (1) identifiers",                # #4
    "must quote the reference sentence it rests on",        # #5
    "(k) A different question about the fictional tool", "(l) A different question about fictional libraries",
    "(m) Same question as (a)-(j)",
])
def test_v23_prompt_has_each_change_and_v22_does_not(needle):
    assert v23.SYSTEM_PROMPT.count(needle) == 1
    assert needle not in v22.SYSTEM_PROMPT


def test_change_1_never_without_change_2():
    i1 = v23.SYSTEM_PROMPT.index("Explicit conflict with the reference")
    i2 = v23.SYSTEM_PROMPT.index("A different choice is not a contradiction")
    assert 0 < i2 - i1 < 1000


def test_v22_text_is_kept_apart_from_the_replaced_spots():
    # every v2.2 line is still present in v2.3 except the three lines that were replaced
    replaced = ("prioritizing the ones most important to the solution", '"evidence": "<one short sentence>"')
    for line in v22.SYSTEM_PROMPT.splitlines():
        if line.strip() and not any(r in line for r in replaced):
            assert line in v23.SYSTEM_PROMPT, line[:80]


def test_messages_use_v22_template_with_v23_system_prompt():
    prepared = v23.prepare_judge_input("Use foo().", [], False)
    m22 = v22.build_messages("t", "b", "x", "ref", "Use foo().", prepared=prepared)
    m23 = v23.build_messages("t", "b", "x", "ref", "Use foo().", prepared=prepared)
    assert m23[1] == m22[1] and m23[0]["content"] == v23.SYSTEM_PROMPT != m22[0]["content"]


# --- parser: N/A -> null --------------------------------------------------------

def test_na_with_context_is_normalised_to_null_and_recorded():
    raw = _raw([_claim("N/A"), _claim("YES"), _claim("n/a", verdict="CONTRADICTED", severity="CORE"), _claim("NO")])
    assert v22.parse_judgment_v2_2(raw, True)["parse_error"] is True      # the iteration-1 failure
    r = v23.parse_judgment_v2_3(raw, True)
    assert r["parse_error"] is False and r["parse_version"] == "pv2.3-1" and r["prompt_version"] == "judge-v2.3"
    assert [c["supported_by_context"] for c in r["claims"]] == [None, "YES", None, "NO"]
    assert r["normalized_fields"].count("supported_by_context_na_with_context") == 2
    assert r["faithfulness"] == 0.5                     # YES / (YES + NO); nulls not counted
    assert r["label"] == "HALUSINASI_PENUH"             # verdicts untouched


def test_na_everywhere_gives_null_faithfulness():
    r = v23.parse_judgment_v2_3(_raw([_claim("N/A"), _claim("N/A")]), True)
    assert r["faithfulness"] is None and r["label"] == "FAKTUAL"


def test_without_context_behaviour_is_v22():
    raw = _raw([_claim("N/A"), _claim("YES")])
    r22, r23 = v22.parse_judgment_v2_2(raw, False), v23.parse_judgment_v2_3(raw, False)
    assert r23["claims"] == r22["claims"] and r23["normalized_fields"] == r22["normalized_fields"]
    assert "supported_by_context_na_with_context" not in r23["normalized_fields"]


def test_other_failures_are_unchanged():
    assert v23.parse_judgment_v2_3(_raw([_claim("MAYBE")]), True)["parse_error"] is True
    assert v23.parse_judgment_v2_3(_raw([_claim(None)]), True)["parse_error"] is True
    assert v23.parse_judgment_v2_3("not json", True)["parse_error"] is True
    assert v23.parse_judgment_v2_3('{"a": "x "y" z"}', True)["parse_error"] is True   # unescaped quotes: still fails


def test_valid_v22_output_parses_identically():
    raw = _raw([_claim("YES"), _claim("NO", verdict="FABRICATED", severity="MINOR")])
    r22, r23 = v22.parse_judgment_v2_2(raw, True), v23.parse_judgment_v2_3(raw, True)
    for k in ("label", "claims", "normalized_fields", "faithfulness", "answer_attempted", "approach_match"):
        assert r23[k] == r22[k], k


# --- runner: JSON mode for all three judges ---------------------------------------

from test_judge_panel import FakeClient, ITEM, MESSAGES, NOSLEEP  # noqa: E402

GOOD_CTX = _raw([_claim("N/A", verdict="CONTRADICTED", severity="CORE")])


@pytest.mark.parametrize("jid", ["llama", "deepseek", "qwen"])
def test_v23_requests_json_mode_for_every_judge(jid):
    c = FakeClient([GOOD_CTX], model=JUDGES_BY_ID[jid].expected_served_model)
    prepared = {"has_context": True, "n_context_items": 5, "citation_map": {}, "offered_items": []}
    r = jp.judge_record(ITEM, JUDGES_BY_ID[jid], MESSAGES, prepared, c, sleep=NOSLEEP,
                        spec=jp.judge_version_spec("judge-v2.3"))
    call = c.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0 and call["max_tokens"] == 2000
    assert ("extra_body" in call) == (jid == "deepseek")
    assert r["prompt_version"] == "judge-v2.3" and r["parse_version"] == "pv2.3-1"
    assert r["request_response_format"] == {"type": "json_object"}
    assert r["parse_ok"] and r["normalized_fields"] == ["supported_by_context_na_with_context"]


def test_v22_requests_stay_without_json_mode():
    c = FakeClient([GOOD_CTX])
    jp.judge_record(ITEM, JUDGES_BY_ID["qwen"], MESSAGES, {"has_context": False, "n_context_items": 0,
                    "citation_map": {}, "offered_items": []}, c, sleep=NOSLEEP)
    assert "response_format" not in c.calls[0]


def test_spec_selection_and_cache_key_by_version():
    assert jp.judge_version_spec("judge-v2.2").response_format is None
    assert jp.judge_version_spec("judge-v2.3").prompt_version == "judge-v2.3"
    with pytest.raises(ValueError):
        jp.judge_version_spec("judge-v2.4")
    items = [{**ITEM, "title": "t", "tags": "", "ground_truth_answer_html": "r", "candidate": "c",
              "retrieved_context": []}]
    done = {(ITEM["item_id"], j.model, "judge-v2.2") for j in jp.PANEL_JUDGES}   # v2.2 results never count for v2.3
    tasks, _, n_cached = jp.plan_tasks(items, {1: "b"}, done, jp.judge_version_spec("judge-v2.3"))
    assert n_cached == 0 and len(tasks) == 3
    assert tasks[0][2][0]["content"] == v23.SYSTEM_PROMPT


def test_example_l_uses_numbers_unrelated_to_calibration_items():
    # iteration-2 fix: (l) first reused 1.0.1 / 11.0.1 / 1.1.0 from calibration items CAL-016/017/058
    ex = v23.SYSTEM_PROMPT[v23.SYSTEM_PROMPT.index("(l) A different"):v23.SYSTEM_PROMPT.index("(m) Same")]
    assert "2.4.0" in ex and "7.3.2" in ex and "2.6.0" in ex
    for n in ("1.0.1", "11.0.1", "1.1.0"):
        assert n not in v23.SYSTEM_PROMPT
