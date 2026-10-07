"""Tests for judge_prompt_v2.py -- derive_label_v2's rule paths,
attempted_claims_conflict, parse_judgment_v2's strict field validation
(missing/invalid fields are parse errors, never silently defaulted)."""

import json

import pytest

from judge_prompt_v2 import derive_label_v2, parse_judgment_v2


def _claim(verdict, severity=None, reference_conflict=False):
    return {"claim": "x", "verdict": verdict, "severity": severity,
            "reference_conflict": reference_conflict, "evidence": "e"}


# ---------------------------------------------------------------------
# derive_label_v2
# ---------------------------------------------------------------------

def test_derive_label_v2_abstain_when_not_attempted_regardless_of_claims():
    assert derive_label_v2(False, []) == "ABSTAIN"
    assert derive_label_v2(False, [_claim("FABRICATED", "CORE")]) == "ABSTAIN"


def test_derive_label_v2_faktual_when_no_errors():
    assert derive_label_v2(True, []) == "FAKTUAL"
    assert derive_label_v2(True, [_claim("SUPPORTED"), _claim("UNVERIFIABLE")]) == "FAKTUAL"


def test_derive_label_v2_sebagian_for_minor_error_only():
    assert derive_label_v2(True, [_claim("CONTRADICTED", "MINOR")]) == "HALUSINASI_SEBAGIAN"
    assert derive_label_v2(True, [_claim("FABRICATED", "MINOR")]) == "HALUSINASI_SEBAGIAN"


def test_derive_label_v2_penuh_for_core_error():
    assert derive_label_v2(True, [_claim("CONTRADICTED", "CORE")]) == "HALUSINASI_PENUH"
    assert derive_label_v2(True, [_claim("FABRICATED", "CORE")]) == "HALUSINASI_PENUH"


def test_derive_label_v2_core_beats_minor_when_both_present():
    claims = [_claim("CONTRADICTED", "MINOR"), _claim("FABRICATED", "CORE")]
    assert derive_label_v2(True, claims) == "HALUSINASI_PENUH"


# ---------------------------------------------------------------------
# parse_judgment_v2 -- happy paths
# ---------------------------------------------------------------------

def _raw(answer_attempted=True, completeness="FULL", claims=None, label="FAKTUAL", reasoning="ok"):
    return json.dumps({
        "answer_attempted": answer_attempted, "completeness": completeness,
        "claims": claims or [], "reasoning": reasoning, "label": label,
    })


def test_parse_judgment_v2_official_label_equals_derived_equals_label():
    raw = _raw(answer_attempted=True, claims=[{"claim": "x", "verdict": "FABRICATED", "severity": "CORE",
                                                "reference_conflict": False, "evidence": "e"}], label="FAKTUAL")
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    # judge SELF-reported FAKTUAL, but a CORE fabrication means the
    # OFFICIAL/derived label must override it to HALUSINASI_PENUH.
    assert parsed["label"] == "HALUSINASI_PENUH"
    assert parsed["derived_label"] == "HALUSINASI_PENUH"
    assert parsed["label"] == parsed["derived_label"]
    assert parsed["judge_reported_label"] == "FAKTUAL"
    assert parsed["consistent"] is False


def test_parse_judgment_v2_consistent_when_judge_label_matches_derived():
    raw = _raw(answer_attempted=True, claims=[], label="FAKTUAL")
    parsed = parse_judgment_v2(raw)
    assert parsed["label"] == "FAKTUAL"
    assert parsed["consistent"] is True


def test_parse_judgment_v2_captures_reference_conflict_per_claim():
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                         "reference_conflict": True, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["claims"][0]["reference_conflict"] is True


def test_parse_judgment_v2_attempted_claims_conflict_flagged():
    """answer_attempted=false but claims listed anyway -- rule 0 still
    wins (label=ABSTAIN), but this inconsistency must be flagged."""
    raw = _raw(answer_attempted=False, completeness="NONE",
               claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                        "reference_conflict": False, "evidence": "e"}], label="ABSTAIN")
    parsed = parse_judgment_v2(raw)
    assert parsed["label"] == "ABSTAIN"
    assert parsed["attempted_claims_conflict"] is True


def test_parse_judgment_v2_no_conflict_when_not_attempted_and_no_claims():
    raw = _raw(answer_attempted=False, completeness="NONE", claims=[], label="ABSTAIN")
    parsed = parse_judgment_v2(raw)
    assert parsed["attempted_claims_conflict"] is False


def test_parse_judgment_v2_strips_markdown_fences():
    raw = "```json\n" + _raw() + "\n```"
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False


# ---------------------------------------------------------------------
# parse_judgment_v2 -- strict validation (missing/invalid -> parse_error)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("broken_raw", [
    "not json at all",
    json.dumps({"completeness": "FULL", "claims": [], "reasoning": "x", "label": "FAKTUAL"}),  # missing answer_attempted
    json.dumps({"answer_attempted": "true", "completeness": "FULL", "claims": [], "reasoning": "x", "label": "FAKTUAL"}),  # answer_attempted not bool
    json.dumps({"answer_attempted": True, "completeness": "WEIRD", "claims": [], "reasoning": "x", "label": "FAKTUAL"}),  # bad completeness
    json.dumps({"answer_attempted": True, "completeness": "FULL", "claims": "not-a-list", "reasoning": "x", "label": "FAKTUAL"}),  # claims not a list
    json.dumps({"answer_attempted": True, "completeness": "FULL",
                "claims": [{"claim": "x", "verdict": "MAYBE", "severity": None, "reference_conflict": False, "evidence": "e"}],
                "reasoning": "x", "label": "FAKTUAL"}),  # bad verdict
    json.dumps({"answer_attempted": True, "completeness": "FULL",
                "claims": [{"claim": "x", "verdict": "SUPPORTED", "severity": "HIGH", "reference_conflict": False, "evidence": "e"}],
                "reasoning": "x", "label": "FAKTUAL"}),  # bad severity
    json.dumps({"answer_attempted": True, "completeness": "FULL", "claims": [], "reasoning": "x", "label": "MAYBE"}),  # bad label
    json.dumps({"answer_attempted": True, "completeness": "FULL", "claims": [], "reasoning": "x"}),  # missing label
])
def test_parse_judgment_v2_missing_or_invalid_fields_are_parse_errors(broken_raw):
    parsed = parse_judgment_v2(broken_raw)
    assert parsed["parse_error"] is True
    assert parsed["parse_version"] is not None


# ---------------------------------------------------------------------
# parse_judgment_v2 -- narrow tolerance (2026-10-07 follow-up): the JSON
# STRING "null"/"none" for severity, and the JSON STRING "true"/"false"
# for reference_conflict, are normalized into their real types. Every
# other unexpected value for either field must STILL be a parse_error --
# this is a narrow allow-list, not general type coercion.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("severity_value", ["null", "NULL", "Null", "none", "NONE"])
def test_parse_judgment_v2_tolerates_string_null_severity(severity_value):
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": severity_value,
                         "reference_conflict": False, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    assert parsed["claims"][0]["severity"] is None
    assert parsed["normalized_fields"] == ["severity"]


def test_parse_judgment_v2_real_null_severity_not_counted_as_normalized():
    """Real JSON null for severity isn't a benign-variant fix -- it was
    already correct, so it must not inflate the normalized_fields count."""
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                         "reference_conflict": False, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    assert parsed["normalized_fields"] == []


@pytest.mark.parametrize("rc_value,expected", [("true", True), ("TRUE", True), ("false", False), ("FALSE", False)])
def test_parse_judgment_v2_tolerates_string_bool_reference_conflict(rc_value, expected):
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                         "reference_conflict": rc_value, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    assert parsed["claims"][0]["reference_conflict"] is expected
    assert parsed["normalized_fields"] == ["reference_conflict"]


def test_parse_judgment_v2_real_bool_reference_conflict_not_counted_as_normalized():
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                         "reference_conflict": False, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["normalized_fields"] == []


def test_parse_judgment_v2_missing_reference_conflict_defaults_false_not_normalized():
    """Key absent entirely (not an unexpected-value case) keeps the old
    default of False and is not reported as a normalization."""
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    assert parsed["claims"][0]["reference_conflict"] is False
    assert parsed["normalized_fields"] == []


@pytest.mark.parametrize("severity_value", ["HIGH", "low", "nulll", 1, 1.0, []])
def test_parse_judgment_v2_other_bad_severity_values_still_parse_error(severity_value):
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": severity_value,
                         "reference_conflict": False, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is True


@pytest.mark.parametrize("rc_value", ["yes", "no", "1", "0", 1, 0, None, [], {}])
def test_parse_judgment_v2_other_bad_reference_conflict_values_still_parse_error(rc_value):
    raw = _raw(claims=[{"claim": "x", "verdict": "SUPPORTED", "severity": None,
                         "reference_conflict": rc_value, "evidence": "e"}])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is True


def test_parse_judgment_v2_multiple_claims_accumulate_normalized_fields_per_occurrence():
    raw = _raw(claims=[
        {"claim": "a", "verdict": "SUPPORTED", "severity": "null", "reference_conflict": "false", "evidence": "e"},
        {"claim": "b", "verdict": "SUPPORTED", "severity": "none", "reference_conflict": False, "evidence": "e"},
    ])
    parsed = parse_judgment_v2(raw)
    assert parsed["parse_error"] is False
    assert parsed["normalized_fields"] == ["severity", "reference_conflict", "severity"]


def test_parse_judgment_v2_parse_version_present_on_success():
    import judge_prompt_v2

    parsed = parse_judgment_v2(_raw())
    assert parsed["parse_version"] == judge_prompt_v2.PARSE_VERSION
