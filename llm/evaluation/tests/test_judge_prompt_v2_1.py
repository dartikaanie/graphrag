"""judge-v2.1 prompt + parser (docs/DECISION_C_SCORING.md, Amendment 4 and Addendum)."""

import hashlib
import json

import judge_prompt_v2
import judge_prompt_v2_1 as v21

# sha256 of judge-v2's SYSTEM_PROMPT as committed before judge-v2.1 existed -- judge-v2 must not change.
V2_SYSTEM_PROMPT_SHA256 = "d8beccfe84eaf7b09d2eb45dcb1779b738103b3bef2a9483382f457a001b6638"


def _raw(claims, attempted=True, label="FAKTUAL", completeness="FULL"):
    return json.dumps({"answer_attempted": attempted, "completeness": completeness, "claims": claims,
                       "reasoning": "r", "label": label})


def _claim(verdict, severity=None, **kw):
    return {"claim": "c", "verdict": verdict, "severity": severity, "reference_conflict": False, "evidence": "e", **kw}


def test_judge_v2_prompt_unchanged_and_versions_distinct():
    assert hashlib.sha256(judge_prompt_v2.SYSTEM_PROMPT.encode()).hexdigest() == V2_SYSTEM_PROMPT_SHA256
    assert judge_prompt_v2.PROMPT_VERSION == "judge-v2"
    assert v21.PROMPT_VERSION == "judge-v2.1"
    assert v21.SYSTEM_PROMPT != judge_prompt_v2.SYSTEM_PROMPT
    assert v21.REQUEST_TEMPERATURE == 0.0 and v21.REQUEST_MAX_TOKENS == 1500


def test_prompt_states_amendment_4_rules_and_addendum():
    p = v21.SYSTEM_PROMPT
    assert "Keep the qualifiers" in p                                              # rule 1
    assert 'the sources do not cover X' in p                                       # rule 2
    assert "For SUPPORTED and UNVERIFIABLE claims, severity is null" in p          # rule 3
    assert "a wrong version number" in p and "affects the main solution" in p      # rule 4
    assert "compiles, runs, or works is UNVERIFIABLE unless the reference contains the same code" in p  # rule 5
    assert "UNVERIFIABLE claims never change the label" in p                       # rule 6
    assert "When the reference addresses a claim, the reference decides" in p      # addendum
    assert "the reference states the claim or directly implies it" in p            # review change 2
    assert "Covering the same topic is not support" in p
    assert "never drop one, even if the list becomes longer than 8" in p           # review change 1
    assert "applies only to SUPPORTED and UNVERIFIABLE claims" in p
    assert '"reference_conflict": true' in p


def test_build_messages_blinds_citations_and_matches_v2_user_message():
    args = ("T", "B", "python", "REF", "Use x [SO-123] and y [SO-45].")
    m21, m2 = v21.build_messages(*args), judge_prompt_v2.build_messages(*args)
    assert m21[1] == m2[1] and "[SO-" not in m21[1]["content"]
    assert m21[0]["content"] == v21.SYSTEM_PROMPT


def test_severity_on_non_error_claims_is_nulled_and_counted():
    out = v21.parse_judgment_v2_1(_raw([_claim("SUPPORTED", "CORE"), _claim("UNVERIFIABLE", "MINOR"),
                                        _claim("SUPPORTED", "null")]))
    assert out["parse_error"] is False and out["label"] == "FAKTUAL"
    assert [c["severity"] for c in out["claims"]] == [None, None, None]
    assert out["normalized_fields"].count("severity_on_non_error") == 2
    assert out["normalized_fields"].count("severity") == 1


def test_error_claim_without_severity_is_kept_and_never_imputed():
    out = v21.parse_judgment_v2_1(_raw([_claim("CONTRADICTED", None)], label="HALUSINASI_SEBAGIAN"))
    assert out["parse_error"] is False
    assert out["label"] == "HALUSINASI_SEVERITY_UNSPECIFIED" and out["n_error_claims_without_severity"] == 1
    assert out["consistent"] is False
    # MINOR + unspecified: the unspecified one could be CORE -> still unspecified
    out = v21.parse_judgment_v2_1(_raw([_claim("FABRICATED", "MINOR"), _claim("CONTRADICTED", None)],
                                       label="HALUSINASI_SEBAGIAN"))
    assert out["label"] == "HALUSINASI_SEVERITY_UNSPECIFIED"
    # a CORE error settles it regardless
    out = v21.parse_judgment_v2_1(_raw([_claim("FABRICATED", "CORE"), _claim("CONTRADICTED", None)],
                                       label="HALUSINASI_PENUH"))
    assert out["label"] == "HALUSINASI_PENUH" and out["consistent"] is True


def test_derived_labels():
    d = v21.derive_label_v2_1
    assert d(False, [_claim("FABRICATED", "CORE")]) == "ABSTAIN"
    assert d(True, []) == "FAKTUAL"
    assert d(True, [_claim("UNVERIFIABLE"), _claim("SUPPORTED")]) == "FAKTUAL"       # rule 6
    assert d(True, [_claim("CONTRADICTED", "MINOR")]) == "HALUSINASI_SEBAGIAN"
    assert d(True, [_claim("CONTRADICTED", "MINOR"), _claim("FABRICATED", "CORE")]) == "HALUSINASI_PENUH"


def test_invalid_output_is_a_parse_error_with_parse_version():
    for raw in ("not json", json.dumps([1]), _raw([_claim("WRONG")]), _raw([_claim("CONTRADICTED", "HUGE")]),
                _raw([], label="HALUSINASI_SEVERITY_UNSPECIFIED"), _raw("x"),
                json.dumps({"answer_attempted": "true", "completeness": "FULL", "claims": [], "label": "FAKTUAL"})):
        out = v21.parse_judgment_v2_1(raw)
        assert out["parse_error"] is True and out["parse_version"] == "pv2.1-1"


def test_abstain_with_claims_is_flagged():
    out = v21.parse_judgment_v2_1(_raw([_claim("SUPPORTED")], attempted=False, label="ABSTAIN", completeness="NONE"))
    assert out["label"] == "ABSTAIN" and out["attempted_claims_conflict"] is True


def test_more_than_8_claims_are_all_kept_and_error_claims_never_dropped():
    claims = [_claim("SUPPORTED")] * 5 + [_claim("UNVERIFIABLE")] * 3 + [_claim("FABRICATED", "CORE")] * 4
    out = v21.parse_judgment_v2_1(_raw(claims, label="HALUSINASI_PENUH"))
    assert out["parse_error"] is False and len(out["claims"]) == 12
    assert out["n_claims"] == 12 and out["n_error_claims"] == 4
    assert out["non_error_claims_over_cap"] is False          # 8 non-error claims: at the limit
    assert out["label"] == "HALUSINASI_PENUH"

    over = [_claim("SUPPORTED")] * 9 + [_claim("CONTRADICTED", "MINOR")]
    out = v21.parse_judgment_v2_1(_raw(over, label="HALUSINASI_SEBAGIAN"))
    assert out["parse_error"] is False and len(out["claims"]) == 10   # nothing dropped
    assert out["non_error_claims_over_cap"] is True and out["label"] == "HALUSINASI_SEBAGIAN"
