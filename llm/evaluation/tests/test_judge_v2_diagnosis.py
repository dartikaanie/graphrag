"""Tests for judge_v2_diagnosis.py's pure helper functions (no file I/O).
2026-10-09 follow-up: why judge-v2 never produces HALUSINASI_SEBAGIAN on
the pilot batch, and quantifying specific leniency patterns."""

from judge_v2_diagnosis import (
    code_validity_claims,
    faktual_majority_unverifiable,
    items_with_minor_error_no_core,
    meta_statement_claims,
    supported_claims_with_severity,
    transition_totals,
    unverifiable_stats,
    verdict_severity_counts,
)


def _claim(verdict, severity=None, claim="x"):
    return {"claim": claim, "verdict": verdict, "severity": severity}


def _record(question_id, run_label, label, claims=None, parse_error=False, run_id="r1", judge_id="primary"):
    return {"question_id": question_id, "run_label": run_label, "label": label,
            "claims": claims or [], "parse_error": parse_error, "run_id": run_id, "judge_id": judge_id}


def test_verdict_severity_counts_excludes_parse_errors():
    records = [
        _record(1, "A", "FAKTUAL", claims=[_claim("SUPPORTED", None)]),
        _record(2, "A", None, claims=[_claim("CONTRADICTED", "CORE")], parse_error=True),
    ]
    counts = verdict_severity_counts(records)
    assert counts[("SUPPORTED", None)] == 1
    assert ("CONTRADICTED", "CORE") not in counts


def test_items_with_minor_error_no_core_detects_reachable_sebagian_condition():
    records = [_record(1, "A", "X", claims=[_claim("CONTRADICTED", "MINOR")])]
    out = items_with_minor_error_no_core(records)
    assert len(out) == 1
    assert out[0]["question_id"] == 1


def test_items_with_minor_error_no_core_excluded_when_core_present():
    records = [_record(1, "A", "X", claims=[_claim("CONTRADICTED", "MINOR"), _claim("FABRICATED", "CORE")])]
    assert items_with_minor_error_no_core(records) == []


def test_supported_claims_with_severity_flags_non_null_severity_on_supported():
    records = [_record(1, "A", "FAKTUAL", claims=[_claim("SUPPORTED", "CORE"), _claim("SUPPORTED", None)])]
    out = supported_claims_with_severity(records)
    assert len(out) == 1
    assert out[0]["severity"] == "CORE"


def test_meta_statement_claims_matches_known_phrasing():
    records = [_record(1, "A", "FAKTUAL", claims=[
        _claim("SUPPORTED", claim="The retrieved sources do not cover this edge case"),
        _claim("SUPPORTED", claim="X works as described"),
    ])]
    out = meta_statement_claims(records)
    assert len(out) == 1
    assert "retrieved sources" in out[0]["claim"]


def test_unverifiable_stats_counts_total_and_core():
    records = [_record(1, "A", "FAKTUAL", claims=[
        _claim("UNVERIFIABLE", None), _claim("UNVERIFIABLE", "CORE"), _claim("SUPPORTED", None),
    ])]
    stats = unverifiable_stats(records)
    assert stats == {"total": 2, "core": 1}


def test_faktual_majority_unverifiable_threshold():
    records = [
        _record(1, "A", "FAKTUAL", claims=[_claim("UNVERIFIABLE"), _claim("UNVERIFIABLE"), _claim("SUPPORTED")]),
        _record(2, "A", "FAKTUAL", claims=[_claim("UNVERIFIABLE"), _claim("SUPPORTED"), _claim("SUPPORTED")]),
        _record(3, "A", "HALUSINASI_PENUH", claims=[_claim("UNVERIFIABLE"), _claim("UNVERIFIABLE")]),
    ]
    out = faktual_majority_unverifiable(records)
    assert [r["question_id"] for r in out] == [1]


def test_code_validity_claims_matches_only_supported():
    records = [_record(1, "A", "FAKTUAL", claims=[
        _claim("SUPPORTED", claim="This is valid Rust code"),
        _claim("CONTRADICTED", claim="This is valid Rust code"),
    ])]
    out = code_validity_claims(records)
    assert len(out) == 1


def test_transition_totals_aggregates_by_v2_outcome():
    v1 = [
        _record(1, "A", "HALUSINASI_PENUH", run_id="r1"),
        _record(2, "A", "HALUSINASI_SEBAGIAN", run_id="r2"),
        _record(3, "A", "FAKTUAL", run_id="r3"),
        _record(4, "A", "HALUSINASI_PENUH", run_id="r4"),
    ]
    v2 = [
        _record(1, "A", "FAKTUAL", run_id="r1"),
        _record(2, "A", "HALUSINASI_PENUH", run_id="r2"),
        _record(3, "A", "FAKTUAL", run_id="r3"),
        _record(4, "A", "ABSTAIN", run_id="r4"),
    ]
    totals = transition_totals(v1, v2, "primary")
    assert totals["n_v1_halusinasi"] == 3
    assert totals["to_faktual"] == 1
    assert totals["to_penuh"] == 1
    assert totals["to_abstain"] == 1


def test_transition_totals_only_joins_matching_judge_id():
    v1 = [_record(1, "A", "HALUSINASI_PENUH", run_id="r1", judge_id="primary"),
          _record(1, "A", "FAKTUAL", run_id="r1", judge_id="secondary")]
    v2 = [_record(1, "A", "FAKTUAL", run_id="r1", judge_id="primary")]
    totals = transition_totals(v1, v2, "primary")
    assert totals["n_v1_halusinasi"] == 1
