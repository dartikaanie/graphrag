"""Smoke tests for judge_v2_leniency_review.py's pure helper functions
(no file I/O, no sklearn) -- the report-writing CLI itself (main()) is
exercised directly against the real pilot files when the report is
generated, not here."""

from judge_v2_leniency_review import build_transition_data, reference_conflict_stats


def _v1_record(run_id, question_id, judge_id, label, run_label="A", claims=None):
    return {"run_id": run_id, "question_id": question_id, "judge_id": judge_id, "label": label,
            "run_label": run_label, "parse_error": False, "claims": claims or []}


def _v2_record(run_id, question_id, judge_id, label, run_label="A", claims=None, parse_error=False):
    return {"run_id": run_id, "question_id": question_id, "judge_id": judge_id, "label": label,
            "run_label": run_label, "parse_error": parse_error, "claims": claims or []}


def test_build_transition_data_counts_matrix_cell():
    v1 = [_v1_record("r1", 1, "primary", "HALUSINASI_PENUH")]
    v2 = [_v2_record("r1", 1, "primary", "FAKTUAL")]
    result = build_transition_data(v1, v2, "primary")
    assert result["n_pairs"] == 1
    assert result["matrix"]["HALUSINASI_PENUH"]["FAKTUAL"] == 1


def test_build_transition_data_only_joins_matching_judge_id():
    v1 = [_v1_record("r1", 1, "primary", "FAKTUAL"), _v1_record("r1", 1, "secondary", "ABSTAIN")]
    v2 = [_v2_record("r1", 1, "primary", "FAKTUAL")]
    result = build_transition_data(v1, v2, "primary")
    assert result["n_pairs"] == 1


def test_build_transition_data_drilldown_captures_halusinasi_to_faktual():
    v1_claims = [
        {"claim": "a", "verdict": "SUPPORTED", "severity": None},
        {"claim": "b", "verdict": "CONTRADICTED", "severity": "CORE"},
    ]
    v2_claims = [{"claim": "a", "verdict": "SUPPORTED", "severity": None, "reference_conflict": True}]
    v1 = [_v1_record("r1", 1, "primary", "HALUSINASI_PENUH", claims=v1_claims)]
    v2 = [_v2_record("r1", 1, "primary", "FAKTUAL", claims=v2_claims)]
    result = build_transition_data(v1, v2, "primary")
    assert len(result["drilldown"]) == 1
    item = result["drilldown"][0]
    assert item["question_id"] == 1
    assert len(item["v1_error_claims"]) == 1
    assert item["v1_error_claims"][0]["verdict"] == "CONTRADICTED"
    assert item["reference_conflict_used"] is True


def test_build_transition_data_no_drilldown_when_v2_not_faktual():
    v1 = [_v1_record("r1", 1, "primary", "HALUSINASI_PENUH")]
    v2 = [_v2_record("r1", 1, "primary", "HALUSINASI_SEBAGIAN")]
    result = build_transition_data(v1, v2, "primary")
    assert result["drilldown"] == []


def test_build_transition_data_treats_parse_error_as_its_own_label():
    v1 = [_v1_record("r1", 1, "primary", "FAKTUAL")]
    v2 = [_v2_record("r1", 1, "primary", None, parse_error=True)]
    result = build_transition_data(v1, v2, "primary")
    assert result["matrix"]["FAKTUAL"]["PARSE_ERROR"] == 1


def test_reference_conflict_stats_counts_items_and_claims():
    records = [
        _v2_record("r1", 1, "primary", "FAKTUAL", run_label="A",
                   claims=[{"claim": "a", "reference_conflict": True}, {"claim": "b", "reference_conflict": False}]),
        _v2_record("r1", 2, "primary", "FAKTUAL", run_label="A",
                   claims=[{"claim": "c", "reference_conflict": False}]),
    ]
    stats = reference_conflict_stats(records, "primary")
    assert stats["A"]["n_items"] == 2
    assert stats["A"]["n_items_with_conflict"] == 1
    assert stats["A"]["pct_items_with_conflict"] == 50.0
    assert stats["A"]["n_claims_with_conflict"] == 1


def test_reference_conflict_stats_excludes_parse_errors():
    records = [_v2_record("r1", 1, "primary", None, run_label="A", parse_error=True)]
    stats = reference_conflict_stats(records, "primary")
    assert stats == {}
