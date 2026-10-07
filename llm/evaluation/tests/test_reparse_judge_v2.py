"""2026-10-07 follow-up: reparse_judge_v2.py re-parses EXISTING judge-v2
raw responses offline (no API calls), writing to a NEW file and never
touching the original."""

import json

from reparse_judge_v2 import reparse_file, reparse_record


def _raw_record(raw_response, run_label="A", condition="A", parse_error=True, call_failed=False):
    return {
        "run_id": "r1", "question_id": 1, "judge_id": "primary", "run_label": run_label,
        "condition": condition, "raw_response": raw_response, "parse_error": parse_error,
        "call_failed": call_failed, "finish_reason": "stop", "request_params": {"model": "x"},
    }


def _good_raw():
    return json.dumps({
        "answer_attempted": True, "completeness": "FULL",
        "claims": [{"claim": "x", "verdict": "SUPPORTED", "severity": "null",
                    "reference_conflict": "false", "evidence": "e"}],
        "reasoning": "ok", "label": "FAKTUAL",
    })


def test_reparse_record_fixes_string_null_severity_and_normalizes():
    record = _raw_record(_good_raw())
    out = reparse_record(record)
    assert out["parse_error"] is False
    assert out["claims"][0]["severity"] is None
    assert out["claims"][0]["reference_conflict"] is False
    assert set(out["normalized_fields"]) == {"severity", "reference_conflict"}


def test_reparse_record_never_mutates_input():
    record = _raw_record(_good_raw())
    original = json.loads(json.dumps(record))  # deep copy for comparison
    reparse_record(record)
    assert record == original


def test_reparse_record_adds_max_tokens_and_truncated():
    record = _raw_record(_good_raw())
    out = reparse_record(record)
    assert out["max_tokens"] is None  # no cap was ever recorded in request_params
    assert out["truncated"] is False  # finish_reason == "stop"


def test_reparse_record_leaves_call_failed_records_alone():
    record = _raw_record(None, call_failed=True)
    record["raw_response"] = None
    out = reparse_record(record)
    assert out["parse_error"] is True
    assert out["raw_response"] is None


def test_reparse_file_writes_new_file_never_touches_original(tmp_path):
    input_path = tmp_path / "jv2_test.jsonl"
    original_content = json.dumps(_raw_record(_good_raw())) + "\n"
    input_path.write_text(original_content)

    stats = reparse_file(input_path)

    assert input_path.read_text() == original_content  # untouched
    output_path = tmp_path / "jv2_test_reparsed.jsonl"
    assert output_path.exists()
    reparsed = json.loads(output_path.read_text().strip())
    assert reparsed["parse_error"] is False


def test_reparse_file_reports_before_after_counts_per_run_label_and_condition(tmp_path):
    input_path = tmp_path / "jv2_counts.jsonl"
    with open(input_path, "w") as f:
        f.write(json.dumps(_raw_record(_good_raw(), run_label="A", condition="A")) + "\n")
        f.write(json.dumps(_raw_record(_good_raw(), run_label="B-plain", condition="B")) + "\n")
        # A genuinely broken record that stays a parse error after reparse too.
        f.write(json.dumps(_raw_record("not json at all", run_label="A", condition="A")) + "\n")

    stats = reparse_file(input_path)

    assert stats["n_total"] == 3
    assert stats["n_before_errors"] == 3
    assert stats["n_after_errors"] == 1
    assert stats["before_errors_by_run_label"] == {"A": 2, "B-plain": 1}
    assert stats["after_errors_by_run_label"] == {"A": 1}
    assert stats["before_errors_by_condition"] == {"A": 2, "B": 1}
    assert stats["after_errors_by_condition"] == {"A": 1}
    assert stats["normalized_field_counts"] == {"severity": 2, "reference_conflict": 2}


def test_reparse_file_is_rerunnable_and_overwrites_its_own_output(tmp_path):
    input_path = tmp_path / "jv2_rerun.jsonl"
    input_path.write_text(json.dumps(_raw_record(_good_raw())) + "\n")
    reparse_file(input_path)
    stats2 = reparse_file(input_path)  # must not error or duplicate on a second pass
    assert stats2["n_total"] == 1
