"""Tests for judge_v1_lookup_service.py -- reads a fixture
judge_v1_run_history.jsonl (never the real one) via monkeypatched paths.
Covers the 3 explicit filtering rules from the user's follow-up:
invalid jobs hidden by default, missing output files shown with a
warning flag (never hidden), non-canonical run_labels shown under
"other configs" (never hidden).
"""

import json

from app.services import invalid_judge_runs_service, judge_v1_lookup_service as svc


def _write_job(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"run_started_at": "2026-10-05T00:00:00+00:00", "run_files": [], "judge": "primary",
            "out": "results/out.jsonl", "judged": 0}
    base.update(fields)
    with open(path, "a") as f:
        f.write(json.dumps(base) + "\n")


def test_output_file_missing_is_flagged_not_hidden(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "does_not_exist.jsonl"
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_judge_v1_jobs()
    assert len(jobs) == 1
    assert jobs[0]["output_file_missing"] is True


def test_output_file_present_is_not_flagged(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text('{"question_id": 1}\n')
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["output_file_missing"] is False


def test_non_canonical_run_label_shown_under_other_configs_not_hidden(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=["some_run_file.jsonl"])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)
    # _run_labels_for_job resolves via resolve_run_metadata -- make it
    # return a pre-grounding-era label, which is NOT in FACTORIAL_RUN_LABELS.
    monkeypatch.setattr(svc, "_run_labels_for_job", lambda record: ["C-uniform"])

    jobs = svc.load_judge_v1_jobs()
    assert len(jobs) == 1  # shown, not hidden
    assert jobs[0]["is_other_config"] is True


def test_canonical_run_label_is_not_flagged_as_other_config(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=["some_run_file.jsonl"])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)
    monkeypatch.setattr(svc, "_run_labels_for_job", lambda record: ["B-grounded"])

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["is_other_config"] is False


def test_invalid_job_hidden_by_default_but_shown_with_show_invalid(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=[], judge="primary")
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    invalid_path = tmp_path / "invalid_judge_runs.jsonl"
    monkeypatch.setattr(invalid_judge_runs_service, "INVALID_JUDGE_RUNS_PATH", invalid_path)

    all_jobs = svc.load_judge_v1_jobs(show_invalid=True)
    job_id = all_jobs[0]["job_id"]
    invalid_judge_runs_service.mark_invalid(job_id, "pre-dashboard smoke test")

    assert svc.load_judge_v1_jobs() == []
    assert len(svc.load_judge_v1_jobs(show_invalid=True)) == 1


def test_get_job_returns_none_for_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", tmp_path / "does_not_exist.jsonl")
    assert svc.get_job("JV1-nonexistent") is None


def test_batch_id_resolved_from_run_files(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=["some_run_file.jsonl"])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)
    monkeypatch.setattr(svc, "_run_labels_for_job", lambda record: ["B-grounded"])
    monkeypatch.setattr(svc, "_batch_id_for_job", lambda record: "BATCH-123")

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["batch_id"] == "BATCH-123"


def test_batch_id_is_none_when_unresolvable(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=["missing.jsonl"])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)
    monkeypatch.setattr(svc, "_run_labels_for_job", lambda record: [None])

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["batch_id"] is None


def test_served_model_mismatch_count_reads_output_file(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text(
        json.dumps({"question_id": 1, "served_model_mismatch": True}) + "\n"
        + json.dumps({"question_id": 2, "served_model_mismatch": False}) + "\n"
        + json.dumps({"question_id": 3, "served_model_mismatch": True}) + "\n"
    )
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["served_model_mismatch_count"] == 2


def test_served_model_mismatch_count_zero_when_file_missing(tmp_path, monkeypatch):
    history_path = tmp_path / "judge_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "does_not_exist.jsonl"
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "JUDGE_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_judge_v1_jobs()
    assert jobs[0]["served_model_mismatch_count"] == 0
