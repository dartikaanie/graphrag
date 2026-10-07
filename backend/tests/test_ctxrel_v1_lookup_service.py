"""Tests for ctxrel_v1_lookup_service.py -- mirrors
test_judge_v1_lookup_service.py's coverage for the SEPARATE context-
relevance-v1 job log, plus the inferred-batch-id fallback that fixes
the "context relevance table not shown" bug (a job's run_files
resolving to NO recorded batch_id must still resolve to the inferred
one, not None)."""

import json

from app.services import ctxrel_v1_lookup_service as svc
from app.services import invalid_context_relevance_outputs_service


def _write_job(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"run_started_at": "2026-10-05T00:00:00+00:00", "run_files": [], "judge": "primary",
            "out": "results/out.jsonl", "judged": 0}
    base.update(fields)
    with open(path, "a") as f:
        f.write(json.dumps(base) + "\n")


def test_output_file_missing_is_flagged_not_hidden(tmp_path, monkeypatch):
    history_path = tmp_path / "context_relevance_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "does_not_exist.jsonl"
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "CTXREL_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_ctxrel_v1_jobs()
    assert len(jobs) == 1
    assert jobs[0]["output_file_missing"] is True


def test_output_file_present_is_not_flagged(tmp_path, monkeypatch):
    history_path = tmp_path / "context_relevance_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "CTXREL_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    jobs = svc.load_ctxrel_v1_jobs()
    assert jobs[0]["output_file_missing"] is False


def test_invalid_output_hidden_by_default_but_shown_with_show_invalid(tmp_path, monkeypatch):
    history_path = tmp_path / "context_relevance_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "ctxrel_v1_smoke_test.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=[])
    monkeypatch.setattr(svc, "CTXREL_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)

    invalid_path = tmp_path / "invalid_context_relevance_outputs.jsonl"
    monkeypatch.setattr(invalid_context_relevance_outputs_service, "INVALID_CTXREL_OUTPUTS_PATH", invalid_path)
    invalid_context_relevance_outputs_service.mark_invalid("ctxrel_v1_smoke_test.jsonl", "smoke test")

    assert svc.load_ctxrel_v1_jobs() == []
    assert len(svc.load_ctxrel_v1_jobs(show_invalid=True)) == 1


def test_batch_id_falls_back_to_inferred_when_not_recorded(tmp_path, monkeypatch):
    """The actual root cause of the "context relevance table not shown"
    bug: a job whose run_files have no RECORDED batch_id must still
    resolve to the SAME inferred batch_id GET /api/history/batches
    would compute, not None."""
    history_path = tmp_path / "context_relevance_v1_run_history.jsonl"
    out_path = tmp_path / "results" / "out.jsonl"
    out_path.parent.mkdir(parents=True)
    out_path.write_text("")
    _write_job(history_path, out=str(out_path), run_files=["some_run_file.jsonl"])
    monkeypatch.setattr(svc, "CTXREL_V1_HISTORY_PATH", history_path)
    monkeypatch.setattr(svc, "JUDGE_EVAL_DIR", tmp_path)
    monkeypatch.setattr(svc, "_run_labels_for_job", lambda record: ["B-plain"])

    from app.services import batch_grouping_service

    monkeypatch.setattr(batch_grouping_service, "find_batch_id_for_output_paths", lambda paths: "inferred-fallback-batch")

    jobs = svc.load_ctxrel_v1_jobs()
    assert jobs[0]["batch_id"] == "inferred-fallback-batch"
