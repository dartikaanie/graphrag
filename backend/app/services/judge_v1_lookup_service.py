"""Read-only lookups over llm/evaluation/logs/judge_v1_run_history.jsonl
(judge-v1's append-only job log) for the dashboard -- separate from the
LEGACY judge's judge_lookup_service.py, which this module never touches
or depends on.

Each line in judge_v1_run_history.jsonl describes ONE judge-v1 CLI/
dashboard launch ("job"): run_files, judge_id, workers, out path, and the
run_batch() summary. There's no natural unique id in the raw record, so
`_job_id()` derives one deterministically (same style as
history_service._history_id()) from (run_started_at, out, judge).

Filtering policy (explicit per-field, never silent):
  - invalid_judge_runs.jsonl entries (pre-dashboard smoke-test jobs) are
    HIDDEN BY DEFAULT, unconditionally -- see invalid_judge_runs_service.
  - A job whose `out` file is missing on disk is STILL SHOWN, with
    `output_file_missing: True` -- the UI renders a visible warning
    instead of silently hiding it.
  - A job whose run_files resolve to a run_label outside the 9 canonical
    factorial labels is STILL SHOWN, with `is_other_config: True` -- the
    UI groups these under "Other configs" instead of hiding them.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from app.services import invalid_judge_runs_service

REPO_ROOT = Path(__file__).resolve().parents[3]
# Where run_files/out paths in judge_v1_run_history.jsonl are relative to
# -- tests monkeypatch THIS to a tmp_path. Kept separate from
# _REAL_JUDGE_EVAL_DIR (below) which always points at the real llm/
# evaluation/ directory, since THAT one must stay real for Python's
# import machinery to find _run_metadata.py etc. regardless of what a
# test is doing with fixture file paths.
JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
_REAL_JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
JUDGE_V1_HISTORY_PATH = JUDGE_EVAL_DIR / "logs" / "judge_v1_run_history.jsonl"

FACTORIAL_RUN_LABELS = {
    "A", "B-plain", "B-grounded",
    "C-uniform-plain", "C-uniform-grounded", "C-trust-plain", "C-trust-grounded",
    "D-plain", "D-grounded",
}


def _ensure_evaluation_on_path() -> None:
    if str(_REAL_JUDGE_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_REAL_JUDGE_EVAL_DIR))


def _run_metadata_module():
    _ensure_evaluation_on_path()
    return importlib.import_module("_run_metadata")


def _job_id(record: dict[str, Any]) -> str:
    basis = f"{record.get('run_started_at')}|{record.get('out')}|{record.get('judge')}"
    return "JV1-" + hashlib.md5(basis.encode()).hexdigest()[:10]


def _resolve_path(raw: str) -> Path:
    """run_files/out entries are relative to llm/evaluation/ (the CLI's
    own cwd convention, see llm_judge_hallucination_v1.py's docstring)."""
    p = Path(raw)
    return p if p.is_absolute() else (JUDGE_EVAL_DIR / p).resolve()


def _run_labels_for_job(record: dict[str, Any]) -> list[str | None]:
    """Best-effort: resolves each run_file's run_label via
    _run_metadata.resolve_run_metadata(). A file that can't be resolved
    (moved/deleted/no matching run_history entry) contributes None rather
    than raising -- this is a DISPLAY filter, not a hard gate like
    load_items()'s use of the same function."""
    run_files = record.get("run_files", [])
    if not run_files:
        return []
    mod = _run_metadata_module()
    labels = []
    for raw in run_files:
        try:
            meta = mod.resolve_run_metadata(str(_resolve_path(raw)))
            labels.append(meta.get("run_label"))
        except Exception:
            labels.append(None)
    return labels


def _batch_id_for_job(record: dict[str, Any]) -> str | None:
    """Best-effort batch_id for a job, resolved the same way as
    _run_labels_for_job -- the run_files' own run_history entries carry
    batch_id, not the judge job record itself. Older runs launched
    before batch_id was recorded (or a file that can't be resolved)
    contribute None from this first pass; the first non-None RECORDED
    value found wins (a job's run_files should all share one batch
    anyway). If none of the run_files have a recorded batch_id (the
    common case for any pilot run predating that feature), falls back
    to the SAME inferred-clustering GET /api/history/batches would
    compute -- without this fallback, a job covering pre-batch_id-era
    runs would never resolve to a batch_id at all, leaving it with no
    "View results" link anywhere in the UI (see batch_grouping_service.
    find_batch_id_for_output_paths()'s docstring)."""
    run_files = record.get("run_files", [])
    if not run_files:
        return None
    mod = _run_metadata_module()
    resolved_paths = []
    for raw in run_files:
        path = str(_resolve_path(raw))
        resolved_paths.append(path)
        try:
            meta = mod.resolve_run_metadata(path)
            batch_id = meta.get("batch_id")
            if batch_id:
                return batch_id
        except Exception:
            continue

    from app.services import batch_grouping_service

    return batch_grouping_service.find_batch_id_for_output_paths(resolved_paths)


def _served_model_mismatch_count(record: dict[str, Any]) -> int:
    """Counts records in the job's output file flagged
    served_model_mismatch=True by llm_judge_hallucination_v1.judge_one()
    -- surfaced here so the UI can show a warning without re-reading the
    file itself. Returns 0 if the file is missing/unreadable (the
    output_file_missing flag already covers that case separately)."""
    out_raw = record.get("out")
    if not out_raw:
        return 0
    path = _resolve_path(out_raw)
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("served_model_mismatch"):
                count += 1
    return count


def load_judge_v1_jobs(show_invalid: bool = False) -> list[dict[str, Any]]:
    """Every judge-v1 job, each annotated with job_id, output_file_missing,
    is_other_config, and run_labels -- newest first. Hides
    invalid_judge_runs.jsonl entries unless show_invalid=True. Never
    raises on a malformed/missing output file -- that's exactly the
    "show it, don't hide it" case this function exists to surface.
    """
    if not JUDGE_V1_HISTORY_PATH.exists():
        return []

    jobs: list[dict[str, Any]] = []
    with open(JUDGE_V1_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["job_id"] = _job_id(record)
            out_raw = record.get("out")
            record["output_file_missing"] = bool(out_raw) and not _resolve_path(out_raw).exists()
            run_labels = _run_labels_for_job(record)
            record["run_labels"] = run_labels
            record["is_other_config"] = any(
                (label is None or label not in FACTORIAL_RUN_LABELS) for label in run_labels
            ) if run_labels else True
            record["batch_id"] = _batch_id_for_job(record)
            record["served_model_mismatch_count"] = _served_model_mismatch_count(record)
            jobs.append(record)

    if not show_invalid:
        invalid_ids = invalid_judge_runs_service.load_invalid_judge_run_ids()
        if invalid_ids:
            jobs = [j for j in jobs if j["job_id"] not in invalid_ids]

    jobs.sort(key=lambda j: j.get("run_started_at", ""), reverse=True)
    return jobs


def get_job(job_id: str, show_invalid: bool = True) -> dict[str, Any] | None:
    for job in load_judge_v1_jobs(show_invalid=show_invalid):
        if job["job_id"] == job_id:
            return job
    return None
