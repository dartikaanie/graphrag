"""Read-only lookups over llm/evaluation/logs/judge_v2_run_history.jsonl
(judge-v2's append-only job log) for the dashboard -- mirrors
judge_v1_lookup_service.py exactly, for a SEPARATE job log. judge-v1's
own job log/lookup service is never touched or depended on by this
module.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from app.services import invalid_judge_runs_service

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
_REAL_JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
JUDGE_V2_HISTORY_PATH = JUDGE_EVAL_DIR / "logs" / "judge_v2_run_history.jsonl"

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
    return "JV2-" + hashlib.md5(basis.encode()).hexdigest()[:10]


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (JUDGE_EVAL_DIR / p).resolve()


def _run_labels_for_job(record: dict[str, Any]) -> list[str | None]:
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
    """Same recorded-then-inferred resolution as judge_v1_lookup_
    service._batch_id_for_job() -- see that function's docstring."""
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


def load_judge_v2_jobs(show_invalid: bool = False) -> list[dict[str, Any]]:
    """Every judge-v2 job, each annotated with job_id, output_file_missing,
    is_other_config, run_labels, and batch_id -- newest first. Hides
    invalid_judge_runs.jsonl entries unless show_invalid=True (SAME
    hide-list as judge-v1's jobs -- job_id namespaces ("JV1-"/"JV2-")
    never collide, so sharing the list is safe)."""
    if not JUDGE_V2_HISTORY_PATH.exists():
        return []

    jobs: list[dict[str, Any]] = []
    with open(JUDGE_V2_HISTORY_PATH) as f:
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
    for job in load_judge_v2_jobs(show_invalid=show_invalid):
        if job["job_id"] == job_id:
            return job
    return None
