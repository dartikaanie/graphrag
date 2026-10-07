"""Read-only lookups over llm/evaluation/logs/context_relevance_v1_run_
history.jsonl (context-relevance-v1's append-only job log) for the
dashboard -- mirrors judge_v1_lookup_service.py's shape exactly, but for
a SEPARATE job log and a SEPARATE hide-list (invalid_context_relevance_
outputs_service, keyed on output FILE, not job_id -- a ctxrel-v1 smoke
test is identified by which file it wrote to, see that service's
docstring). Never touches judge_v1_run_history.jsonl or its files.

Without this service, context-relevance-v1 jobs were COMPLETELY
invisible in the dashboard -- RunJudgeV1Page.tsx's "Judge results"
section only ever listed judge_v1_lookup_service's (hallucination)
jobs. A user who launched a context-relevance job had no on-screen
confirmation it ran, no batch_id it resolved to, and no link to its
results -- see docs/agent_prompt_judge_results_export_bugfix.md.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from app.services import invalid_context_relevance_outputs_service

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
_REAL_JUDGE_EVAL_DIR = REPO_ROOT / "llm" / "evaluation"
CTXREL_V1_HISTORY_PATH = JUDGE_EVAL_DIR / "logs" / "context_relevance_v1_run_history.jsonl"

FACTORIAL_RUN_LABELS = {
    "A", "B-plain", "B-grounded",
    "C-uniform-plain", "C-uniform-grounded", "C-trust-plain", "C-trust-grounded",
    "D-plain", "D-grounded",
}


def _ensure_evaluation_on_path() -> None:
    import sys

    if str(_REAL_JUDGE_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(_REAL_JUDGE_EVAL_DIR))


def _run_metadata_module():
    _ensure_evaluation_on_path()
    import importlib

    return importlib.import_module("_run_metadata")


def _job_id(record: dict[str, Any]) -> str:
    basis = f"{record.get('run_started_at')}|{record.get('out')}|{record.get('judge')}"
    return "CTXREL-" + hashlib.md5(basis.encode()).hexdigest()[:10]


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


def load_ctxrel_v1_jobs(show_invalid: bool = False) -> list[dict[str, Any]]:
    """Every context-relevance-v1 job, annotated with job_id,
    output_file_missing, is_other_config, run_labels, and batch_id --
    newest first. Hides jobs whose `out` basename is registered in
    logs/invalid_context_relevance_outputs.jsonl (e.g. a manual smoke-
    test path) unless show_invalid=True."""
    if not CTXREL_V1_HISTORY_PATH.exists():
        return []

    jobs: list[dict[str, Any]] = []
    with open(CTXREL_V1_HISTORY_PATH) as f:
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
            jobs.append(record)

    if not show_invalid:
        invalid_basenames = invalid_context_relevance_outputs_service.load_invalid_output_basenames()
        if invalid_basenames:
            jobs = [j for j in jobs if Path(j.get("out", "")).name not in invalid_basenames]

    jobs.sort(key=lambda j: j.get("run_started_at", ""), reverse=True)
    return jobs
