"""Reads run_history.jsonl per condition and presents it as one unified,
paginated, filterable "transaction" table (PLAN_UI_UX.md §5.5/§7.7).

Deliberately reads ONLY the explicit active logs/run_history.jsonl per
condition folder -- never a recursive glob -- per the repo-audit note in
PLAN_UI_UX.md §1.5 / §4.5 (results_old/ and stray files must never leak in).
Lines that fail json.loads() are skipped, not fatal, so one corrupted line
never takes the whole endpoint down.
"""

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

HISTORY_PATHS: dict[str, Path] = {
    "A": REPO_ROOT / "llm" / "a_pure_llm" / "logs" / "run_history.jsonl",
    "B": REPO_ROOT / "llm" / "b_rag" / "logs" / "run_history.jsonl",
    "C": REPO_ROOT / "llm" / "c_graphrag" / "logs" / "run_history.jsonl",
}


def _history_id(condition: str, record: dict[str, Any]) -> str:
    """Stable synthetic id: CLI runs never had a run_id, so derive one
    deterministically from content that uniquely identifies a run (start
    time + output path) -- same record always hashes to the same id, so
    /api/history/{id} links stay valid across reloads.
    """
    basis = f"{condition}|{record.get('run_started_at')}|{record.get('output_path')}"
    return f"{condition}-{hashlib.md5(basis.encode()).hexdigest()[:10]}"


def _load_condition_history(condition: str) -> list[dict[str, Any]]:
    path = HISTORY_PATHS[condition]
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record.setdefault("condition", condition)
            record["history_id"] = _history_id(condition, record)
            records.append(record)
    return records


def list_history(
    condition: str | None, date_from: str | None, date_to: str | None, page: int, page_size: int
) -> tuple[list[dict[str, Any]], int]:
    conditions = [condition.upper()] if condition else list(HISTORY_PATHS.keys())
    all_records: list[dict[str, Any]] = []
    for c in conditions:
        all_records.extend(_load_condition_history(c))

    if date_from:
        all_records = [r for r in all_records if str(r.get("run_started_at", "")) >= date_from]
    if date_to:
        # inclusive of the whole day if only a date (no time) was given
        cutoff = date_to if "T" in date_to else date_to + "T23:59:59"
        all_records = [r for r in all_records if str(r.get("run_started_at", "")) <= cutoff]

    all_records.sort(key=lambda r: r.get("run_started_at", ""), reverse=True)

    total = len(all_records)
    start = (page - 1) * page_size
    page_records = all_records[start : start + page_size]
    return page_records, total


def get_history_detail(history_id: str) -> dict[str, Any] | None:
    condition = history_id.split("-", 1)[0]
    if condition not in HISTORY_PATHS:
        return None
    record = next((r for r in _load_condition_history(condition) if r["history_id"] == history_id), None)
    if not record:
        return None

    results: list[dict[str, Any]] = []
    output_path = record.get("output_path")
    if output_path and Path(output_path).exists():
        with open(output_path) as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results.append({
                    "question_id": row.get("question_id"),
                    "index": i,
                    "total": record.get("n_processed", len(results) + 1),
                    "status": "done",
                    "similarity": row.get("cosine_similarity"),
                    "error": None,
                })

    started_at = record.get("run_started_at")
    finished_at = None
    if started_at and record.get("duration_sec") is not None:
        try:
            finished_at = (
                datetime.fromisoformat(started_at) + timedelta(seconds=record["duration_sec"])
            ).isoformat()
        except ValueError:
            pass

    summary_keys = [
        "n_processed", "cosine_similarity_mean", "cosine_similarity_median",
        "pct_similarity_above_0_5", "pct_with_citation", "pct_with_valid_citation",
        "avg_retrieval_latency_sec",
    ]
    summary = {k: record[k] for k in summary_keys if k in record}

    return {
        "run_id": history_id,
        "condition": condition,
        "mode": "single" if "single" in str(output_path) else "batch",
        "status": record.get("status", "unknown"),
        "params": {
            "provider": record.get("provider"),
            "model": record.get("model"),
            "n_sample": record.get("n_sample_target"),
            "seed": record.get("seed"),
            "top_k": record.get("top_k"),
            "n_anchor": record.get("n_anchor"),
            "n_semantic_expansion": record.get("n_semantic_expansion"),
        },
        "created_at": started_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "progress": {"current": record.get("n_processed", 0), "total": record.get("n_sample_target", 0)},
        "results": results,
        "summary": summary,
        "error": None if record.get("status") != "failed" else "See condition logs for details.",
        "output_path": output_path,
    }


def get_history_result_detail(history_id: str, question_id: int) -> dict[str, Any] | None:
    detail = get_history_detail(history_id)
    if not detail or not detail.get("output_path"):
        return None
    output_path = Path(detail["output_path"])
    if not output_path.exists():
        return None
    with open(output_path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("question_id") == question_id:
                return record
    return None


def total_pages(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size))
