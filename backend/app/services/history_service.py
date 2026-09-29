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

from app.services import judge_lookup_service

REPO_ROOT = Path(__file__).resolve().parents[3]

HISTORY_PATHS: dict[str, Path] = {
    "A": REPO_ROOT / "llm" / "a_pure_llm" / "logs" / "run_history.jsonl",
    "B": REPO_ROOT / "llm" / "b_rag" / "logs" / "run_history.jsonl",
    "C": REPO_ROOT / "llm" / "c_graphrag" / "logs" / "run_history.jsonl",
    "D": REPO_ROOT / "llm" / "d_lightrag" / "logs" / "run_history.jsonl",
}

# The moment PROMPT_VERSION="v2" (llm/prompts.py) and D_RETRIEVAL_VERSION="v2"
# (llm/d_lightrag.py, the EMBED_SIM fix) were introduced -- see docs/README.md
# §8. Neither field existed in run_history.jsonl before this instant, so a
# record with run_started_at strictly before this cutoff and MISSING the
# field can be reliably inferred as "v1"; a record after the cutoff but
# still missing the field is genuinely ambiguous (could be a stale cached
# .pyc, a manually edited record, etc.) and must show "--", never a guess.
PROMPT_V2_CUTOFF = "2026-09-28T19:06:21+00:00"

# The moment the SEPARATE determinism fix landed: D_RETRIEVAL_VERSION "v2"
# -> "v3" (LIMIT-before-answer-join fix + deterministic tie-breaks
# everywhere in D), and C_RETRIEVAL_VERSION "v2" introduced for the first
# time (C's retrieved_context changed for 7/10 pilot questions once
# deterministic tie-breaks were added -- see docs/README.md §8). A record
# missing c_retrieval_version/d_retrieval_version entirely and dated before
# THIS cutoff used the old non-deterministic-tie-break code, regardless of
# whether it also predates PROMPT_V2_CUTOFF.
C_D_RETRIEVAL_FIX_CUTOFF = "2026-09-28T22:08:24+00:00"


def _infer_version(record: dict[str, Any], field: str, cutoff: str = PROMPT_V2_CUTOFF) -> str | None:
    """value if the record has `field`; "v1 (inferred)" if it predates
    `cutoff` and doesn't; None (-> "--" in the UI) otherwise."""
    value = record.get(field)
    if value is not None:
        return value
    started_at = record.get("run_started_at")
    if started_at and str(started_at) < cutoff:
        return "v1 (inferred)"
    return None


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
        "avg_retrieval_latency_sec", "duration_sec",
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
            "oversample_pool": record.get("oversample_pool"),
            "n_candidates_after_token_filter": record.get("n_candidates_after_token_filter"),
            "top_k": record.get("top_k"),
            "n_anchor": record.get("n_anchor"),
            "n_semantic_expansion": record.get("n_semantic_expansion"),
            "require_citation": record.get("require_citation"),
            # Was silently dropped before -- present in run_history.jsonl for
            # B/C (see engine_service.py) since it was first wired, but never
            # surfaced here, so Compare/History always showed "--" for it
            # regardless of what was actually used.
            "log_full_candidates": record.get("log_full_candidates"),
            "index_pool": record.get("index_pool"),
            "fusion_mode": record.get("fusion_mode"),
            "fusion_w_path_trust": record.get("fusion_w_path_trust"),
            "fusion_w_intrinsic": record.get("fusion_w_answer_intrinsic_trust"),
            "semantic_expansion_trust_cap": record.get("semantic_expansion_trust_cap"),
            "enable_semantic_expansion": record.get("enable_semantic_expansion"),
            "n_low_level": record.get("n_low_level"),
            "n_high_level": record.get("n_high_level"),
            "require_grounding": record.get("require_grounding"),
            # "v1 (inferred)" only when reliably inferable from run_started_at
            # vs. PROMPT_V2_CUTOFF above -- never guessed otherwise ("--").
            "prompt_version": _infer_version(record, "prompt_version"),
            "d_retrieval_version": (
                _infer_version(record, "d_retrieval_version", C_D_RETRIEVAL_FIX_CUTOFF) if condition == "D" else None
            ),
            "c_retrieval_version": (
                _infer_version(record, "c_retrieval_version", C_D_RETRIEVAL_FIX_CUTOFF) if condition == "C" else None
            ),
        },
        "created_at": started_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "progress": {"current": record.get("n_processed", 0), "total": record.get("n_sample_target", 0)},
        "results": results,
        "summary": summary,
        "error": None if record.get("status") != "failed" else "See condition logs for details.",
        "output_path": output_path,
        "judge_evaluations": judge_lookup_service.get_judge_evaluations_for_input(output_path) if output_path else [],
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
                record["judge_results"] = judge_lookup_service.get_judge_results_for_question(
                    detail["output_path"], question_id,
                )
                return record
    return None


def delete_history_record(history_id: str) -> bool:
    """Remove one record from its condition's run_history.jsonl (rewrite the
    file without that line). Deliberately leaves the underlying results
    .jsonl (output_path) untouched -- this deletes the "transaction log"
    entry, which is what the History UI lists, not the raw experiment data
    itself, so a delete here can never lose the actual per-question results
    on disk. Returns False if the record/condition wasn't found."""
    condition = history_id.split("-", 1)[0]
    if condition not in HISTORY_PATHS:
        return False
    path = HISTORY_PATHS[condition]
    if not path.exists():
        return False

    kept_lines = []
    found = False
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            record.setdefault("condition", condition)
            if _history_id(condition, record) == history_id:
                found = True
                continue
            kept_lines.append(line if line.endswith("\n") else line + "\n")

    if not found:
        return False

    with open(path, "w") as f:
        f.writelines(kept_lines)
    return True


def total_pages(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size))


# ---------------------------------------------------------------------
# Phase 3 -- sample-consistency check for History -> Compare. The whole
# point of a paired comparison (this thesis's design) is that every
# condition evaluated the SAME question_ids -- this answers "did it
# actually?" directly from the result files, instead of trusting that
# seed/oversample_pool were entered identically (Phase 1/2 make that
# easier to get right, but this is the check that PROVES it per comparison).
# ---------------------------------------------------------------------

# (output_path -> (mtime, [question_id, ...])), process-lifetime, invalidated
# by mtime so a re-run of the same file (resume/force) is picked up.
#
# Reads the TOP-LEVEL "question_id" via json.loads(line)["question_id"] --
# NOT a regex over the raw line. A regex matching the first `"question_id":`
# substring would rely on it always being the first key in the dict, which
# is true of every condition script's record={...} literal TODAY but is not
# a contract anything enforces, and every record ALSO has one or more
# NESTED "question_id" keys inside retrieved_context items (a different
# field entirely -- the source thread id of that retrieved chunk, not the
# question being evaluated). A regex is one refactor away from silently
# reading the wrong number. json.loads() costs more per line than a regex
# would, but correctness here matters more than the difference at n<=384.
_QID_CACHE: dict[str, tuple[float, list[int]]] = {}


def _read_question_ids_cached(output_path: str) -> list[int]:
    path = Path(output_path)
    if not path.exists():
        return []
    mtime = path.stat().st_mtime
    cached = _QID_CACHE.get(output_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    ids: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("question_id")
            if qid is not None:
                ids.append(int(qid))
    _QID_CACHE[output_path] = (mtime, ids)
    return ids


def compute_sample_consistency(history_ids: list[str]) -> dict[str, Any]:
    """`sample_consistency`:
      - "identical"     : every run's question_id SET is exactly the same.
      - "subset_nested"  : not identical, but every run's set is a subset of
        the largest run's set (nested sampling -- e.g. an n=10 pilot inside
        the planned n=384 sample). Also reports `is_exact_prefix`: whether
        each smaller run's question_ids, IN ORDER, equal the first N
        entries of the largest run's question_ids (true nested sampling,
        not just "happens to be a subset").
      - "different"      : at least one run has a question_id the others
        (or the largest run) don't -- NOT a valid paired comparison.
    Runs whose output_path is missing/unreadable are silently excluded from
    the returned `runs` list (not fatal) -- fewer than 2 resolvable runs
    yields "different" with a `note` explaining why, so a caller always gets
    a directly-usable verdict rather than a crash.
    """
    runs: list[dict[str, Any]] = []
    for hid in history_ids:
        detail = get_history_detail(hid)
        if not detail or not detail.get("output_path"):
            continue
        qids = _read_question_ids_cached(detail["output_path"])
        runs.append({
            "history_id": hid,
            "condition": detail.get("condition"),
            "n": len(qids),
            "_question_ids": qids,
        })

    if len(runs) < 2:
        return {
            "sample_consistency": "different",
            "runs": [{k: v for k, v in r.items() if k != "_question_ids"} for r in runs],
            "intersection_size": len(runs[0]["_question_ids"]) if runs else 0,
            "is_exact_prefix": None,
            "note": "Fewer than 2 runs had a readable output_path -- cannot compare.",
        }

    sets = [set(r["_question_ids"]) for r in runs]
    intersection = set.intersection(*sets)
    identical = all(s == sets[0] for s in sets)

    is_exact_prefix = None
    if identical:
        consistency = "identical"
    else:
        largest_set = max(sets, key=len)
        nested = all(s <= largest_set for s in sets)
        if nested:
            consistency = "subset_nested"
            by_size = sorted(runs, key=lambda r: r["n"])
            largest_run = by_size[-1]
            is_exact_prefix = all(
                r["_question_ids"] == largest_run["_question_ids"][: r["n"]] for r in by_size[:-1]
            )
        else:
            consistency = "different"

    return {
        "sample_consistency": consistency,
        "runs": [{k: v for k, v in r.items() if k != "_question_ids"} for r in runs],
        "intersection_size": len(intersection),
        "is_exact_prefix": is_exact_prefix,
    }
