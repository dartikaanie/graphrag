"""
llm_judge_context_relevance_v1.py
=====================================
Registry-based, single-item-per-call context relevance judge -- a
SEPARATE module from llm_judge_context_relevance.py (no suffix), which
already exists, is already producing data (results/ctxrel__...,
logs/context_relevance_run_history.jsonl), and is NEVER modified by this
file. The two differ in contract, not just version number:

  - llm_judge_context_relevance.py judges ALL k retrieved items of a
    question in ONE call (+ a second reference-aware "sufficiency"
    call), with Indonesian 3-class item labels (RELEVAN/SEBAGIAN/
    TIDAK_RELEVAN), raw --judge-provider/--judge-model CLI args (no
    registry), and per-file-only resume.
  - This module judges ONE (question, single context item) pair per
    call, labels RELEVANT/PARTIAL/IRRELEVANT (matching this task's
    spec), uses the judge registry (judge_clients.py -- same
    primary/secondary roles, served-model check, pricing as judge-v1),
    and DEDUPLICATES across every run file given to it on
    (question_id, answer_id, sha256(chunk_text)) -- since retrieved
    context has been verified byte-identical between the plain and
    grounded run of the same retrieval method (see docs/
    agent_prompt_judge_results_export.md Step 0), this means a
    plain/grounded pair shares one judged result instead of two.

ISOLATION FROM RETRIEVAL METADATA (same requirement as the sibling
module): the prompt gets ONLY the question (title/body/tags) and the
single context item's chunk_text -- never trust_weight/combined_score/
hop/source_stage/is_accepted/rel_type, and never which run/condition/
run_label the item came from. The candidate answer and reference answer
are also never shown to this judge -- Step 3 of the task spec is
explicit that this is a retrieval-content judge, not a generation judge.

Condition A has no retrieved_context at all and is excluded by load_items
(a file resolving to condition A raises, by the caller's choice, or is
simply never passed in -- see --run-files validation in main()).

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 llm_judge_context_relevance_v1.py \\
        --run-files ../c_graphrag/results/condition_c_...fw0-7-0-3_v3_....jsonl \\
                    ../c_graphrag/results/condition_c_...fw0-7-0-3_ungrounded_v3_....jsonl \\
        --judge primary --limit 2 --out results/ctxrel_v1_smoke_test.jsonl

SETUP .env: sama dengan judge-v1 (DEEPINFRA_API_KEY/OPENAI_API_KEY via
judge_clients.py) + QUESTIONS_PARQUET.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _judge_common import append_manifest, html_to_text, load_question_bodies, strip_fences  # noqa: E402
from _run_metadata import RunMetadataNotFoundError, resolve_run_metadata  # noqa: E402
from judge_clients import get_judge_client  # noqa: E402
from llm.manifest import write_manifest  # noqa: E402

log = print

PROMPT_VERSION = "ctxrel-v1"
LABELS = ("RELEVANT", "PARTIAL", "IRRELEVANT")
LABEL_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS = 6
REQUEST_TEMPERATURE = 0.0
_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# File -> deduplicated context items
# ---------------------------------------------------------------------------

def format_tags(raw_tags: str | None) -> str:
    if not raw_tags:
        return ""
    return ", ".join(re.findall(r"<([^>]+)>", raw_tags)) or raw_tags


def _text_hash(chunk_text: str) -> str:
    return hashlib.sha256((chunk_text or "").encode("utf-8")).hexdigest()


def load_items(run_files: list[str], limit: int | None) -> list[dict]:
    """Reads every run file's retrieved_context, flattens to one row per
    (question_id, context item), and deduplicates on (question_id,
    answer_id, text_hash) across ALL given files -- a plain/grounded pair
    of the same retrieval method contributes the item only once. --limit
    applies PER FILE to the question list (not to the deduplicated item
    count), matching load_items()'s per-file --limit convention in
    llm_judge_hallucination_v1.py.

    Raises RunMetadataNotFoundError (uncaught, by design) if a run file
    has no matching run_history.jsonl entry.
    """
    from llm.manifest import assert_single_config_hash

    seen: dict[tuple, dict] = {}
    for path in run_files:
        metadata = resolve_run_metadata(path)
        count = 0
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
                count += 1
                if limit is not None and count >= limit:
                    break
        assert_single_config_hash(records, source_label=path)

        for record in records:
            qid = record["question_id"]
            retrieved = record.get("retrieved_context") or []
            for item in retrieved:
                chunk_text = item.get("chunk_text", "") or ""
                key = (qid, item.get("answer_id"), _text_hash(chunk_text))
                if key in seen:
                    continue
                seen[key] = {
                    "question_id": qid,
                    "answer_id": item.get("answer_id"),
                    "text_hash": _text_hash(chunk_text),
                    "chunk_text": chunk_text,
                    "title": record.get("title", ""),
                    "tags": format_tags(record.get("tags")),
                    "source_files": [str(metadata["source_file"])],
                }
    return list(seen.values())


# ---------------------------------------------------------------------------
# Prompt construction + parsing
# ---------------------------------------------------------------------------

def build_messages(title: str, body_text: str, tags: str, chunk_text: str) -> list[dict]:
    system_content = (
        "You are an impartial evaluator (LLM-as-judge) assessing RETRIEVAL "
        "RELEVANCE for a software-engineering Q&A retrieval system. You will "
        "receive a question and exactly ONE retrieved text snippet. You do NOT "
        "know which system retrieved it, its rank/score, or any other metadata. "
        "Judge purely from whether this snippet's CONTENT would help answer "
        "the question. Respond with ONLY a single JSON object, no markdown "
        "fences, no extra text."
    )
    question_section = f"Question title: {title}\n"
    if tags:
        question_section += f"Tags: {tags}\n"
    if body_text:
        question_section += f"Question body: {body_text}\n"

    user_content = (
        f"{question_section}\n"
        f"Retrieved item:\n{chunk_text}\n\n"
        "Assign exactly one label:\n"
        "- \"RELEVANT\": directly addresses this problem.\n"
        "- \"PARTIAL\": related topic, but a different problem or only part of it.\n"
        "- \"IRRELEVANT\": unrelated to this problem.\n\n"
        "Respond with ONLY this JSON object:\n"
        "{\n"
        '  "label": "RELEVANT" | "PARTIAL" | "IRRELEVANT",\n'
        '  "reason": "<one sentence>"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_judgment(raw: str) -> dict:
    try:
        data = json.loads(strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return {"label": None, "reason": None, "parse_error": True}
    if not isinstance(data, dict):
        return {"label": None, "reason": None, "parse_error": True}
    label = data.get("label")
    if label not in LABELS:
        return {"label": None, "reason": None, "parse_error": True}
    reason = data.get("reason")
    reason = reason if isinstance(reason, str) else None
    return {"label": label, "reason": reason, "parse_error": False}


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _resume_key(r: dict) -> tuple:
    """(question_id, answer_id, text_hash, judge_id, judge_model,
    prompt_version) -- content-keyed, not run_id-keyed, so the dedup
    this module performs at load time is ALSO honored across separate
    CLI invocations: judging the same item again (e.g. because a later
    batch includes a run file sharing that item) is skipped for good,
    not just within one process's in-memory dedup."""
    return (
        r.get("question_id"), r.get("answer_id"), r.get("text_hash"),
        r.get("judge_id"), r.get("judge_model"), r.get("prompt_version"),
    )


def load_already_done(output_path: Path) -> set[tuple]:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_resume_key(json.loads(line)))
            except Exception:
                continue
    return done


def load_failed_keys(failures_path: Path) -> set[tuple]:
    keys = set()
    if not failures_path.exists():
        return keys
    with open(failures_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_resume_key(json.loads(line)))
            except Exception:
                continue
    return keys


# ---------------------------------------------------------------------------
# One (item, judge) call -- retry with exponential backoff + jitter
# (same shape as llm_judge_hallucination_v1.call_judge_with_retry)
# ---------------------------------------------------------------------------

def call_judge_with_retry(client, config, messages: list[dict]) -> tuple[dict | None, list[dict]]:
    import openai

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    attempts: list[dict] = []
    for attempt in range(1, RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS + 1):
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=config.model, messages=messages, temperature=REQUEST_TEMPERATURE,
            )
            latency_s = time.monotonic() - start
            attempts.append({"attempt": attempt, "error": None, "status_code": None})
            usage = response.usage
            return {
                "raw": response.choices[0].message.content,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "latency_s": round(latency_s, 3),
                "response_id": response.id,
                "response_model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }, attempts
        except retryable as e:
            status_code = getattr(e, "status_code", None)
            error_text = f"{type(e).__name__}: {e}"
            attempts.append({"attempt": attempt, "error": error_text, "status_code": status_code})
            if attempt == RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS:
                break
            sleep_s = min(60, (2 ** (attempt - 1))) + random.uniform(0, 1)
            log(f"      [retry {attempt}/{RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS}] {error_text} -- sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            error_text = f"{type(e).__name__}: {e}"
            attempts.append({"attempt": attempt, "error": error_text, "status_code": status_code})
            break

    return None, attempts


def judge_one(item: dict, judge_id: str, client, config, body_text: str) -> dict:
    messages = build_messages(item["title"], body_text, item["tags"], item["chunk_text"])
    timestamp = datetime.now(timezone.utc).isoformat()
    payload, attempts = call_judge_with_retry(client, config, messages)

    base = {
        "question_id": item["question_id"],
        "answer_id": item["answer_id"],
        "text_hash": item["text_hash"],
        "source_files": item["source_files"],
        "judge_id": judge_id,
        "judge_model": config.model,
        "judge_base_url": config.base_url,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": timestamp,
        "attempt_count": len(attempts),
    }

    if payload is None:
        base.update({
            "label": None, "reason": None, "parse_error": True, "call_failed": True,
            "error": attempts[-1]["error"] if attempts else None,
            "raw_response": None, "prompt_tokens": None, "completion_tokens": None,
            "latency_s": None, "response_model": None, "served_model_mismatch": None,
        })
        return base

    parsed = parse_judgment(payload["raw"])
    base.update({
        "label": parsed["label"], "reason": parsed["reason"], "parse_error": parsed["parse_error"],
        "call_failed": False, "error": None,
        "raw_response": payload["raw"],
        "prompt_tokens": payload["prompt_tokens"], "completion_tokens": payload["completion_tokens"],
        "latency_s": payload["latency_s"], "response_model": payload["response_model"],
    })

    expected = getattr(config, "expected_served_model", None)
    served_model_mismatch = bool(expected) and payload["response_model"] != expected
    base["served_model_mismatch"] = served_model_mismatch
    if served_model_mismatch:
        log(f"      [WARN] judge '{judge_id}': response_model='{payload['response_model']}' "
            f"differs from registry's expected_served_model='{expected}'.")
    return base


def write_record(output_path: Path, record: dict) -> None:
    with _WRITE_LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def append_resolution_record(resolved_path: Path, record: dict) -> None:
    resolution = {
        "question_id": record["question_id"], "answer_id": record["answer_id"],
        "text_hash": record["text_hash"], "judge_id": record["judge_id"],
        "prompt_version": record["prompt_version"], "resolved": True,
        "resolved_at_utc": record["timestamp_utc"],
    }
    write_record(resolved_path, resolution)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(items: list[dict], judge_ids: list[str], questions_parquet: str,
              output_path: Path, failures_path: Path, workers: int,
              resolved_path: Path | None = None, on_progress=None,
              check_cancel=None) -> dict:
    clients = {jid: get_judge_client(jid) for jid in judge_ids}

    question_ids = sorted({item["question_id"] for item in items})
    bodies_html = load_question_bodies(questions_parquet, question_ids)

    done = load_already_done(output_path)
    previously_failed_keys = load_failed_keys(failures_path)
    resolved_path = resolved_path or failures_path.with_name(failures_path.stem + "_resolved.jsonl")

    skipped = 0
    tasks = []
    for item in items:
        body_html = bodies_html.get(item["question_id"], "")
        body_text = html_to_text(body_html) if body_html else ""
        for judge_id in judge_ids:
            judge_model = clients[judge_id][1].model
            key = (item["question_id"], item["answer_id"], item["text_hash"], judge_id, judge_model, PROMPT_VERSION)
            if key in done:
                skipped += 1
                continue
            tasks.append((item, judge_id, body_text))

    failed = 0
    parse_errors = 0
    judged = 0
    served_model_mismatches = 0
    label_counts: dict[str, int] = {}
    token_totals: dict[str, int] = {jid: 0 for jid in judge_ids}

    def _run(task):
        item, judge_id, body_text = task
        client, config = clients[judge_id]
        return judge_one(item, judge_id, client, config, body_text)

    total_count = len(tasks)
    done_count = 0
    if check_cancel is not None and check_cancel():
        tasks = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        for future in as_completed(futures):
            record = future.result()
            done_count += 1
            if record["call_failed"]:
                write_record(failures_path, record)
                failed += 1
                if on_progress:
                    on_progress({"judge_id": record["judge_id"], "question_id": record["question_id"],
                                 "status": "failed", "done_count": done_count, "total_count": total_count})
                continue
            write_record(output_path, record)
            judged += 1
            key = _resume_key(record)
            if key in previously_failed_keys:
                append_resolution_record(resolved_path, record)
            if record["parse_error"]:
                parse_errors += 1
            if record.get("served_model_mismatch"):
                served_model_mismatches += 1
            for tok_field in ("prompt_tokens", "completion_tokens"):
                v = record.get(tok_field)
                if v:
                    token_totals[record["judge_id"]] += v
            label = record["label"] or "PARSE_ERROR"
            label_counts[label] = label_counts.get(label, 0) + 1
            if on_progress:
                on_progress({"judge_id": record["judge_id"], "question_id": record["question_id"],
                             "status": "done", "done_count": done_count, "total_count": total_count})

    return {
        "judged": judged, "skipped": skipped, "failed": failed,
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / judged, 3) if judged else None,
        "served_model_mismatches": served_model_mismatches,
        "label_distribution": label_counts,
        "total_tokens_per_judge": token_totals,
    }


# ---------------------------------------------------------------------------
# Per-run metrics (Step 3.3) -- joins a run's OWN retrieved_context order
# back against the (question_id, answer_id, text_hash)-keyed judged
# records, so rank position is derived per-run at read time, never stored
# on the (content-deduplicated) judged record itself.
# ---------------------------------------------------------------------------

def load_ctxrel_records(output_paths: list[Path]) -> dict[tuple, dict]:
    """Keyed by (question_id, answer_id, text_hash) -- last record wins
    on a key collision across files (a later judge run of the SAME
    judge_id/prompt_version re-judging is expected to be a strict
    append, so this only matters if two output files for the same
    judge_id are passed, which callers shouldn't do)."""
    by_key: dict[tuple, dict] = {}
    for path in output_paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_key[(r.get("question_id"), r.get("answer_id"), r.get("text_hash"))] = r
    return by_key


def compute_run_context_relevance(run_file: str, ctxrel_by_key: dict[tuple, dict]) -> dict:
    """Per-run_label-ready metrics for ONE run file: % RELEVANT/PARTIAL/
    IRRELEVANT over all context items that have a judged record, % of
    questions with >=1 RELEVANT item, mean relevance score, and
    relevance by rank position (rank 1 vs ranks 2-5). Items without a
    judged record yet are excluded from all of these (never treated as
    IRRELEVANT by default) -- `n_unjudged` reports how many were
    skipped so a caller can tell a complete metric from a partial one.
    """
    n_total = 0
    n_unjudged = 0
    label_counts: dict[str, int] = {}
    scores = []
    rank1_scores = []
    rank_rest_scores = []
    questions_with_relevant: set = set()
    all_questions: set = set()

    with open(run_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record.get("question_id")
            retrieved = record.get("retrieved_context") or []
            if not retrieved:
                continue
            all_questions.add(qid)
            for rank, item in enumerate(retrieved, start=1):
                n_total += 1
                key = (qid, item.get("answer_id"), _text_hash(item.get("chunk_text", "") or ""))
                judged = ctxrel_by_key.get(key)
                if judged is None or judged.get("label") not in LABELS:
                    n_unjudged += 1
                    continue
                label = judged["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
                score = LABEL_SCORE[label]
                scores.append(score)
                (rank1_scores if rank == 1 else rank_rest_scores).append(score)
                if label == "RELEVANT":
                    questions_with_relevant.add(qid)

    n_judged = n_total - n_unjudged
    return {
        "n_items_total": n_total,
        "n_items_judged": n_judged,
        "n_items_unjudged": n_unjudged,
        "pct_relevant": round(100 * label_counts.get("RELEVANT", 0) / n_judged, 2) if n_judged else None,
        "pct_partial": round(100 * label_counts.get("PARTIAL", 0) / n_judged, 2) if n_judged else None,
        "pct_irrelevant": round(100 * label_counts.get("IRRELEVANT", 0) / n_judged, 2) if n_judged else None,
        "n_questions": len(all_questions),
        "n_questions_with_relevant": len(questions_with_relevant),
        "pct_questions_with_relevant": round(100 * len(questions_with_relevant) / len(all_questions), 2) if all_questions else None,
        "mean_relevance_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_relevance_score_rank1": round(sum(rank1_scores) / len(rank1_scores), 4) if rank1_scores else None,
        "mean_relevance_score_rank2plus": round(sum(rank_rest_scores) / len(rank_rest_scores), 4) if rank_rest_scores else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-files", nargs="+", required=True)
    parser.add_argument("--judge", choices=["primary", "secondary", "both"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max questions per run file (smoke tests)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    args = parser.parse_args()

    for path in args.run_files:
        if not Path(path).exists():
            log(f"[ERROR] run file not found: {path}")
            sys.exit(1)

    judge_ids = ["primary", "secondary"] if args.judge == "both" else [args.judge]

    try:
        items = load_items(args.run_files, args.limit)
    except RunMetadataNotFoundError as e:
        log(f"[ERROR] {e}")
        sys.exit(1)
    if not items:
        log("[ERROR] no context items found in --run-files (condition A has no retrieved_context)")
        sys.exit(1)
    log(f"[config] {len(items)} unique (question, context item) pair(s) after dedup, "
        f"from {len(args.run_files)} run file(s), judge(s)={judge_ids}")

    if not args.questions_parquet:
        log("[ERROR] --questions-parquet (or QUESTIONS_PARQUET in .env) is required")
        sys.exit(1)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = output_path.with_name(output_path.stem + "_failures.jsonl")

    started_at = datetime.now(timezone.utc).isoformat()
    summary = run_batch(items, judge_ids, args.questions_parquet, output_path, failures_path, args.workers)

    log("\n=== SUMMARY ===")
    log(f"Items judged : {summary['judged']}")
    log(f"Skipped      : {summary['skipped']} (already in {output_path})")
    log(f"Failed calls : {summary['failed']} (logged to {failures_path})")
    log(f"Parse errors : {summary['parse_errors']} ({summary['parse_error_rate']})")
    log(f"Label distribution: {json.dumps(summary['label_distribution'], indent=2)}")
    log(f"Total tokens per judge: {summary['total_tokens_per_judge']}")

    finished_at = datetime.now(timezone.utc).isoformat()
    append_manifest("context_relevance_v1_run_history.jsonl", {
        "run_started_at": started_at, "run_files": args.run_files, "judge": args.judge,
        "limit": args.limit, "workers": args.workers, "out": str(output_path),
        "prompt_version": PROMPT_VERSION, **summary,
    })

    write_manifest(
        output_path, run_label=None,
        config={"run_files": args.run_files, "judge": args.judge, "limit": args.limit,
                "workers": args.workers, "questions_parquet": args.questions_parquet},
        started_at_utc=started_at, finished_at_utc=finished_at,
        item_counts={"attempted": summary["judged"] + summary["failed"],
                     "succeeded": summary["judged"], "failed": summary["failed"]},
        prompt_version=PROMPT_VERSION,
        total_tokens=summary["total_tokens_per_judge"],
        extra_output_files=[failures_path],
    )


if __name__ == "__main__":
    main()
