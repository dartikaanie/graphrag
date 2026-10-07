"""
llm_judge_hallucination_v2.py
=====================================
Runner CLI for judge-v2 (judge_prompt_v2.py) -- a fix to judge-v1's
prompt for the systematic disagreement patterns found in the pilot
(see docs/JUDGE_V2_CHANGES.md). Mirrors llm_judge_hallucination_v1.py's
structure exactly (registry roles, workers, resume, raw-response
capture, failures/resolution logs, manifests, served-model check,
blinding) via the SAME shared helpers in _judge_common.py -- v1 is
NEVER imported or modified here, the two runners are independent.

Output records use judge_prompt_v2.parse_judgment_v2()'s field names:
`label`/`derived_label` are BOTH the deterministic official label (see
that module's docstring for why), `judge_reported_label` is what the
judge itself said, plus `answer_attempted`/`completeness`/
`attempted_claims_conflict` and per-claim `reference_conflict`.

Output filenames include "judge-v2" (via build_output_path-style
naming in engine_service.py's judge_v2_output_paths()) -- a v2 run can
never append into a v1 file, and vice versa; readers that mix the two
versions' records are refused the same way judge_agreement.py already
refuses a mixed judge_prompt_version file.

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 llm_judge_hallucination_v2.py \\
        --run-files ../a_pure_llm/results/condition_a_openai_gpt-4o-mini_n10_seed42.jsonl \\
        --judge both --limit 3 --out results/jv2_smoke_test.jsonl

SETUP .env (DEEPINFRA_API_KEY, OPENAI_API_KEY) -- lihat judge_clients.py.
QUESTIONS_PARQUET juga harus diset.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _judge_common import append_manifest, html_to_text, load_question_bodies  # noqa: E402
from _judge_common import append_judge_resolution_record as append_resolution_record  # noqa: E402
from _judge_common import call_judge_llm_with_retry  # noqa: E402
from _judge_common import check_served_model_mismatch  # noqa: E402
from _judge_common import judge_resume_key as _resume_key  # noqa: E402
from _judge_common import load_judge_already_done as load_already_done  # noqa: E402
from _judge_common import load_judge_failed_keys as load_failed_keys  # noqa: E402
from _judge_common import write_judge_record as write_record  # noqa: E402
from _run_metadata import RunMetadataNotFoundError, resolve_run_metadata  # noqa: E402
from judge_clients import get_judge_client  # noqa: E402
from llm.manifest import write_manifest  # noqa: E402
from judge_prompt_v2 import BLINDING_VERSION, PROMPT_VERSION, build_messages, parse_judgment_v2  # noqa: E402

log = print

REQUEST_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# File -> run metadata + item loading -- identical to v1's load_items()
# ---------------------------------------------------------------------------

def format_tags(raw_tags: str | None) -> str:
    if not raw_tags:
        return ""
    return ", ".join(re.findall(r"<([^>]+)>", raw_tags)) or raw_tags


def load_items(run_files: list[str], limit: int | None) -> list[dict]:
    from llm.manifest import assert_single_config_hash

    items = []
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
            items.append({
                **metadata,
                "question_id": record["question_id"],
                "title": record.get("title", ""),
                "tags": format_tags(record.get("tags")),
                "ground_truth_answer_html": record.get("ground_truth_answer", "") or "",
                "candidate": record.get("llm_answer", "") or "",
                "config_hash": record.get("config_hash"),
            })
    return items


def call_judge_with_retry(client, config, messages: list[dict]) -> tuple[dict | None, list[dict]]:
    return call_judge_llm_with_retry(client, config, messages, REQUEST_TEMPERATURE, log)


# ---------------------------------------------------------------------------
# One (item, judge) end to end -> output record
# ---------------------------------------------------------------------------

def _messages_sha256(messages: list[dict]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def judge_one(item: dict, judge_id: str, client, config, body_text: str, reference_text: str) -> dict:
    messages = build_messages(
        title=item["title"],
        body=body_text,
        tags=item["tags"],
        reference=reference_text,
        candidate=item["candidate"],
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    payload, attempts = call_judge_with_retry(client, config, messages)

    base = {
        "run_id": item["run_id"],
        "config_hash": item.get("config_hash"),
        "condition": item["condition"],
        "run_label": item["run_label"],
        "fusion_mode": item["fusion_mode"],
        "grounding": item.get("grounding"),
        "retrieval_version": item["retrieval_version"],
        "source_file": item["source_file"],
        "question_id": item["question_id"],
        "judge_id": judge_id,
        "judge_model": config.model,
        "judge_base_url": config.base_url,
        "prompt_version": PROMPT_VERSION,
        "blinding_version": BLINDING_VERSION,
        "timestamp_utc": timestamp,
        "messages_sha256": _messages_sha256(messages),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }

    if payload is None:
        base.update({
            "label": None, "derived_label": None, "judge_reported_label": None, "consistent": None,
            "answer_attempted": None, "completeness": None, "attempted_claims_conflict": None,
            "parse_error": True, "claims": [], "reasoning": None,
            "raw_response": None, "prompt_tokens": None, "completion_tokens": None,
            "latency_s": None, "call_failed": True,
            "error": attempts[-1]["error"] if attempts else None,
            "response_id": None, "response_model": None, "response_created": None,
            "system_fingerprint": None, "finish_reason": None, "usage_full": None,
            "request_params": {"model": config.model, "temperature": REQUEST_TEMPERATURE},
            "served_model_mismatch": None,
        })
        return base

    parsed = parse_judgment_v2(payload["raw"])
    base.update({
        "label": parsed.get("label"),
        "derived_label": parsed.get("derived_label"),
        "judge_reported_label": parsed.get("judge_reported_label"),
        "consistent": parsed.get("consistent"),
        "answer_attempted": parsed.get("answer_attempted"),
        "completeness": parsed.get("completeness"),
        "attempted_claims_conflict": parsed.get("attempted_claims_conflict"),
        "parse_error": parsed.get("parse_error", False),
        "claims": parsed.get("claims", []),
        "reasoning": parsed.get("reasoning"),
        "raw_response": payload["raw"],
        "prompt_tokens": payload["prompt_tokens"],
        "completion_tokens": payload["completion_tokens"],
        "latency_s": payload["latency_s"],
        "call_failed": False,
        "error": None,
        "response_id": payload["response_id"],
        "response_model": payload["response_model"],
        "response_created": payload["response_created"],
        "system_fingerprint": payload["system_fingerprint"],
        "finish_reason": payload["finish_reason"],
        "usage_full": payload["usage_full"],
        "request_params": payload["request_params"],
    })

    base["served_model_mismatch"] = check_served_model_mismatch(payload["response_model"], config, judge_id, log)
    return base


# ---------------------------------------------------------------------------
# Batch driver -- identical shape to v1's run_batch()
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
        body_text = html_to_text(body_html) if body_html else "[question body not found in parquet]"
        reference_text = html_to_text(item["ground_truth_answer_html"])
        for judge_id in judge_ids:
            judge_model = clients[judge_id][1].model
            key = (item["run_id"], item.get("config_hash"), item["question_id"], judge_id,
                   judge_model, PROMPT_VERSION, BLINDING_VERSION)
            if key in done:
                skipped += 1
                continue
            tasks.append((item, judge_id, body_text, reference_text))

    failed = 0
    parse_errors = 0
    judged = 0
    served_model_mismatches = 0
    attempted_claims_conflicts = 0
    label_counts: dict[str, dict[str, int]] = {}
    token_totals: dict[str, int] = {jid: 0 for jid in judge_ids}

    def _run(task):
        item, judge_id, body_text, reference_text = task
        client, config = clients[judge_id]
        return judge_one(item, judge_id, client, config, body_text, reference_text)

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
                                 "run_id": record["run_id"], "status": "failed",
                                 "done_count": done_count, "total_count": total_count})
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
            if record.get("attempted_claims_conflict"):
                attempted_claims_conflicts += 1
            for tok_field in ("prompt_tokens", "completion_tokens"):
                v = record.get(tok_field)
                if v:
                    token_totals[record["judge_id"]] += v
            label_counts.setdefault(record["run_label"], {})
            label = record["label"] or "PARSE_ERROR"
            label_counts[record["run_label"]][label] = label_counts[record["run_label"]].get(label, 0) + 1
            if on_progress:
                on_progress({"judge_id": record["judge_id"], "question_id": record["question_id"],
                             "run_id": record["run_id"], "status": "done",
                             "done_count": done_count, "total_count": total_count})

    return {
        "judged": judged,
        "skipped": skipped,
        "failed": failed,
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / judged, 3) if judged else None,
        "served_model_mismatches": served_model_mismatches,
        "attempted_claims_conflicts": attempted_claims_conflicts,
        "label_distribution_per_run_label": label_counts,
        "total_tokens_per_judge": token_totals,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-files", nargs="+", required=True,
                         help="One or more run JSONL paths (explicit, no recursive glob)")
    parser.add_argument("--judge", choices=["primary", "secondary", "both"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Max items per run file (smoke tests)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build and print messages for the first item, no API call")
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"),
                         help="Default from QUESTIONS_PARQUET in .env")
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
        log("[ERROR] no items loaded from --run-files")
        sys.exit(1)
    log(f"[config] {len(items)} item(s) from {len(args.run_files)} run file(s), judge(s)={judge_ids}")

    if args.dry_run:
        if not args.questions_parquet:
            log("[ERROR] --questions-parquet (or QUESTIONS_PARQUET in .env) is required even for --dry-run, "
                "to build a realistic prompt")
            sys.exit(1)
        item = items[0]
        bodies = load_question_bodies(args.questions_parquet, [item["question_id"]])
        body_text = html_to_text(bodies.get(item["question_id"], ""))
        reference_text = html_to_text(item["ground_truth_answer_html"])
        messages = build_messages(
            title=item["title"], body=body_text, tags=item["tags"],
            reference=reference_text, candidate=item["candidate"],
        )
        log(f"[dry-run] item: run_id={item['run_id']} run_label={item['run_label']} question_id={item['question_id']}")
        for m in messages:
            log(f"\n--- {m['role']} ---\n{m['content']}")
        return

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
    log(f"Failed calls : {summary['failed']} (logged to {failures_path}, NOT in {output_path} -- will retry on next run)")
    log(f"Parse errors : {summary['parse_errors']} ({summary['parse_error_rate']})")
    log(f"Attempted-claims conflicts: {summary['attempted_claims_conflicts']}")
    log(f"Label distribution per run_label: {json.dumps(summary['label_distribution_per_run_label'], indent=2)}")
    log(f"Total tokens per judge: {summary['total_tokens_per_judge']}")

    finished_at = datetime.now(timezone.utc).isoformat()
    append_manifest("judge_v2_run_history.jsonl", {
        "run_started_at": started_at,
        "run_files": args.run_files,
        "judge": args.judge,
        "limit": args.limit,
        "workers": args.workers,
        "out": str(output_path),
        "prompt_version": PROMPT_VERSION,
        **summary,
    })

    write_manifest(
        output_path,
        run_label=None,
        config={
            "run_files": args.run_files, "judge": args.judge, "limit": args.limit,
            "workers": args.workers, "questions_parquet": args.questions_parquet,
        },
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        item_counts={"attempted": summary["judged"] + summary["failed"],
                     "succeeded": summary["judged"], "failed": summary["failed"]},
        prompt_version=PROMPT_VERSION,
        blinding_version=BLINDING_VERSION,
        total_tokens=summary["total_tokens_per_judge"],
        extra_output_files=[failures_path],
    )


if __name__ == "__main__":
    main()
