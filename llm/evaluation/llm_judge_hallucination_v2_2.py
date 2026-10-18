"""
llm_judge_hallucination_v2_2.py
=====================================
Runner CLI for judge-v2.2 (judge_prompt_v2_2.py; docs/DECISION_C_SCORING.md,
Amendment 11). Same structure as the v2.1 runner (registry roles, workers,
resume, raw-response capture, failures/resolution logs, manifests,
served-model check, calibration guard) -- the v2.1 runner is imported only
for its small helpers and is never modified.

Differences from the v2.1 runner:
  - the retrieval context of each record (B/C/D) is sent as [K1]..[K5];
    condition A gets an explicit "no retrieved context";
  - the candidate is blinded with blind-v3; the [SO-<id>] -> [Kn] map is
    built from the record's context BEFORE blinding and stored per record;
  - every record carries supported_by_context / cited_items /
    cited_item_supports per claim, approach_match, and the derived
    faithfulness and citation_support (judge_prompt_v2_2);
  - PROMPT_VERSION "judge-v2.2", temperature 0, max_tokens 2000, recorded
    per record and in the manifest.

CARA PAKAI (from llm/evaluation)
----------------------------------------------------------------------------
    python3 llm_judge_hallucination_v2_2.py --run-files <run>.jsonl --judge both --out results/jv2_2_<name>.jsonl
    python3 llm_judge_hallucination_v2_2.py --run-files <run>.jsonl --judge primary --out x.jsonl --dry-run
"""

import argparse
import hashlib
import json
import os
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
from judge_binary import derive_hallucinated, severity_of  # noqa: E402
from judge_clients import get_judge_client  # noqa: E402
from judge_prompt_v2_2 import (BLINDING_VERSION, PARSE_VERSION, PROMPT_VERSION, REQUEST_MAX_TOKENS,  # noqa: E402
                               REQUEST_TEMPERATURE, build_messages, candidate_with_citation_labels,
                               parse_judgment_v2_2, prepare_judge_input)
from llm.manifest import write_manifest  # noqa: E402
from llm_judge_hallucination_v2_1 import format_tags  # noqa: E402

log = print

CONDITIONS_WITHOUT_CONTEXT = ("A",)
LAST_CALIBRATION_GUARD: dict = {}


def has_context_for(condition: str | None) -> bool:
    return (condition or "").upper() not in CONDITIONS_WITHOUT_CONTEXT


def load_items(run_files: list[str], limit: int | None, allow_unannotated_calibration: bool = False) -> list[dict]:
    from _judge_common import assert_calibration_judging_allowed
    from llm.manifest import assert_single_config_hash

    global LAST_CALIBRATION_GUARD
    LAST_CALIBRATION_GUARD = assert_calibration_judging_allowed(run_files, allow_unannotated_calibration, log)

    items = []
    for path in run_files:
        metadata = resolve_run_metadata(path)
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
                    if limit is not None and len(records) >= limit:
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
                "retrieved_context": record.get("retrieved_context") or [],
                "config_hash": record.get("config_hash"),
            })
    return items


def _messages_sha256(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


_EMPTY_PARSE = {
    "label": None, "derived_label": None, "judge_reported_label": None, "consistent": None,
    "answer_attempted": None, "completeness": None, "attempted_claims_conflict": None, "parse_error": True,
    "claims": [], "reasoning": None, "approach_match": None, "faithfulness": None, "citation_support": None,
    "n_cited_claims": None, "normalized_fields": [], "n_claims": None, "n_error_claims": None,
    "n_error_claims_without_severity": None, "non_error_claims_over_cap": None,
}


def judge_one(item: dict, judge_id: str, client, config, body_text: str, reference_text: str,
              keep_citation_labels: bool = False) -> dict:
    """`keep_citation_labels` is for the authority bias probe only (judge_bias_probes.py): the candidate keeps
    its citations as [Kn] labels instead of having them removed."""
    has_context = has_context_for(item.get("condition"))
    prepared = prepare_judge_input(item["candidate"], item["retrieved_context"], has_context)
    if keep_citation_labels:
        prepared["candidate_blind"] = candidate_with_citation_labels(item["candidate"], prepared["citation_map"])
    messages = build_messages(item["title"], body_text, item["tags"], reference_text, item["candidate"],
                              prepared=prepared)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload, attempts = call_judge_llm_with_retry(client, config, messages, REQUEST_TEMPERATURE, log,
                                                  REQUEST_MAX_TOKENS)
    base = {
        "run_id": item["run_id"], "config_hash": item.get("config_hash"), "condition": item["condition"],
        "run_label": item["run_label"], "fusion_mode": item.get("fusion_mode"), "grounding": item.get("grounding"),
        "retrieval_version": item.get("retrieval_version"), "source_file": item["source_file"],
        "question_id": item["question_id"], "judge_id": judge_id, "judge_model": config.model,
        "judge_base_url": config.base_url, "prompt_version": PROMPT_VERSION, "judge_version": PROMPT_VERSION,
        "blinding_version": BLINDING_VERSION, "parse_version": PARSE_VERSION,
        "temperature": REQUEST_TEMPERATURE, "max_tokens": REQUEST_MAX_TOKENS, "timestamp_utc": timestamp,
        "messages_sha256": _messages_sha256(messages), "attempt_count": len(attempts), "attempts": attempts,
        "has_context": has_context, "n_context_items": prepared["n_context_items"],
        "citation_map": prepared["citation_map"], "n_cited_statements": len(prepared["cited_statements"]),
        "probe_citation_labels_kept": keep_citation_labels,
    }
    if payload is None:
        base.update({**_EMPTY_PARSE, "raw_response": None, "prompt_tokens": None, "completion_tokens": None,
                     "latency_s": None, "call_failed": True, "error": attempts[-1]["error"] if attempts else None,
                     "response_id": None, "response_model": None, "response_created": None,
                     "system_fingerprint": None, "finish_reason": None, "usage_full": None,
                     "request_params": {"model": config.model, "temperature": REQUEST_TEMPERATURE,
                                        "max_tokens": REQUEST_MAX_TOKENS},
                     "served_model_mismatch": None, "truncated": None, "hallucinated": None, "severity_label": None})
        return base

    parsed = parse_judgment_v2_2(payload["raw"], has_context, prepared["offered_items"])
    base.update({k: parsed.get(k, v) for k, v in _EMPTY_PARSE.items()})
    base.update({
        "parse_error": parsed.get("parse_error", False), "raw_response": payload["raw"],
        "prompt_tokens": payload["prompt_tokens"], "completion_tokens": payload["completion_tokens"],
        "latency_s": payload["latency_s"], "call_failed": False, "error": None,
        "response_id": payload["response_id"], "response_model": payload["response_model"],
        "response_created": payload["response_created"], "system_fingerprint": payload["system_fingerprint"],
        "finish_reason": payload["finish_reason"], "usage_full": payload["usage_full"],
        "request_params": payload["request_params"], "truncated": payload["finish_reason"] == "length",
    })
    base["hallucinated"] = derive_hallucinated(base)
    base["severity_label"] = severity_of(base)
    base["served_model_mismatch"] = check_served_model_mismatch(payload["response_model"], config, judge_id, log)
    return base


def run_batch(items: list[dict], judge_ids: list[str], questions_parquet: str, output_path: Path,
              failures_path: Path, workers: int, resolved_path: Path | None = None, clients: dict | None = None,
              bodies_html: dict | None = None) -> dict:
    """`clients` / `bodies_html` may be injected (tests); otherwise from the registry / the parquet."""
    clients = clients or {jid: get_judge_client(jid) for jid in judge_ids}
    if bodies_html is None:
        bodies_html = load_question_bodies(questions_parquet, sorted({it["question_id"] for it in items}))
    done = load_already_done(output_path)
    previously_failed = load_failed_keys(failures_path)
    resolved_path = resolved_path or failures_path.with_name(failures_path.stem + "_resolved.jsonl")

    tasks, skipped = [], 0
    for item in items:
        body_html = bodies_html.get(item["question_id"], "")
        body_text = html_to_text(body_html) if body_html else "[question body not found in parquet]"
        reference_text = html_to_text(item["ground_truth_answer_html"])
        for judge_id in judge_ids:
            key = (item["run_id"], item.get("config_hash"), item["question_id"], judge_id,
                   clients[judge_id][1].model, PROMPT_VERSION, BLINDING_VERSION, PARSE_VERSION, REQUEST_MAX_TOKENS)
            if key in done:
                skipped += 1
                continue
            tasks.append((item, judge_id, body_text, reference_text))

    counts = {"judged": 0, "failed": 0, "parse_errors": 0, "served_model_mismatches": 0}
    tokens = {jid: 0 for jid in judge_ids}

    def _run(task):
        item, judge_id, body_text, reference_text = task
        client, config = clients[judge_id]
        return judge_one(item, judge_id, client, config, body_text, reference_text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_run, t) for t in tasks]):
            record = future.result()
            if record["call_failed"]:
                write_record(failures_path, record)
                counts["failed"] += 1
                continue
            write_record(output_path, record)
            counts["judged"] += 1
            if _resume_key(record) in previously_failed:
                append_resolution_record(resolved_path, record)
            counts["parse_errors"] += bool(record["parse_error"])
            counts["served_model_mismatches"] += bool(record.get("served_model_mismatch"))
            for f in ("prompt_tokens", "completion_tokens"):
                tokens[record["judge_id"]] += record.get(f) or 0
    return {**counts, "skipped": skipped,
            "parse_error_rate": round(counts["parse_errors"] / counts["judged"], 3) if counts["judged"] else None,
            "total_tokens_per_judge": tokens}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-files", nargs="+", required=True)
    parser.add_argument("--judge", choices=["primary", "secondary", "both"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the messages of the first item, no API call")
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--allow-unannotated-calibration", action="store_true",
                        help="Override the calibration guard (logged in the run history and manifest)")
    args = parser.parse_args(argv)

    for path in args.run_files:
        if not Path(path).exists():
            log(f"[ERROR] run file not found: {path}")
            return 1
    judge_ids = ["primary", "secondary"] if args.judge == "both" else [args.judge]
    try:
        items = load_items(args.run_files, args.limit, args.allow_unannotated_calibration)
    except RunMetadataNotFoundError as e:
        log(f"[ERROR] {e}")
        return 1
    if not items:
        log("[ERROR] no items loaded from --run-files")
        return 1
    if not args.questions_parquet:
        log("[ERROR] --questions-parquet (or QUESTIONS_PARQUET in .env) is required")
        return 1

    if args.dry_run:
        item = items[0]
        body = html_to_text(load_question_bodies(args.questions_parquet, [item["question_id"]]).get(item["question_id"], ""))
        prepared = prepare_judge_input(item["candidate"], item["retrieved_context"], has_context_for(item["condition"]))
        for m in build_messages(item["title"], body, item["tags"], html_to_text(item["ground_truth_answer_html"]),
                                item["candidate"], prepared=prepared):
            log(f"\n--- {m['role']} ---\n{m['content']}")
        return 0

    from judge_versions import JudgeVersionError, check_run_allowed
    try:  # Amendment 11 freeze guard: allowed versions + revision limit
        version_purpose = check_run_allowed(PROMPT_VERSION, args.run_files, log)
    except JudgeVersionError as e:
        log(f"[ERROR] {e}")
        return 1

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = output_path.with_name(output_path.stem + "_failures.jsonl")
    started_at = datetime.now(timezone.utc).isoformat()
    summary = run_batch(items, judge_ids, args.questions_parquet, output_path, failures_path, args.workers)
    log(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")

    append_manifest("judge_v2_2_run_history.jsonl", {
        "run_started_at": started_at, "run_files": args.run_files, "judge": args.judge, "limit": args.limit,
        "workers": args.workers, "out": str(output_path), "prompt_version": PROMPT_VERSION,
        "blinding_version": BLINDING_VERSION, "temperature": REQUEST_TEMPERATURE, "max_tokens": REQUEST_MAX_TOKENS,
        "calibration_guard": LAST_CALIBRATION_GUARD, "version_purpose": version_purpose, **summary,
    })
    write_manifest(
        output_path, run_label=None,
        config={"run_files": args.run_files, "judge": args.judge, "limit": args.limit, "workers": args.workers,
                "questions_parquet": args.questions_parquet, "temperature": REQUEST_TEMPERATURE,
                "max_tokens": REQUEST_MAX_TOKENS, "calibration_guard": LAST_CALIBRATION_GUARD,
                "version_purpose": version_purpose},
        started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        item_counts={"attempted": summary["judged"] + summary["failed"], "succeeded": summary["judged"],
                     "failed": summary["failed"]},
        prompt_version=PROMPT_VERSION, judge_version=PROMPT_VERSION, blinding_version=BLINDING_VERSION,
        total_tokens=summary["total_tokens_per_judge"], extra_output_files=[failures_path],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
