"""
llm_judge_hallucination_v1.py
=====================================
Runner CLI untuk judge-v1 (judge_prompt_v1.py): LLM-as-Judge 3-kelas +
ABSTAIN untuk mengevaluasi halusinasi jawaban Kondisi A/B/C/D, dengan
primary judge Llama-3.3-70B-Instruct (DeepInfra, model family BEDA dari
generator) dan secondary judge gpt-4o-mini (OpenAI, HANYA utk inter-judge
agreement -- lihat judge_clients.py).

NAMA SENGAJA DIBEDAKAN dari llm_judge_hallucination.py (TANPA sufiks
_v1) yang SUDAH ADA di folder ini -- itu judge versi lama (3 kelas tanpa
ABSTAIN, satu provider OpenAI-generic, satu --condition per run, dipakai
dashboard History via judge_lookup_service.py + logs/judge_run_history.
jsonl). File ini TIDAK MENGUBAH ATAU MENGGANTI judge lama itu sama
sekali -- keduanya hidup berdampingan, output & manifest-nya terpisah
total (results/jv1__... vs results/judged__..., logs/
judge_v1_run_history.jsonl vs logs/judge_run_history.jsonl).

TIDAK MENYENTUH RETRIEVED_CONTEXT SAMA SEKALI -- judge-v1 (lihat
SYSTEM_PROMPT di judge_prompt_v1.py) mengukur **hallucination
(reference-based)**: candidate dinilai HANYA terhadap reference answer
(accepted SO answer) + pengetahuan umum judge sendiri. Ini BUKAN
**faithfulness (context-grounded)** -- metrik berbeda yang diukur
llm_judge_hallucination.py (versi lama, TANPA sufiks _v1) & llm_judge_
context_relevance.py, yang menilai candidate terhadap retrieved_context
yang benar-benar dilihat generator. Dua metrik ini TIDAK bisa saling
dipertukarkan sbg sinonim meski keduanya menyinggung "apakah jawaban
cocok dengan sesuatu" -- faithfulness (context-grounded) menjawab "apakah
klaim didukung oleh konteks yang diberikan ke generator", sedangkan
hallucination (reference-based) menjawab "apakah klaim konsisten dengan
accepted answer SO + pengetahuan umum", independen dari context apa pun
yang (mungkin) dipakai generator. Karena itu retrieved_context, condition,
dan run_id TIDAK PERNAH dikirim ke judge-v1 -- hanya title, tags, body
(dari parquet via _judge_common.load_question_bodies), reference answer,
dan candidate (llm_answer, sitasinya di-strip oleh
judge_prompt_v1.blind_candidate()).

KETERBATASAN DATA: record run manapun TIDAK punya field run_id/condition/
fusion_mode sendiri. run_id, condition, run_label (mis. "C-uniform" vs
"C-trust_weighted" -- DUA VARIAN Kondisi C yang HARUS tetap terpisah utk
analisis ablasi), fusion_mode, dan retrieval_version SEMUA diambil dari
_run_metadata.py, yang membaca logs/run_history.jsonl per kondisi --
SUMBER YANG SAMA dibaca backend/app/services/history_service.py utk
halaman Run Comparison dashboard -- di-join lewat `output_path`. TIDAK
PERNAH ditebak dari nama file hasil semata (nama file hanya dipakai utk
menentukan kondisi mana yg run_history.jsonl harus dicek). Kalau suatu
file hasil tidak punya entri run_history.jsonl yang cocok,
_run_metadata.RunMetadataNotFoundError di-raise dan SCRIPT INI BERHENTI
(exit 1) -- tidak pernah diam-diam fallback ke tebakan nama file.

ITEM YANG GAGAL SETELAH SEMUA RETRY tidak ditulis ke --out (supaya re-run
otomatis mencoba ulang lewat mekanisme resume yang sama) -- ditulis ke
file TERPISAH <out>_failures.jsonl sebagai gantinya.

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 llm_judge_hallucination_v1.py \\
        --run-files ../a_pure_llm/results/condition_a_openai_gpt-4o-mini_n10_seed42.jsonl \\
                    ../b_rag/results/condition_b_openai_gpt-4o-mini_n10_seed42.jsonl \\
        --judge both --limit 3 --out results/jv1_smoke_test.jsonl

SETUP .env (DEEPINFRA_API_KEY, OPENAI_API_KEY) -- lihat judge_clients.py.
QUESTIONS_PARQUET juga harus diset (sama seperti kondisi A/B/C/D dan
llm_judge_context_relevance.py).
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

from _judge_common import append_manifest, html_to_text, load_question_bodies  # noqa: E402
from _run_metadata import RunMetadataNotFoundError, resolve_run_metadata  # noqa: E402
from judge_clients import get_judge_client  # noqa: E402
from llm.manifest import write_manifest  # noqa: E402
from judge_prompt_v1 import BLINDING_VERSION, PROMPT_VERSION, build_messages, parse_judgment  # noqa: E402

log = print

RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS = 6
_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# File -> run metadata + item loading
# ---------------------------------------------------------------------------

def format_tags(raw_tags: str | None) -> str:
    """SORD tags are stored like '<c#><uwp>' -- turn into 'c#, uwp' for
    the judge prompt's 'Tags:' line."""
    if not raw_tags:
        return ""
    return ", ".join(re.findall(r"<([^>]+)>", raw_tags)) or raw_tags


def load_items(run_files: list[str], limit: int | None) -> list[dict]:
    """Read every item from the given run files, each tagged with its
    AUTHORITATIVE run metadata (run_id, condition, run_label, fusion_mode,
    retrieval_version, source_file) from _run_metadata.resolve_run_metadata()
    -- never guessed from the filename alone. --limit applied PER FILE (so
    --limit 3 with 2 files gives up to 6 items total, matching the "smoke
    test per run file" framing in the task spec).

    Raises RunMetadataNotFoundError (uncaught, by design -- main() lets it
    propagate and exits) if any run file has no matching run_history.jsonl
    entry, per the "stop and tell me" requirement -- this must never
    silently fall back to filename-only inference.
    """
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
        # Refuse to judge a file that mixes answers from genuinely
        # different configs/prompt_versions under one filename -- see
        # llm.manifest.assert_single_config_hash's docstring. Raises
        # MixedConfigHashError (uncaught, by design, same as
        # RunMetadataNotFoundError above) rather than silently judging a
        # blend no single config_hash/run_label actually describes.
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


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _resume_key(r: dict) -> tuple:
    """(run_id, config_hash, question_id, judge_id, judge_model,
    prompt_version, blinding_version) -- ALL seven must match for a
    record to count as "already done" for a given item. config_hash and
    judge_model are included (not just run_id/judge_id) so a generation
    re-run under a NEW config_hash, or a judge model swapped under the
    same judge_id (e.g. after a role change), is never silently treated
    as already-judged. blinding_version is included so a future blinding
    scheme change also forces a re-judge rather than reusing a
    differently-blinded record's "done" status."""
    return (
        r.get("run_id"), r.get("config_hash"), r.get("question_id"), r.get("judge_id"),
        r.get("judge_model"), r.get("prompt_version"), r.get("blinding_version"),
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


# ---------------------------------------------------------------------------
# One (item, judge) call -- retry with exponential backoff + jitter
# ---------------------------------------------------------------------------

REQUEST_TEMPERATURE = 0.0


def call_judge_with_retry(client, config, messages: list[dict]) -> tuple[dict | None, list[dict]]:
    """Returns (response_payload, attempts). response_payload is None
    only if every attempt failed -- caller logs the failure and moves on,
    it never raises (so one bad item can't crash the batch). `attempts`
    is a list of {"attempt": int, "error": str|None, "status_code":
    int|None} for EVERY attempt made (not just the last) -- Step 5 data-
    retention requirement: nothing about a retried call is dropped, even
    on eventual success.

    response_payload, when present, carries the FULL raw response
    metadata (not just the parsed content) per the same requirement:
    response id/model/created/system_fingerprint/finish_reason/usage,
    plus the request params actually sent (model, temperature,
    max_tokens if set) -- enough to audit exactly what was asked for and
    what came back, not just the judge's parsed label.
    """
    import openai

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,  # covers 5xx
    )

    attempts: list[dict] = []
    for attempt in range(1, RETRYABLE_EXCEPTIONS_MAX_ATTEMPTS + 1):
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=REQUEST_TEMPERATURE,
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
                "response_created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": response.choices[0].finish_reason,
                "usage_full": usage.model_dump() if usage else None,
                "request_params": {"model": config.model, "temperature": REQUEST_TEMPERATURE},
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
            # Non-retryable (e.g. 400 bad request, auth error) -- fail fast, no point retrying.
            status_code = getattr(e, "status_code", None)
            error_text = f"{type(e).__name__}: {e}"
            attempts.append({"attempt": attempt, "error": error_text, "status_code": status_code})
            break

    return None, attempts


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
        # Step 5 data retention: exact messages sent are NOT stored verbatim
        # (they're large and fully reconstructible), but a SHA-256 + the
        # rebuild fields (run_id/source_file/question_id -- already above --
        # plus judge_id/prompt_version/blinding_version) let anyone rebuild
        # them deterministically via load_items()+the same body lookup and
        # verify the hash matches, proving nothing was silently altered.
        "messages_sha256": _messages_sha256(messages),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }

    if payload is None:
        base.update({
            "label": None, "derived_label": None, "consistent": None,
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

    parsed = parse_judgment(payload["raw"])
    base.update({
        "label": parsed.get("label"),
        "derived_label": parsed.get("derived_label"),
        "consistent": parsed.get("consistent"),
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

    expected = getattr(config, "expected_served_model", None)
    served_model_mismatch = bool(expected) and payload["response_model"] != expected
    base["served_model_mismatch"] = served_model_mismatch
    if served_model_mismatch:
        log(f"      [WARN] judge '{judge_id}': response_model='{payload['response_model']}' "
            f"differs from registry's expected_served_model='{expected}' -- the provider may have "
            f"changed which model/variant it actually serves for this judge_id. Investigate before "
            f"trusting further records from this judge_id.")
    return base


def write_record(output_path: Path, record: dict) -> None:
    with _WRITE_LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_failed_keys(failures_path: Path) -> set[tuple]:
    """Keys of every item EVER recorded in <out>_failures.jsonl (across
    all past runs) -- used only to detect when a retry succeeds, so that
    can be logged to a resolution log. The failures file itself is never
    read for resume/skip purposes (that's output_path's job) and is NEVER
    deleted or rewritten -- Step 5 data retention: a failure record stays
    forever, even after a later success."""
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


def append_resolution_record(resolved_path: Path, record: dict) -> None:
    """A previously-failed (run_id, question_id, judge_id, prompt_version)
    succeeded on this run -- note it here, in a SEPARATE append-only log,
    rather than touching/deleting the original failure record in
    <out>_failures.jsonl (which stays as permanent evidence it happened)."""
    resolution = {
        "run_id": record["run_id"], "question_id": record["question_id"],
        "judge_id": record["judge_id"], "prompt_version": record["prompt_version"],
        "resolved": True, "resolved_at_utc": record["timestamp_utc"],
    }
    write_record(resolved_path, resolution)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(items: list[dict], judge_ids: list[str], questions_parquet: str,
              output_path: Path, failures_path: Path, workers: int,
              resolved_path: Path | None = None, on_progress=None,
              check_cancel=None) -> dict:
    """`on_progress(dict)` -- called after EVERY task (success or failure),
    with {"judge_id", "question_id", "run_id", "status": "done"|"failed",
    "done_count", "total_count"} -- optional, used by the dashboard
    backend to push live progress/throughput/ETA. `check_cancel()` -> bool
    -- checked before submitting each task; already-submitted tasks still
    finish and get written normally (never leaves a half-written record).
    Both optional so the CLI (main()) behavior is unchanged."""
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
    label_counts: dict[str, dict[str, int]] = {}
    token_totals: dict[str, int] = {jid: 0 for jid in judge_ids}

    def _run(task):
        item, judge_id, body_text, reference_text = task
        client, config = clients[judge_id]
        return judge_one(item, judge_id, client, config, body_text, reference_text)

    total_count = len(tasks)
    done_count = 0
    if check_cancel is not None and check_cancel():
        tasks = []  # cancelled before any work started -- nothing to submit

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, t) for t in tasks]
        for future in as_completed(futures):
            record = future.result()
            done_count += 1
            if record["call_failed"]:
                # NEVER written to output_path -- its (run_id, question_id,
                # judge_id, prompt_version) key must stay absent from
                # load_already_done() so the NEXT run retries it instead of
                # treating a failed call as "done". Logged to a separate
                # file instead, purely for visibility/debugging.
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
    log(f"Label distribution per run_label: {json.dumps(summary['label_distribution_per_run_label'], indent=2)}")
    log(f"Total tokens per judge: {summary['total_tokens_per_judge']}")

    finished_at = datetime.now(timezone.utc).isoformat()
    append_manifest("judge_v1_run_history.jsonl", {
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
        run_label=None,  # a judge run can cover MULTIPLE run_labels (one per --run-files entry) -- see summary
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
