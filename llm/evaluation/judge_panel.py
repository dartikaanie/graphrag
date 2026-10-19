"""
judge_panel.py
==============
Runner for the three-judge LLM-as-Judge panel (judge_panel_config.py;
docs/DECISION_C_SCORING.md, Amendment 12). Every judge gets the SAME
messages: judge_prompt_v2_2's frozen prompt, input template, blind-v3 and
context block, built once per item. Each response is parsed with
judge_prompt_v2_2.parse_judgment_v2_2 (same normalization rules, not
widened); the per-judge label is the parser's derived label, never the
judge's self-reported one. The panel label is judge_panel_aggregate.

Per judge and item: up to MAX_RETRIES retries (backoff) on a network error
or a parse failure; a non-retryable API error (e.g. 400/401) fails at once.
DeepSeek-V3.2 is sent reasoning_effort="none"; every response is checked
for reasoning content (reasoning_content / reasoning field, <think> block).

Outputs (all append-only except the derived panel file):
  <out>.jsonl            per-judge records; the resume cache, keyed by
                         (item_id, judge_model, prompt_version)
  <out>_failures.jsonl   per-judge records still failed after retries
                         (retried on the next run)
  <out>_panel.jsonl      panel records, rebuilt from <out>.jsonl each run

Guards:
  - the Amendment 11 freeze guard (judge_versions.check_run_allowed):
    official test-sample runs are refused until the judge is frozen;
  - the calibration guard (_judge_common.assert_calibration_judging_allowed);
  - annotation exclusion: items in the calibration key or the
    confirmation key are never judged, and for calibration run files no
    item whose QUESTION is in the calibration sample under any condition
    (the annotator must not see judge reasoning on a question they label).
    If a calibration run file is given, the verified calibration key must
    be available, otherwise the run is refused;
  - cost: the projected cost is printed before any call; --max-cost-usd
    aborts when the projection exceeds it.

CARA PAKAI (from llm/evaluation)
    python3 judge_panel.py --run-files <run>.jsonl ... --out results/panel_<name>.jsonl --max-cost-usd 1.00
    python3 judge_panel.py --run-files <run>.jsonl --question-ids 123 456 --out x.jsonl --dry-run
    python3 judge_panel.py --run-files <a>.jsonl <b>.jsonl --items 0:123 1:456 --out x.jsonl --max-cost-usd 0.10
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judge_binary import derive_hallucinated  # noqa: E402
from judge_panel_aggregate import UNRESOLVED, UNSPECIFIED, aggregate_labels  # noqa: E402
from judge_panel_config import (MAX_RETRIES, MAX_TOKENS, PANEL_ID, PANEL_JUDGES, TEMPERATURE,  # noqa: E402
                                PanelJudge, judge_cost_usd, panel_public_view)
from judge_prompt_v2_2 import (BLINDING_VERSION, PARSE_VERSION, PROMPT_VERSION, _load_json,  # noqa: E402
                               build_messages, parse_judgment_v2_2, prepare_judge_input)

log = print

CACHE_KEY_FIELDS = ("item_id", "judge_model", "prompt_version")
# Rough chars-per-token for the cost projection (deliberately low -> projection on the high side).
CHARS_PER_TOKEN_EST = 3.0
_PARSE_FIELDS = ("label", "derived_label", "judge_reported_label", "consistent", "answer_attempted", "completeness",
                 "attempted_claims_conflict", "claims", "reasoning", "approach_match", "faithfulness",
                 "citation_support", "n_cited_claims", "n_claims", "n_error_claims",
                 "n_error_claims_without_severity", "non_error_claims_over_cap")


class PanelGuardError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def item_id_of(run_id: str, question_id) -> str:
    return f"{run_id}:{int(question_id)}"


def parse_item_specs(specs: list[str] | None, n_run_files: int) -> dict[int, set[int]] | None:
    """--items "<run-file index>:<question_id>" -> {run file index: {question_id}}."""
    if not specs:
        return None
    out: dict[int, set[int]] = {}
    for spec in specs:
        idx, _, qid = spec.partition(":")
        if not (idx.isdigit() and qid.isdigit()) or int(idx) >= n_run_files:
            raise ValueError(f"--items {spec!r}: expected <run-file index 0..{n_run_files - 1}>:<question_id>")
        out.setdefault(int(idx), set()).add(int(qid))
    return out


def load_panel_items(run_files: list[str], question_ids: list[int] | None = None,
                     items_by_file: dict[int, set[int]] | None = None) -> list[dict]:
    """Items of the given run files, as the judge-v2.2 runner loads them, plus item_id. `question_ids` selects
    in every file, `items_by_file` per file. The calibration guard is applied separately in main() so its
    result can be recorded."""
    from _run_metadata import resolve_run_metadata
    from llm.manifest import assert_single_config_hash
    from llm_judge_hallucination_v2_1 import format_tags

    wanted = {int(q) for q in question_ids} if question_ids else None
    items = []
    for idx, path in enumerate(run_files):
        if items_by_file is not None and idx not in items_by_file:
            continue
        meta = resolve_run_metadata(path)
        records = [json.loads(line) for line in open(path) if line.strip()]
        assert_single_config_hash(records, source_label=path)
        for r in records:
            if wanted is not None and int(r["question_id"]) not in wanted:
                continue
            if items_by_file is not None and int(r["question_id"]) not in items_by_file[idx]:
                continue
            items.append({
                **meta, "item_id": item_id_of(meta["run_id"], r["question_id"]),
                "question_id": int(r["question_id"]), "title": r.get("title", ""),
                "tags": format_tags(r.get("tags")), "ground_truth_answer_html": r.get("ground_truth_answer", "") or "",
                "candidate": r.get("llm_answer", "") or "", "retrieved_context": r.get("retrieved_context") or [],
                "config_hash": r.get("config_hash"),
            })
    return items


def annotation_exclusions(run_files: list[str], key_dir: Path | None = None, calibration_paths: set | None = None,
                          load_key=None) -> dict:
    """{"pairs": {(run_id, question_id)}, "question_ids": {question_id}, "keys": {...}} to exclude.
    pairs: every item of the calibration and confirmation keys (whichever are exported);
    question_ids: every question of the calibration sample, applied only to calibration run files.
    Refuses (PanelGuardError) if a calibration run file is given and the calibration key cannot be verified."""
    from annotation_tool import AnnotationError, key_dir_from, load_verified_key

    load_key = load_key or load_verified_key
    key_dir = key_dir or key_dir_from(None)
    if calibration_paths is None:
        from _judge_common import load_calibration_registry
        from judge_versions import REPO_ROOT
        calibration_paths = {str((REPO_ROOT / r["output_path"]).resolve())
                             for r in load_calibration_registry().get("runs", [])}
    uses_calibration = any(str(Path(f).resolve()) in calibration_paths for f in run_files)

    pairs, cal_qids, keys = set(), set(), {}
    for purpose in ("calibration", "confirmation"):
        try:
            key = load_key(purpose, key_dir)
        except AnnotationError as e:
            if purpose == "calibration" and uses_calibration:
                raise PanelGuardError(f"calibration run files given but the calibration key cannot be verified "
                                      f"({e}); refusing, the 60 annotation items could not be excluded")
            keys[purpose] = f"not available: {e}"
            continue
        items = key.get("items") or []
        pairs |= {(k["run_id"], int(k["question_id"])) for k in items}
        if purpose == "calibration":
            cal_qids = {int(k["question_id"]) for k in items}
        keys[purpose] = f"verified, {len(items)} items"
    return {"pairs": pairs, "question_ids": cal_qids if uses_calibration else set(), "keys": keys,
            "calibration_paths": calibration_paths}


def apply_exclusions(items: list[dict], excl: dict, explicit_question_ids: bool) -> tuple[list[dict], dict]:
    """Drops excluded items (counts only, never ids, are reported). An explicitly requested question that is
    excluded is an error, not a silent skip."""
    kept, n_pair, n_question = [], 0, 0
    for it in items:
        is_cal_file = str(Path(it["source_file"]).resolve()) in excl["calibration_paths"]
        if (it["run_id"], it["question_id"]) in excl["pairs"]:
            n_pair += 1
        elif is_cal_file and it["question_id"] in excl["question_ids"]:
            n_question += 1
        else:
            kept.append(it)
    if explicit_question_ids and (n_pair or n_question):
        raise PanelGuardError(f"{n_pair + n_question} requested item(s) belong to the human annotation sample "
                              f"(or share its questions); choose other question ids")
    return kept, {"excluded_annotation_items": n_pair, "excluded_same_question_as_calibration": n_question}


# ---------------------------------------------------------------------------
# One judge call
# ---------------------------------------------------------------------------

def reasoning_present(message: dict) -> bool:
    """True if the response carries any reasoning/thinking content."""
    for k in ("reasoning_content", "reasoning"):
        if message.get(k):
            return True
    return "<think>" in (message.get("content") or "")


def cache_key(r: dict) -> tuple:
    return tuple(r.get(f) for f in CACHE_KEY_FIELDS)


def _messages_sha256(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _retryable_errors():
    import openai
    return (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)


def call_judge(client, judge: PanelJudge, messages: list[dict], has_context: bool, offered_items,
               max_retries: int = MAX_RETRIES, sleep=time.sleep) -> dict:
    """At most 1 + max_retries attempts; retries on a retryable API error or a parse failure. Returns the
    response/parse part of a per-judge record (call_failed / parse_ok say how it ended)."""
    retryable = _retryable_errors()
    attempts, last = [], None
    for attempt in range(1, max_retries + 2):
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=judge.model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                **({"extra_body": judge.extra_body} if judge.extra_body else {}))
            latency = round(time.monotonic() - start, 3)
        except retryable as e:
            attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}",
                             "status_code": getattr(e, "status_code", None)})
            if attempt <= max_retries:
                sleep(2 ** attempt + random.uniform(0, 1))
            continue
        except Exception as e:  # non-retryable (400, auth, ...): fail fast
            attempts.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}",
                             "status_code": getattr(e, "status_code", None)})
            break
        message = response.choices[0].message.model_dump()
        usage = response.usage.model_dump() if response.usage else {}
        raw = message.get("content")
        parsed = parse_judgment_v2_2(raw, has_context, offered_items)
        parse_ok = not parsed.get("parse_error", False)
        last = {"message": message, "usage": usage, "raw": raw, "parsed": parsed, "parse_ok": parse_ok,
                "latency_s": latency, "response": response}
        attempts.append({"attempt": attempt, "error": None if parse_ok else "parse_error", "status_code": None,
                         "finish_reason": response.choices[0].finish_reason,
                         "completion_tokens": usage.get("completion_tokens")})
        if parse_ok or attempt > max_retries:
            break
        sleep(2 ** attempt + random.uniform(0, 1))

    if last is None:
        return {"call_failed": True, "parse_ok": False, "error": attempts[-1]["error"] if attempts else None,
                "attempts": attempts, "raw_response": None, "parsed_json": None, "finish_reason": None,
                "prompt_tokens": None, "completion_tokens": None, "usage_full": None, "latency_s": None,
                "response_model": None, "response_id": None, "reasoning_present": None,
                "normalized_fields": [], **{f: None for f in _PARSE_FIELDS}}
    resp, parsed = last["response"], last["parsed"]
    # tokens/cost of ALL successful attempts (a parse-failure retry is paid too)
    return {
        "call_failed": False, "parse_ok": last["parse_ok"], "parse_error": not last["parse_ok"],
        "error": None if last["parse_ok"] else "parse_error", "attempts": attempts,
        "raw_response": last["raw"], "parsed_json": _load_json(last["raw"]) if last["raw"] else None,
        "normalized_fields": parsed.get("normalized_fields", []) if last["parse_ok"] else [],
        **{f: (parsed.get(f) if last["parse_ok"] else None) for f in _PARSE_FIELDS},
        "finish_reason": resp.choices[0].finish_reason,
        "prompt_tokens": last["usage"].get("prompt_tokens"), "completion_tokens": last["usage"].get("completion_tokens"),
        "usage_full": last["usage"], "latency_s": last["latency_s"], "response_model": resp.model,
        "response_id": resp.id, "reasoning_present": reasoning_present(last["message"]),
        "reasoning_content_chars": len(last["message"].get("reasoning_content") or ""),
    }


def judge_record(item: dict, judge: PanelJudge, messages: list[dict], prepared: dict, client,
                 sleep=time.sleep) -> dict:
    out = call_judge(client, judge, messages, prepared["has_context"], prepared["offered_items"], sleep=sleep)
    rec = {
        "panel_id": PANEL_ID, "item_id": item["item_id"], "run_id": item["run_id"],
        "question_id": item["question_id"], "condition": item["condition"], "run_label": item["run_label"],
        "config_hash": item.get("config_hash"), "source_file": item["source_file"],
        "judge_id": judge.judge_id, "judge_model": judge.model, "provider": judge.provider,
        "prompt_version": PROMPT_VERSION, "blinding_version": BLINDING_VERSION, "parse_version": PARSE_VERSION,
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "request_extra_body": judge.extra_body,
        "thinking_disabled_requested": judge.thinking_disabled,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "messages_sha256": _messages_sha256(messages),
        "has_context": prepared["has_context"], "n_context_items": prepared["n_context_items"],
        "citation_map": prepared["citation_map"], **out,
    }
    # thinking_disabled: requested off AND no reasoning content observed in the response
    rec["thinking_disabled"] = bool(judge.thinking_disabled and rec.get("reasoning_present") is False)
    if rec.get("reasoning_present"):
        log(f"      [WARN] {judge.judge_id}: reasoning content in the response for {item['item_id']}")
    rec["served_model_mismatch"] = (rec["response_model"] is not None
                                    and rec["response_model"] != judge.expected_served_model)
    if rec["served_model_mismatch"]:
        log(f"      [WARN] {judge.judge_id}: response.model {rec['response_model']!r} != expected "
            f"{judge.expected_served_model!r}")
    rec["hallucinated"] = derive_hallucinated({**rec, "parse_error": not rec["parse_ok"]})
    paid = [a for a in rec["attempts"] if a.get("completion_tokens") is not None]
    rec["n_paid_attempts"] = len(paid)
    rec["estimated_cost_usd"] = judge_cost_usd(judge, rec["prompt_tokens"], rec["completion_tokens"])
    if rec["estimated_cost_usd"] is not None and len(paid) > 1:  # earlier parse-failure attempts were paid too
        extra_out = sum(a["completion_tokens"] for a in paid[:-1])
        rec["estimated_cost_usd"] = round(rec["estimated_cost_usd"] + (len(paid) - 1) * rec["prompt_tokens"] / 1e6
                                          * judge.price_per_m_input + extra_out / 1e6 * judge.price_per_m_output, 8)
    rec["provider_estimated_cost_usd"] = (rec.get("usage_full") or {}).get("estimated_cost")
    return rec


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()


def append_jsonl(path: Path, record: dict) -> None:
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def failures_path_of(out: Path) -> Path:
    return out.with_name(out.stem + "_failures.jsonl")


def panel_path_of(out: Path) -> Path:
    return out.with_name(out.stem + "_panel.jsonl")


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def project_cost(tasks: list[tuple], done_records: list[dict]) -> dict:
    """tasks: [(item, judge, messages)]. Input tokens are estimated from the message length; output tokens are
    the mean completion tokens of this judge's completed records, or MAX_TOKENS (upper bound) when none."""
    per_judge = defaultdict(lambda: {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    mean_out = {}
    for j in PANEL_JUDGES:
        outs = [r["completion_tokens"] for r in done_records
                if r.get("judge_model") == j.model and r.get("completion_tokens") is not None]
        mean_out[j.judge_id] = (sum(outs) / len(outs)) if outs else MAX_TOKENS
    for item, judge, messages in tasks:
        tin = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN_EST
        tout = mean_out[judge.judge_id]
        p = per_judge[judge.judge_id]
        p["n_calls"] += 1
        p["input_tokens"] += tin
        p["output_tokens"] += tout
        p["cost_usd"] += tin / 1e6 * judge.price_per_m_input + tout / 1e6 * judge.price_per_m_output
    total = sum(p["cost_usd"] for p in per_judge.values())
    return {"per_judge": {k: {**v, "input_tokens": round(v["input_tokens"]), "output_tokens": round(v["output_tokens"]),
                              "cost_usd": round(v["cost_usd"], 5)} for k, v in per_judge.items()},
            "total_usd": round(total, 5), "worst_case_with_retries_usd": round(total * (1 + MAX_RETRIES), 5),
            "output_token_basis": {k: ("mean of completed records" if v != MAX_TOKENS else "max_tokens (upper bound)")
                                   for k, v in mean_out.items()}}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def build_panel_records(items: list[dict], judge_records: list[dict]) -> list[dict]:
    """One panel record per item. A judge without a usable record (none, or parse failure) counts as missing."""
    by = {}
    for r in judge_records:
        if r.get("panel_id") == PANEL_ID and r.get("prompt_version") == PROMPT_VERSION:
            by[(r["item_id"], r["judge_model"])] = r
    out = []
    for it in items:
        labels = {}
        for j in PANEL_JUDGES:
            r = by.get((it["item_id"], j.model))
            labels[j.judge_id] = r["label"] if r and r.get("parse_ok") and not r.get("call_failed") else None
        out.append({"item_id": it["item_id"], "run_id": it["run_id"], "question_id": it["question_id"],
                    "condition": it["condition"], "run_label": it["run_label"], "prompt_version": PROMPT_VERSION,
                    **aggregate_labels(labels, PANEL_ID)})
    return out


def summarize(judge_records: list[dict], panel_records: list[dict], new_records: list[dict]) -> dict:
    per_judge = {}
    for j in PANEL_JUDGES:
        rs = [r for r in judge_records if r["judge_model"] == j.model]
        new = [r for r in new_records if r["judge_model"] == j.model]
        per_judge[j.judge_id] = {
            "model": j.model, "n_records": len(rs), "parse_ok": f"{sum(1 for r in rs if r.get('parse_ok'))}/{len(rs)}",
            "parse_ok_this_run": f"{sum(1 for r in new if r.get('parse_ok'))}/{len(new)}",
            "finish_reason_this_run": dict(Counter(r.get("finish_reason") for r in new)),
            "response_models": dict(Counter(r.get("response_model") for r in rs)),
            "finish_reason": dict(Counter(r.get("finish_reason") for r in rs)),
            "labels": dict(Counter(r.get("label") for r in rs)),
            "unspecified_votes": sum(1 for r in rs if r.get("label") == UNSPECIFIED),
            # items still without a usable result from this judge after this run (in <out>_failures.jsonl)
            "final_failures": sum(1 for p in panel_records if p["judge_labels"].get(j.judge_id) is None),
            "reasoning_present": sum(1 for r in rs if r.get("reasoning_present")),
            "served_model_mismatch": sum(1 for r in rs if r.get("served_model_mismatch")),
            "normalized_fields": dict(Counter(f for r in rs for f in (r.get("normalized_fields") or []))),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rs),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in rs),
            "cost_usd_all_records": round(sum(r.get("estimated_cost_usd") or 0 for r in rs), 6),
            "cost_usd_this_run": round(sum(r.get("estimated_cost_usd") or 0 for r in new), 6),
            "provider_cost_usd_this_run": round(sum(r.get("provider_estimated_cost_usd") or 0 for r in new), 6),
        }
    by_cond = defaultdict(Counter)
    for p in panel_records:
        by_cond[p["run_label"]][p["panel_label"]] += 1
    return {
        "per_judge": per_judge,
        "panel_labels_by_run_label": {k: dict(v) for k, v in by_cond.items()},
        "unresolved_by_run_label": {k: v.get(UNRESOLVED, 0) for k, v in by_cond.items()},
        "unspecified_by_run_label": {k: v.get(UNSPECIFIED, 0) for k, v in by_cond.items()},
        "n_panel_incomplete": sum(1 for p in panel_records if p["panel_incomplete"]),
        "n_unanimous": sum(1 for p in panel_records if p["unanimous"]),
        "n_items": len(panel_records),
        "actual_cost_usd_this_run": round(sum(r.get("estimated_cost_usd") or 0 for r in new_records), 6),
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def plan_tasks(items: list[dict], bodies_text: dict, done: set) -> tuple[list[tuple], dict, int]:
    """[(item, judge, messages)] for every (item, judge) not in the cache; {item_id: prepared}; n cached."""
    from _judge_common import html_to_text
    from llm_judge_hallucination_v2_2 import has_context_for

    tasks, prepared_by_item, n_cached = [], {}, 0
    for it in items:
        prepared = prepare_judge_input(it["candidate"], it["retrieved_context"], has_context_for(it["condition"]))
        prepared_by_item[it["item_id"]] = prepared
        body = bodies_text.get(it["question_id"]) or "[question body not found in parquet]"
        messages = build_messages(it["title"], body, it["tags"], html_to_text(it["ground_truth_answer_html"]),
                                  it["candidate"], prepared=prepared)
        for j in PANEL_JUDGES:
            if (it["item_id"], j.model, PROMPT_VERSION) in done:
                n_cached += 1
            else:
                tasks.append((it, j, messages))
    return tasks, prepared_by_item, n_cached


def run_tasks(tasks: list[tuple], prepared_by_item: dict, clients: dict, out: Path, concurrency: int,
              sleep=time.sleep) -> list[dict]:
    """Item-major submission, so an item's three judges run concurrently (bounded by `concurrency`)."""
    new = []

    def _one(task):
        it, j, messages = task
        return judge_record(it, j, messages, prepared_by_item[it["item_id"]], clients[j.judge_id], sleep=sleep)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for rec in pool.map(_one, tasks):
            usable = rec["parse_ok"] and not rec["call_failed"]
            append_jsonl(out if usable else failures_path_of(out), rec)
            new.append(rec)
            log(f"  {rec['item_id']} {rec['judge_id']:<8} label={rec['label']} parse_ok={rec['parse_ok']} "
                f"finish={rec['finish_reason']} tokens={rec['prompt_tokens']}/{rec['completion_tokens']}")
    return new


def make_clients() -> dict:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    clients = {}
    for j in PANEL_JUDGES:
        key = os.getenv(j.api_key_env)
        if not key:
            raise RuntimeError(f"{j.api_key_env} not set (repo-root .env)")
        clients[j.judge_id] = OpenAI(api_key=key, base_url=j.base_url)
    return clients


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-files", nargs="+", required=True)
    parser.add_argument("--question-ids", nargs="+", type=int, default=None,
                        help="Only these question ids (in every run file given)")
    parser.add_argument("--items", nargs="+", default=None,
                        help="Only these items: <run-file index>:<question_id> (index into --run-files)")
    parser.add_argument("--out", required=True, help="Per-judge JSONL (the resume cache)")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent API calls")
    parser.add_argument("--max-cost-usd", type=float, required=True,
                        help="Abort before any call if the projected cost exceeds this")
    parser.add_argument("--questions-parquet", default=None)
    parser.add_argument("--key-dir", default=None, help="Annotation key directory (outside the repo)")
    parser.add_argument("--allow-unannotated-calibration", action="store_true",
                        help="Calibration guard override (logged); annotation items stay excluded regardless")
    parser.add_argument("--dry-run", action="store_true", help="Plan and project the cost; no API call")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from _judge_common import CalibrationGuardError, append_manifest, assert_calibration_judging_allowed
    from _judge_common import html_to_text, load_question_bodies
    from annotation_tool import key_dir_from
    from judge_panel_config import DEFAULT_CONCURRENCY
    from judge_versions import JudgeVersionError, check_run_allowed

    concurrency = args.concurrency or DEFAULT_CONCURRENCY
    questions_parquet = args.questions_parquet or os.getenv("QUESTIONS_PARQUET")
    for p in args.run_files:
        if not Path(p).exists():
            log(f"[ERROR] run file not found: {p}")
            return 1
    try:
        version_purpose = check_run_allowed(PROMPT_VERSION, args.run_files, log)
        guard = assert_calibration_judging_allowed(args.run_files, args.allow_unannotated_calibration, log)
        excl = annotation_exclusions(args.run_files, key_dir_from(args.key_dir))
        items_by_file = parse_item_specs(args.items, len(args.run_files))
        items = load_panel_items(args.run_files, args.question_ids, items_by_file)
        items, excl_counts = apply_exclusions(items, excl, explicit_question_ids=bool(args.question_ids or args.items))
    except (JudgeVersionError, CalibrationGuardError, PanelGuardError, ValueError) as e:
        log(f"[ERROR] {e}")
        return 1
    log(f"[annotation exclusion] keys: {excl['keys']}; {excl_counts}")
    if not items:
        log("[ERROR] no items to judge after exclusions")
        return 1
    if not questions_parquet or not Path(questions_parquet).exists():
        log(f"[ERROR] questions parquet not available ({questions_parquet!r}); the judge input needs the question "
            f"body, refusing to judge with a different input")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_records = read_jsonl(out)
    done = {cache_key(r) for r in done_records}
    bodies = {q: html_to_text(b) for q, b in
              load_question_bodies(questions_parquet, sorted({it["question_id"] for it in items})).items()}
    missing_bodies = sorted({it["question_id"] for it in items} - set(bodies))
    if missing_bodies:
        log(f"[ERROR] {len(missing_bodies)} question bodies not found in the parquet; refusing")
        return 1
    tasks, prepared_by_item, n_cached = plan_tasks(items, bodies, done)
    projection = project_cost(tasks, done_records)
    log(f"\n[plan] {len(items)} items x {len(PANEL_JUDGES)} judges: {len(tasks)} calls to make, {n_cached} cached")
    log(f"[cost] projected ${projection['total_usd']:.4f} (worst case with all retries "
        f"${projection['worst_case_with_retries_usd']:.4f}); limit ${args.max_cost_usd:.4f}")
    log(json.dumps(projection["per_judge"], indent=2))
    if projection["total_usd"] > args.max_cost_usd:
        log(f"[ABORT] projected cost ${projection['total_usd']:.4f} exceeds --max-cost-usd {args.max_cost_usd}")
        return 2
    if args.dry_run:
        return 0

    started = datetime.now(timezone.utc).isoformat()
    new = run_tasks(tasks, prepared_by_item, make_clients(), out, concurrency) if tasks else []
    judge_records = read_jsonl(out)
    item_ids = {it["item_id"] for it in items}
    panel = build_panel_records(items, [r for r in judge_records if r["item_id"] in item_ids])
    panel_path = panel_path_of(out)
    with open(panel_path, "w", encoding="utf-8") as f:
        for p in panel:
            f.write(json.dumps(p) + "\n")
    summary = summarize([r for r in judge_records if r["item_id"] in item_ids], panel, new)
    log(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")
    append_manifest("judge_panel_run_history.jsonl", {
        "run_started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_files": args.run_files, "question_ids_given": bool(args.question_ids), "items_given": args.items,
        "n_items": len(items),
        "out": str(out), "panel_out": str(panel_path), "panel": panel_public_view(), "concurrency": concurrency,
        "max_cost_usd": args.max_cost_usd, "projection": projection, "calibration_guard": guard,
        "annotation_exclusion": {"keys": excl["keys"], **excl_counts}, "version_purpose": version_purpose,
        "n_calls": len(new), "n_cached": n_cached, "actual_cost_usd": summary["actual_cost_usd_this_run"],
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
