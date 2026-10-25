"""
judge_panel.py
==============
Runner for the three-judge LLM-as-Judge panel (judge_panel_config.py;
docs/DECISION_C_SCORING.md, Amendment 12). Every judge gets the SAME
messages for the selected --judge-version (judge-v2.2, the default, or
judge-v2.3): that version's prompt, the v2.2 input template, blind-v3 and
context block, built once per item, and the same request parameters
(judge-v2.3 adds JSON mode). Each response is parsed with that version's
parser; the per-judge label is the parser's derived label, never the
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judge_binary import derive_hallucinated  # noqa: E402
from llm import archive_paths  # noqa: E402
from judge_panel_aggregate import UNRESOLVED, UNSPECIFIED, aggregate_labels  # noqa: E402
from judge_panel_config import (MAX_RETRIES, PANEL_ID, PANEL_JUDGES, PANELS, SCREENING_BY_ID,  # noqa: E402
                                SCREENING_PANEL_ID, PanelJudge, judge_cost_usd, panel_public_view)
from judge_prompt_v2_2 import _load_json  # noqa: E402

log = print

CACHE_KEY_FIELDS = ("item_id", "judge_model", "prompt_version")


class JudgeVersionSpec:
    """Everything that differs between judge prompt versions. The same spec is used for all three judges."""

    def __init__(self, module, parse, response_format=None):
        self.module = module
        self.prompt_version, self.parse_version = module.PROMPT_VERSION, module.PARSE_VERSION
        self.blinding_version = module.BLINDING_VERSION
        self.temperature, self.max_tokens = module.REQUEST_TEMPERATURE, module.REQUEST_MAX_TOKENS
        self.response_format = response_format
        self.build_messages, self.prepare_judge_input, self.parse = module.build_messages, module.prepare_judge_input, parse


def judge_version_spec(version: str) -> JudgeVersionSpec:
    """judge-v2.2: as in the iteration-1 runs (no JSON mode). judge-v2.3: JSON mode + the pv2.3 parser."""
    if version == "judge-v2.2":
        import judge_prompt_v2_2 as m
        return JudgeVersionSpec(m, m.parse_judgment_v2_2)
    if version == "judge-v2.3":
        import judge_prompt_v2_3 as m
        return JudgeVersionSpec(m, m.parse_judgment_v2_3, m.REQUEST_RESPONSE_FORMAT)
    raise ValueError(f"unknown judge version {version!r}")


JUDGE_VERSION_CHOICES = ("judge-v2.2", "judge-v2.3")
DEFAULT_JUDGE_VERSION = "judge-v2.2"
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
        records = [json.loads(line) for line in open(archive_paths.resolve(path)) if line.strip()]
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


def select_calibration_set(items: list[dict], key: dict) -> list[dict]:
    """--calibration-set: exactly the items of the (verified) calibration key, nothing else."""
    wanted = {(k["run_id"], int(k["question_id"])) for k in key["items"]}
    picked = [it for it in items if (it["run_id"], it["question_id"]) in wanted]
    found = {(it["run_id"], it["question_id"]) for it in picked}
    if len(picked) != len(wanted) or found != wanted:
        raise PanelGuardError(f"calibration set: {len(found)} of the {len(wanted)} key items found in the run files "
                              f"(missing {len(wanted - found)}); give the three calibration run files")
    return picked


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
               max_retries: int = MAX_RETRIES, sleep=time.sleep, spec: JudgeVersionSpec | None = None) -> dict:
    """At most 1 + max_retries attempts; retries on a retryable API error or a parse failure. Returns the
    response/parse part of a per-judge record (call_failed / parse_ok say how it ended)."""
    spec = spec or judge_version_spec(DEFAULT_JUDGE_VERSION)
    retryable = _retryable_errors()
    attempts, last = [], None
    request = {"model": judge.model, "messages": messages, "temperature": spec.temperature,
               "max_tokens": spec.max_tokens}
    if spec.response_format:
        request["response_format"] = spec.response_format
    if judge.extra_body:
        request["extra_body"] = judge.extra_body
    for attempt in range(1, max_retries + 2):
        try:
            start = time.monotonic()
            response = client.chat.completions.create(**request)
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
        parsed = spec.parse(raw, has_context, offered_items)
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
                 sleep=time.sleep, spec: JudgeVersionSpec | None = None, panel_id: str = PANEL_ID) -> dict:
    spec = spec or judge_version_spec(DEFAULT_JUDGE_VERSION)
    out = call_judge(client, judge, messages, prepared["has_context"], prepared["offered_items"], sleep=sleep,
                     spec=spec)
    rec = {
        "panel_id": panel_id, "item_id": item["item_id"], "run_id": item["run_id"],
        "question_id": item["question_id"], "condition": item["condition"], "run_label": item["run_label"],
        "config_hash": item.get("config_hash"), "source_file": item["source_file"],
        "judge_id": judge.judge_id, "judge_model": judge.model, "provider": judge.provider,
        "prompt_version": spec.prompt_version, "blinding_version": spec.blinding_version,
        "parse_version": spec.parse_version, "temperature": spec.temperature, "max_tokens": spec.max_tokens,
        "request_response_format": spec.response_format, "request_extra_body": judge.extra_body,
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

def project_cost(tasks: list[tuple], done_records: list[dict], max_tokens: int | None = None, judges=None,
                 basis_records: list[dict] | None = None) -> dict:
    """tasks: [(item, judge, messages)]. Input tokens: the actual prompt tokens recorded for the identical messages
    (same messages_sha256) in `basis_records`, largest over the recorded judges, else estimated from the message
    length. Output tokens: the mean completion tokens of this judge's completed records, or MAX_TOKENS (upper
    bound) when none."""
    max_tokens = max_tokens or judge_version_spec(DEFAULT_JUDGE_VERSION).max_tokens
    per_judge = defaultdict(lambda: {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
    measured_in = {}
    for r in basis_records or []:
        if r.get("messages_sha256") and r.get("prompt_tokens"):
            measured_in[r["messages_sha256"]] = max(measured_in.get(r["messages_sha256"], 0), r["prompt_tokens"])
    n_measured = 0
    mean_out = {}
    out_basis = {}
    for j in judges or PANEL_JUDGES:
        for source, recs in (("mean of completed records", done_records), ("mean of basis records", basis_records or [])):
            outs = [r["completion_tokens"] for r in recs
                    if r.get("judge_model") == j.model and r.get("completion_tokens") is not None]
            if outs:
                mean_out[j.judge_id], out_basis[j.judge_id] = sum(outs) / len(outs), source
                break
        else:
            mean_out[j.judge_id], out_basis[j.judge_id] = max_tokens, "max_tokens (upper bound)"
    for item, judge, messages in tasks:
        sha = _messages_sha256(messages)
        n_measured += sha in measured_in
        tin = measured_in.get(sha) or sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN_EST
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
            "output_token_basis": out_basis,
            "input_token_basis": f"{n_measured}/{len(tasks)} calls measured (same messages in basis records), "
                                 f"rest estimated from length"}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def build_panel_records(items: list[dict], judge_records: list[dict], prompt_version: str = DEFAULT_JUDGE_VERSION,
                        judges=None, panel_id: str = PANEL_ID) -> list[dict]:
    """One panel record per item. A judge without a usable record (none, or parse failure) counts as missing."""
    by = {}
    for r in judge_records:
        if r.get("panel_id") == panel_id and r.get("prompt_version") == prompt_version:
            by[(r["item_id"], r["judge_model"])] = r
    out = []
    for it in items:
        labels = {}
        for j in judges or PANEL_JUDGES:
            r = by.get((it["item_id"], j.model))
            labels[j.judge_id] = r["label"] if r and r.get("parse_ok") and not r.get("call_failed") else None
        out.append({"item_id": it["item_id"], "run_id": it["run_id"], "question_id": it["question_id"],
                    "condition": it["condition"], "run_label": it["run_label"], "prompt_version": prompt_version,
                    **aggregate_labels(labels, panel_id)})
    return out


def summarize(judge_records: list[dict], panel_records: list[dict], new_records: list[dict], judges=None) -> dict:
    per_judge = {}
    for j in judges or PANEL_JUDGES:
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

def plan_tasks(items: list[dict], bodies_text: dict, done: set,
               spec: JudgeVersionSpec | None = None, judges=None) -> tuple[list[tuple], dict, int]:
    """[(item, judge, messages)] for every (item, judge) not in the cache; {item_id: prepared}; n cached."""
    from _judge_common import html_to_text
    from llm_judge_hallucination_v2_2 import has_context_for

    spec = spec or judge_version_spec(DEFAULT_JUDGE_VERSION)
    tasks, prepared_by_item, n_cached = [], {}, 0
    for it in items:
        prepared = spec.prepare_judge_input(it["candidate"], it["retrieved_context"], has_context_for(it["condition"]))
        prepared_by_item[it["item_id"]] = prepared
        body = bodies_text.get(it["question_id"]) or "[question body not found in parquet]"
        messages = spec.build_messages(it["title"], body, it["tags"], html_to_text(it["ground_truth_answer_html"]),
                                  it["candidate"], prepared=prepared)
        for j in judges or PANEL_JUDGES:
            if (it["item_id"], j.model, spec.prompt_version) in done:
                n_cached += 1
            else:
                tasks.append((it, j, messages))
    return tasks, prepared_by_item, n_cached


class BudgetGuard:
    """Runtime spend cap: a call starts only if the spend so far, plus the worst case of every call in flight,
    plus the worst case of this call (all retries at max_tokens) stays within max_cost_usd."""

    def __init__(self, max_cost_usd: float | None, max_tokens: int, input_tokens_of):
        self.max_cost_usd, self.max_tokens, self.input_tokens_of = max_cost_usd, max_tokens, input_tokens_of
        self.spent = self.reserved = 0.0
        self.n_skipped = 0
        self._lock = threading.Lock()

    def worst_case(self, judge: PanelJudge, messages: list[dict]) -> float:
        tin = self.input_tokens_of(messages)
        return (1 + MAX_RETRIES) * (tin / 1e6 * judge.price_per_m_input + self.max_tokens / 1e6 * judge.price_per_m_output)

    def reserve(self, judge, messages) -> float | None:
        if self.max_cost_usd is None:
            return 0.0
        w = self.worst_case(judge, messages)
        with self._lock:
            if self.spent + self.reserved + w > self.max_cost_usd:
                self.n_skipped += 1
                return None
            self.reserved += w
            return w

    def settle(self, reserved: float, cost: float | None) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += cost or 0.0


def run_tasks(tasks: list[tuple], prepared_by_item: dict, clients: dict, out: Path, concurrency: int,
              sleep=time.sleep, spec: JudgeVersionSpec | None = None, panel_id: str = PANEL_ID,
              budget: BudgetGuard | None = None, record_extra: dict | None = None) -> list[dict]:
    """Item-major submission, so an item's three judges run concurrently (bounded by `concurrency`). With a
    budget guard, a call that could push the spend over the cap is not made (counted in budget.n_skipped)."""
    new = []

    def _one(task):
        it, j, messages = task
        held = budget.reserve(j, messages) if budget else 0.0
        if held is None:
            return None
        rec = judge_record(it, j, messages, prepared_by_item[it["item_id"]], clients[j.judge_id], sleep=sleep,
                           spec=spec, panel_id=panel_id)
        if budget:
            budget.settle(held, rec.get("estimated_cost_usd"))
        if record_extra:  # e.g. the frozen configuration a panel-v2 record was produced under
            rec.update(record_extra)
        return rec

    # as_completed, not pool.map: every finished call is written at once. With pool.map one stalled call at the head of
    # the queue held back the writing of all later (already paid) results, and killing the run lost them.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(_one, t) for t in tasks]):
            rec = fut.result()
            if rec is None:
                continue
            usable = rec["parse_ok"] and not rec["call_failed"]
            append_jsonl(out if usable else failures_path_of(out), rec)
            new.append(rec)
            log(f"  {rec['item_id']} {rec['judge_id']:<8} label={rec['label']} parse_ok={rec['parse_ok']} "
                f"finish={rec['finish_reason']} tokens={rec['prompt_tokens']}/{rec['completion_tokens']}")
    return new


CLIENT_TIMEOUT_SEC = 180.0   # per request (the SDK default of 600 s let one stalled connection hold a run for ~30 minutes)


def make_clients(judges=None) -> dict:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    clients = {}
    for j in judges or PANEL_JUDGES:
        key = os.getenv(j.api_key_env)
        if not key:
            raise RuntimeError(f"{j.api_key_env} not set (repo-root .env)")
        clients[j.judge_id] = OpenAI(api_key=key, base_url=j.base_url, timeout=CLIENT_TIMEOUT_SEC)
    return clients


def panel_v2_registry_gate(run_files: list[str], purpose: str, spec, projection: dict, log=print) -> dict:
    """Amendment 14 (draft) gate for a panel-v2 judging run (alpha stage 2 or official): judge-v2.3 only; every run
    file registered with the matching status (test sample -> 'official', dev -> 'dev-stage2'); the registry budget
    set and the projected cost (with retries for safety: the worst case is reported, the projection is enforced)
    within the remaining DeepInfra budget. Returns what the run records and registers."""
    from judge_versions import PURPOSE_OFFICIAL
    from llm import run_registry as rr

    if spec.prompt_version != "judge-v2.3":
        raise PanelGuardError(f"panel-v2 runs use judge-v2.3 only (got {spec.prompt_version})")
    status = rr.STATUS_OFFICIAL if purpose == PURPOSE_OFFICIAL else rr.STATUS_DEV_STAGE2
    reg = rr.load()
    try:
        entries = rr.assert_reportable(run_files, (status,), reg=reg)
    except rr.RegistryError as e:
        raise PanelGuardError(f"run files must be registered as {status!r} first "
                              f"(python3 -m llm.run_registry register --status {status} <file>): {e}")
    remaining = rr.deepinfra_remaining(reg)
    if remaining is None:
        raise PanelGuardError("no DeepInfra budget set (python3 -m llm.run_registry set-budget <USD>)")
    if projection["total_usd"] > remaining:
        raise PanelGuardError(f"projected ${projection['total_usd']:.4f} exceeds the remaining DeepInfra budget "
                              f"${remaining:.4f} (spent ${rr.deepinfra_spent(reg):.4f})")
    log(f"[registry] {len(entries)} {status} run(s); DeepInfra budget remaining ${remaining:.4f}, projected "
        f"${projection['total_usd']:.4f}")
    frozen = rr.frozen_params_hash()
    return {"status": status, "input_run_ids": [e["run_id"] for e in entries], "remaining_before_usd": remaining,
            "record_extra": {"frozen_params_hash": frozen, "amendment": "Amendment 14", "registry_status": status}}


def register_judge_run(gate: dict, out: Path, panel_path, panel_view: dict, cost: float, started: str, n_items: int,
                       log=print) -> None:
    """Records the spend in the DeepInfra ledger and registers the judge run (same status as its inputs)."""
    from llm import archive_paths
    from llm import run_registry as rr

    rr.record_deepinfra_spend(f"{gate['status']} judging {out.name}", cost)
    files = {"output": archive_paths.repo_rel(archive_paths.canonical(out))}
    if panel_path:
        files["panel"] = archive_paths.repo_rel(archive_paths.canonical(panel_path))
    run_id = f"{out.stem}@{started[:19]}"
    try:
        rr.register({"run_id": run_id, "status": gate["status"], "kind": "judge", "condition": None,
                     "grounding": None, "alpha": None, "n": n_items, "seed": None, "generator": None,
                     "judge_config": panel_view, "commit": _git_head(), "created_at": started, "files": files,
                     "cost_usd": cost, "cost_source": "sum of estimated_cost_usd of this run's records",
                     "notes": f"inputs: {gate['input_run_ids']}",
                     "frozen_params_hash": gate["record_extra"]["frozen_params_hash"]})
        log(f"[registry] judge run {run_id} registered ({gate['status']}); spend ${cost:.4f} recorded")
    except rr.RegistryError as e:
        log(f"[WARN] judge run not registered: {e}")


def _git_head() -> str | None:
    import subprocess
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


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
    parser.add_argument("--judge-version", choices=JUDGE_VERSION_CHOICES, default=DEFAULT_JUDGE_VERSION,
                        help="Judge prompt version, the same for all three judges (default: %(default)s)")
    parser.add_argument("--calibration-set", action="store_true",
                        help="Judge exactly the 60 human-labelled calibration items (Amendment 12). Requires the "
                             "frozen annotation sheet (hash = freeze record) and the finalized reference; no override")
    parser.add_argument("--screen-judge", choices=sorted(SCREENING_BY_ID), default=None,
                        help="Amendment 14 screening: run ONE candidate judge (not the panel) with the frozen prompt, "
                             "parser and blinding; requires --calibration-set and --screening-rule; no panel file")
    parser.add_argument("--screening-rule", default=None,
                        help="The pre-registered eligibility rule (Markdown); its SHA-256 is recorded in the history")
    parser.add_argument("--panel", choices=sorted(PANELS), default=PANEL_ID,
                        help="Panel composition (default %(default)s); panel-v2 = Amendment 14 draft")
    parser.add_argument("--official", action="store_true",
                        help="Official test-sample run files (Amendment 14): requires --panel panel-v2, judge-v2.3, "
                             "the Amendment 14 approval flag, registered 'official' runs and the registry budget")
    parser.add_argument("--token-basis", nargs="+", default=None,
                        help="Stored judge outputs whose prompt_tokens (same messages_sha256) feed the cost projection")
    args = parser.parse_args(argv)

    screening = None
    judges, panel_id = PANELS[args.panel], args.panel
    if args.screen_judge and args.panel != PANEL_ID:
        log("[ERROR] --screen-judge runs one candidate judge, not a panel; leave --panel at its default")
        return 1
    if args.screen_judge:
        if not args.calibration_set or not args.screening_rule or not Path(args.screening_rule).is_file():
            log("[ERROR] --screen-judge needs --calibration-set and an existing --screening-rule file")
            return 1
        judges, panel_id = (SCREENING_BY_ID[args.screen_judge],), SCREENING_PANEL_ID
        rule = Path(args.screening_rule)
        screening = {"judge_id": args.screen_judge, "model": judges[0].model, "rule_path": str(rule),
                     "rule_sha256": hashlib.sha256(rule.read_bytes()).hexdigest()}
        log(f"[screening] {judges[0].model}; rule {rule.name} sha256 {screening['rule_sha256'][:16]}...")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    from _judge_common import CalibrationGuardError, append_manifest, assert_calibration_judging_allowed
    from _judge_common import html_to_text, load_question_bodies
    from annotation_tool import key_dir_from
    from judge_panel_config import DEFAULT_CONCURRENCY
    from judge_versions import JudgeVersionError, check_run_allowed

    concurrency = args.concurrency or DEFAULT_CONCURRENCY
    spec = judge_version_spec(args.judge_version)
    questions_parquet = args.questions_parquet or os.getenv("QUESTIONS_PARQUET")
    # Pre-official archive (2026-10-19): a run file is identified by its ORIGINAL path (the calibration registry,
    # the annotation key and the guards record it) and opened wherever it now is (archive_paths.resolve).
    args.run_files = [str(archive_paths.canonical(p)) for p in args.run_files]
    for p in args.run_files:
        if not archive_paths.resolve(p).exists():
            log(f"[ERROR] run file not found: {p}")
            return 1
    from annotation_tool import AnnotationError, assert_annotation_frozen, load_verified_key
    frozen = None
    try:
        version_purpose = check_run_allowed(spec.prompt_version, args.run_files, log, official=args.official,
                                            panel_id=None if args.screen_judge else panel_id)
        if args.calibration_set:
            if args.allow_unannotated_calibration or args.question_ids or args.items:
                raise PanelGuardError("--calibration-set takes no override and no item selection")
            frozen = assert_annotation_frozen("calibration", "annotator1")
            log(f"[freeze] {frozen['file_name']} matches its freeze record (sha256 {frozen['sha256']})")
        guard = assert_calibration_judging_allowed(args.run_files, args.allow_unannotated_calibration, log)
        if args.calibration_set:
            if not guard["calibration_run_files"] or not guard["reference_ok"]:
                raise PanelGuardError(f"--calibration-set needs the calibration run files and the finalized "
                                      f"reference ({guard['reference_status']})")
            items = select_calibration_set(load_panel_items(args.run_files),
                                           load_verified_key("calibration", key_dir_from(args.key_dir)))
            excl = {"keys": {"calibration": "verified; its 60 items ARE the run (--calibration-set)"}}
            excl_counts = {"calibration_set_items": len(items)}
        else:
            excl = annotation_exclusions(args.run_files, key_dir_from(args.key_dir))
            items_by_file = parse_item_specs(args.items, len(args.run_files))
            items = load_panel_items(args.run_files, args.question_ids, items_by_file)
            items, excl_counts = apply_exclusions(items, excl,
                                                  explicit_question_ids=bool(args.question_ids or args.items))
    except (JudgeVersionError, CalibrationGuardError, PanelGuardError, AnnotationError, ValueError) as e:
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
    done_records = [r for r in done_records if r.get("prompt_version") == spec.prompt_version]
    tasks, prepared_by_item, n_cached = plan_tasks(items, bodies, done, spec, judges)
    basis = [r for f in (args.token_basis or []) for r in read_jsonl(archive_paths.resolve(f))]
    projection = project_cost(tasks, done_records, spec.max_tokens, judges, basis)
    log(f"\n[plan] {spec.prompt_version} (parse {spec.parse_version}, response_format {spec.response_format}): "
        f"{len(items)} items x {len(judges)} judges: {len(tasks)} calls to make, {n_cached} cached")
    log(f"[cost] input basis: {projection['input_token_basis']}")
    log(f"[cost] projected ${projection['total_usd']:.4f} (worst case with all retries "
        f"${projection['worst_case_with_retries_usd']:.4f}); limit ${args.max_cost_usd:.4f}")
    log(json.dumps(projection["per_judge"], indent=2))
    if projection["total_usd"] > args.max_cost_usd:
        log(f"[ABORT] projected cost ${projection['total_usd']:.4f} exceeds --max-cost-usd {args.max_cost_usd}")
        return 2
    registry_gate = None
    if panel_id == "panel-v2" and not args.screen_judge:
        try:
            registry_gate = panel_v2_registry_gate(args.run_files, version_purpose, spec, projection, log)
        except PanelGuardError as e:
            log(f"[ERROR] {e}")
            return 1
    if args.dry_run:
        return 0

    started = datetime.now(timezone.utc).isoformat()
    measured = {r["messages_sha256"]: r["prompt_tokens"] for r in basis if r.get("messages_sha256") and r.get("prompt_tokens")}
    budget = BudgetGuard(args.max_cost_usd, spec.max_tokens,
                         lambda msgs: measured.get(_messages_sha256(msgs))
                         or sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN_EST)
    new = run_tasks(tasks, prepared_by_item, make_clients(judges), out, concurrency, spec=spec, panel_id=panel_id,
                    budget=budget, record_extra=registry_gate and registry_gate["record_extra"]) if tasks else []
    if budget.n_skipped:
        log(f"[BUDGET] {budget.n_skipped} call(s) not made: they could have pushed the spend over "
            f"--max-cost-usd {args.max_cost_usd}")
    judge_records = read_jsonl(out)
    item_ids = {it["item_id"] for it in items}
    panel = build_panel_records(items, [r for r in judge_records if r["item_id"] in item_ids], spec.prompt_version,
                                judges, panel_id)
    panel_path = None
    if screening is None:  # a single screened judge is not a panel: no panel file
        panel_path = panel_path_of(out)
        with open(panel_path, "w", encoding="utf-8") as f:
            for p in panel:
                f.write(json.dumps(p) + "\n")
    summary = summarize([r for r in judge_records if r["item_id"] in item_ids
                         and r.get("prompt_version") == spec.prompt_version], panel, new, judges)
    log(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}")
    append_manifest("judge_panel_run_history.jsonl", {
        "run_started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "run_files": args.run_files, "question_ids_given": bool(args.question_ids), "items_given": args.items,
        "n_items": len(items),
        "out": str(out), "panel_out": str(panel_path) if panel_path else None,
        "panel": panel_public_view(spec, judges, panel_id), "concurrency": concurrency, "screening": screening,
        "token_basis": args.token_basis, "budget_calls_skipped": budget.n_skipped,
        "max_cost_usd": args.max_cost_usd, "projection": projection, "calibration_guard": guard,
        "calibration_set": args.calibration_set, "annotation_freeze": frozen,
        "annotation_exclusion": {"keys": excl["keys"], **excl_counts}, "version_purpose": version_purpose,
        "n_calls": len(new), "n_cached": n_cached, "actual_cost_usd": summary["actual_cost_usd_this_run"],
        "official": args.official, "registry_gate": registry_gate and {k: v for k, v in registry_gate.items()
                                                                      if k != "record_extra"},
    })
    if registry_gate:
        register_judge_run(registry_gate, out, panel_path, panel_public_view(spec, judges, panel_id),
                           summary["actual_cost_usd_this_run"], started, len(items), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
