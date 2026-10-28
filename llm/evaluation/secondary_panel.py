#!/usr/bin/env python3
"""
secondary_panel.py
==================
Secondary metrics of the official evaluation judged by the 3-judge PANEL (operational amendment 12j, 2026-10-28; it
supersedes the single-judge design of Amendment 16 for these two metrics, so inter-judge agreement can be reported).
A NEW entry point: secondary_metrics.py (single judge) is untouched, and so is judge_panel.py (its judge definitions,
call_judge, BudgetGuard, reconcile_ledger and clients are imported and reused as they are).

Two stages, each with its own cap, ledger booking, [identity] check, registration and output files:
  stage 1  --metric ansrel   answer relevance  A, B, C, D   1536 answers x 3 judges = 4608 records
  stage 2  --metric ctxrel   context relevance B, C, D      4063 unique (question, answer, text) triples x 3 = 12189

UNCHANGED: prompts ctxrel-v2 / ansrel-v1 (their own builders), labels, label->score maps (1.0 / 0.5 / 0.0), parsers
(llm_judge_context_relevance_v1.parse_judgment, llm_judge_answer_relevance._parse_answer_relevance_output), blinding
(blind_candidate_v3 for the answer; question + one item for ctxrel), the ctxrel dedup, one prompt per call.
Judges: the three panel-v2 judges (judge_panel_config.PANEL_V2_JUDGES), temperature 0, with the request settings of the
hallucination panel (judge-v2.3: JSON mode, max_tokens, the judge's extra_body, up to MAX_RETRIES retries on a retryable
API error or a parse failure).

Aggregation (per key; key = (cell, question) for ansrel, (question, answer, text) for ctxrel):
  * panel label = majority (>= 2 of the usable judge labels); three different labels -> the median label on the
    ordinal scale (ansrel MENJAWAB_SEBAGIAN, ctxrel PARTIAL); the number decided by majority vs median is reported;
  * fewer than 2 usable judge records -> "incomplete": excluded, never imputed, counted;
  * exactly 2 usable records that DISAGREE -> "unresolved": excluded and counted (the rules above do not cover it; this
    is the hallucination panel's treatment of the same case);
  * panel score = score of the panel label; per-judge mean score and the mean of the judges' scores are reported too.
  * cells are pooled (micro): ansrel over answers, ctxrel over items (llm_judge_context_relevance_v1.
    compute_run_context_relevance is reused for the ctxrel extras).
  * agreement: % all three agree, pairwise %, Fleiss' kappa, Krippendorff's alpha (ordinal), over keys with 3 usable.

    python3 llm/evaluation/secondary_panel.py --metric ansrel                      # plan (dry run), nothing is called
    python3 llm/evaluation/secondary_panel.py --metric ansrel --run --max-cost-usd X --run-files A=.. B=.. C=.. D=..
    python3 llm/evaluation/secondary_panel.py --report                             # whatever stages exist

--judges llama,deepseek runs a SUBSET of the panel (default: all three). A judge outside the subset is reported as
"not run" and a stage whose three judges are not all complete produces NO panel file, panel label, agreement or panel
mean: only per-judge figures and a labelled 2-judge interim agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
for _p in (EVAL_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm import archive_paths  # noqa: E402
import secondary_metrics as sm  # noqa: E402
from judge_panel_config import MAX_RETRIES, PANEL_V2_JUDGES, judge_cost_usd  # noqa: E402

log = print
JUDGES = PANEL_V2_JUDGES
OUT_DIR = REPO_ROOT / "results" / "official_secondary_panel"
UNUSABLE_LIMIT = 0.02
INCOMPLETE, UNRESOLVED = "INCOMPLETE", "UNRESOLVED"

METRICS = {
    "ansrel": {"prompt": sm.ANSREL_PROMPT, "labels": ("TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN", "MENJAWAB"),  # ordinal, low -> high
               "score": sm.ANSREL_LABEL_SCORE, "est_tokens": sm.EST_TOKENS["ansrel"], "conditions": "ABCD"},
    "ctxrel": {"prompt": sm.CTXREL_PROMPT, "labels": ("IRRELEVANT", "PARTIAL", "RELEVANT"),
               "score": {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}, "est_tokens": sm.EST_TOKENS["ctxrel"],
               "conditions": "BCD"},
}
_LOCK = threading.Lock()


class StageError(RuntimeError):
    pass


def out_paths(metric: str, out_dir=None) -> dict:
    d = Path(out_dir) if out_dir else OUT_DIR
    return {"records": d / f"{metric}_panel_official.jsonl", "panel": d / f"{metric}_panel_official_panel.jsonl"}


def failures_path_of(out: Path) -> Path:
    return Path(out).with_name(Path(out).stem + "_failures.jsonl")


# ---------------------------------------------------------------------------
# Request settings and parsers (the hallucination panel's request, the existing secondary parsers)
# ---------------------------------------------------------------------------

class RequestSpec:
    """What jp.call_judge reads from a JudgeVersionSpec: temperature, max_tokens, response_format, parse."""

    def __init__(self, metric: str):
        import judge_prompt_v2_3 as v23

        self.metric = metric
        self.temperature, self.max_tokens = v23.REQUEST_TEMPERATURE, v23.REQUEST_MAX_TOKENS
        self.response_format = v23.REQUEST_RESPONSE_FORMAT
        self.prompt_version = METRICS[metric]["prompt"]

    def parse(self, raw, has_context=False, offered_items=()) -> dict:
        if self.metric == "ansrel":
            import llm_judge_answer_relevance as ar

            p = ar._parse_answer_relevance_output(raw)
            return {"parse_error": p is None, "label": p and p["answer_relevance_label"]}
        import llm_judge_context_relevance_v1 as cx

        p = cx.parse_judgment(raw)
        return {"parse_error": p["parse_error"], "label": p["label"]}


# ---------------------------------------------------------------------------
# Keys, items, expected sets
# ---------------------------------------------------------------------------

def key_of(rec: dict, metric: str) -> tuple:
    if metric == "ansrel":
        return (rec.get("source_file"), rec.get("question_id"))
    return (rec.get("question_id"), rec.get("answer_id"), rec.get("text_hash"))


def _ctx_files(runs: dict) -> list[str]:
    return [p for c, p in runs.values() if c != "A"]


def expected_keys(metric: str, runs: dict) -> set:
    if metric == "ansrel":
        return {(str(archive_paths.canonical(p)), r["question_id"]) for _, p in runs.values() for r in sm._read(p)}
    import llm_judge_context_relevance_v1 as cx

    files = _ctx_files(runs)
    return {(it["question_id"], it["answer_id"], it["text_hash"]) for it in (cx.load_items(files, None) if files else [])}


def build_items(metric: str, runs: dict) -> list[dict]:
    """One dict per (key): the messages every judge receives, plus the identifying fields of the record."""
    import os
    from _judge_common import html_to_text, load_question_bodies

    qp = os.getenv("QUESTIONS_PARQUET")
    items = []
    if metric == "ansrel":
        import llm_judge_answer_relevance as ar
        from judge_blinding import blind_candidate_v3

        for _, path in runs.values():
            recs = sm._read(path)
            bodies = load_question_bodies(qp, [r["question_id"] for r in recs])
            for r in recs:
                body = html_to_text(bodies.get(r["question_id"]), 3000) if bodies.get(r["question_id"]) else ""
                items.append({"source_file": str(archive_paths.canonical(path)), "question_id": r["question_id"],
                              "messages": ar.build_answer_relevance_prompt(r.get("title", ""), body,
                                                                           blind_candidate_v3(r.get("llm_answer", "")))})
        return items
    import llm_judge_context_relevance_v1 as cx

    raw = cx.load_items(_ctx_files(runs), None)
    bodies = load_question_bodies(qp, sorted({it["question_id"] for it in raw}))
    build = cx.get_prompt_builder(METRICS[metric]["prompt"])
    for it in raw:
        html = bodies.get(it["question_id"], "")
        items.append({"question_id": it["question_id"], "answer_id": it["answer_id"], "text_hash": it["text_hash"],
                      "messages": build(it["title"], html_to_text(html) if html else "", it["tags"], it["chunk_text"])})
    return items


def done_keys(metric: str, out: Path) -> set:
    """(key, judge_id) of USABLE records; failed / unparseable calls live in the failures file and are retried."""
    return {(key_of(r, metric), r.get("judge_id")) for r in sm._read_tolerant(out)
            if r.get("parse_ok") and r.get("label") is not None}


# ---------------------------------------------------------------------------
# Plan (dry run): counts and a fixed-size cost estimate, nothing is called
# ---------------------------------------------------------------------------

def plan(metric: str, runs: dict, out: Path | None = None, judges=None) -> dict:
    judges = judges or JUDGES
    tin, tout = METRICS[metric]["est_tokens"]
    n_keys = len(expected_keys(metric, runs))
    done = done_keys(metric, out) if out else set()
    per_judge = {}
    for j in judges:
        pending = n_keys - sum(1 for (_, jid) in done if jid == j.judge_id)
        per_judge[j.judge_id] = {"calls": n_keys, "pending": pending,
                                 "cost_usd": round(n_keys * (tin * j.price_per_m_input + tout * j.price_per_m_output) / 1e6, 4),
                                 "pending_cost_usd": round(pending * (tin * j.price_per_m_input + tout * j.price_per_m_output) / 1e6, 4)}
    total = round(sum(v["cost_usd"] for v in per_judge.values()), 4)
    return {"metric": metric, "keys": n_keys, "per_judge": per_judge, "total_calls": n_keys * len(judges),
            "projected_usd": total, "with_buffer_usd": round(total * 1.1, 4), "proposed_cap_usd": round(total * 1.1 * 1.25, 4),
            "pending_projected_usd": round(sum(v["pending_cost_usd"] for v in per_judge.values()), 4),
            "note": "fixed-size token estimate (no call was made); earlier judge runs came in 16-25% under their projection"}


def plan_text(p: dict) -> str:
    lines = [f"Secondary-metrics PANEL plan, stage {p['metric']}: {p['keys']} keys x {len(p['per_judge'])} judges"]
    for jid, v in p["per_judge"].items():
        lines.append(f"  {jid:<12} calls {v['calls']:>6}  pending {v['pending']:>6}  projected ${v['cost_usd']:.4f}")
    lines += [f"TOTAL calls {p['total_calls']}  projected ${p['projected_usd']:.4f}  with 10% buffer ${p['with_buffer_usd']:.4f}  "
              f"proposed cap (1.25 x buffered) ${p['proposed_cap_usd']:.4f}", p["note"], "NO CALL WAS MADE."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One call and the guarded batch
# ---------------------------------------------------------------------------

def make_record(metric: str, item: dict, judge, out: dict, spec: RequestSpec) -> dict:
    rec = {"metric": metric, "panel_id": "panel-v2", "judge_id": judge.judge_id, "judge_model": judge.model,
           "provider": judge.provider, "prompt_version": spec.prompt_version, "temperature": spec.temperature,
           "max_tokens": spec.max_tokens, "request_response_format": spec.response_format,
           "request_extra_body": judge.extra_body, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    rec.update({k: item[k] for k in (("source_file", "question_id") if metric == "ansrel"
                                      else ("question_id", "answer_id", "text_hash"))})
    label = out.get("label") if out.get("parse_ok") else None
    rec.update({"label": label, "score": METRICS[metric]["score"].get(label), "parse_ok": bool(out.get("parse_ok")),
                "call_failed": bool(out.get("call_failed")), "error": out.get("error"), "attempts": out.get("attempts"),
                "raw_response": out.get("raw_response"), "finish_reason": out.get("finish_reason"),
                "prompt_tokens": out.get("prompt_tokens"), "completion_tokens": out.get("completion_tokens"),
                "response_model": out.get("response_model"), "latency_s": out.get("latency_s"),
                "reasoning_present": out.get("reasoning_present")})
    rec["served_model_mismatch"] = rec["response_model"] is not None and rec["response_model"] != judge.expected_served_model
    cost = judge_cost_usd(judge, rec["prompt_tokens"], rec["completion_tokens"])
    paid = [a for a in (rec["attempts"] or []) if a.get("completion_tokens") is not None]
    if cost is not None and len(paid) > 1:  # earlier parse-failure attempts were paid too (as in judge_panel.judge_record)
        cost = round(cost + (len(paid) - 1) * rec["prompt_tokens"] / 1e6 * judge.price_per_m_input
                     + sum(a["completion_tokens"] for a in paid[:-1]) / 1e6 * judge.price_per_m_output, 8)
    rec["n_paid_attempts"] = len(paid)
    rec["estimated_cost_usd"] = cost
    return rec


def run_stage(metric: str, items: list[dict], clients: dict, out: Path, guard, concurrency: int = 6,
              sleep=None, judges=None) -> dict:
    """Calls every judge for every item that has no usable record yet; every finished call is written at once
    (usable -> out, failed / unparseable -> the failures file). A call the guard refuses is counted, not made."""
    import judge_panel as jp
    import time

    spec = RequestSpec(metric)
    sleep = sleep or time.sleep
    judges = judges or JUDGES
    done = done_keys(metric, out)
    tasks = [(it, j) for it in items for j in judges if (key_of(it, metric), j.judge_id) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"calls": 0, "usable": 0, "failed": 0, "cached": len(items) * len(judges) - len(tasks)}

    def one(task):
        it, j = task
        held = guard.reserve(j, it["messages"]) if guard else 0.0
        if held is None:
            return None
        res = jp.call_judge(clients[j.judge_id], j, it["messages"], False, (), sleep=sleep, spec=spec)
        rec = make_record(metric, it, j, res, spec)
        if guard:
            guard.settle(held, rec.get("estimated_cost_usd"))
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            rec = fut.result()
            if rec is None:
                continue
            usable = rec["parse_ok"] and not rec["call_failed"]
            with _LOCK, open(out if usable else failures_path_of(out), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            stats["calls"] += 1
            stats["usable" if usable else "failed"] += 1
    return stats


# ---------------------------------------------------------------------------
# Aggregation and agreement
# ---------------------------------------------------------------------------

def aggregate_key(judge_labels: dict, metric: str) -> dict:
    """judge_labels {judge_id: label or None}. Returns panel_label (or INCOMPLETE / UNRESOLVED) and how it was decided."""
    order = METRICS[metric]["labels"]
    valid = {j: lab for j, lab in judge_labels.items() if lab is not None}
    if len(valid) < 2:
        return {"panel_label": INCOMPLETE, "decided_by": None, "n_valid": len(valid)}
    votes = Counter(valid.values())
    lab, n = votes.most_common(1)[0]
    if n >= 2:
        return {"panel_label": lab, "decided_by": "majority", "n_valid": len(valid)}
    if len(valid) == 3:  # three different labels: the median on the ordinal scale
        return {"panel_label": sorted(valid.values(), key=order.index)[1], "decided_by": "median", "n_valid": 3}
    return {"panel_label": UNRESOLVED, "decided_by": None, "n_valid": len(valid)}  # two usable judges that disagree


def fleiss_kappa(rows: list[list[str]], categories) -> float | None:
    from judge_interjudge_analysis import fleiss_kappa as fk

    k = fk(rows, categories)
    return None if k is None else round(k, 4)


def krippendorff_alpha_ordinal(rows: list[list[str]], order) -> float | None:
    """Krippendorff's alpha, ordinal metric (Krippendorff 2011): coincidence matrix o_ck, n_c; delta^2_ck =
    (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2; alpha = 1 - D_o / D_e. Units with < 2 values are skipped."""
    rank = {lab: i for i, lab in enumerate(order)}
    units = [[rank[x] for x in r if x is not None] for r in rows]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    k = len(order)
    o = [[0.0] * k for _ in range(k)]
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[a][b] += 1 / (m - 1)
    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n <= 1:
        return None

    def d2(c, kk):
        lo, hi = min(c, kk), max(c, kk)
        return (sum(n_c[lo:hi + 1]) - (n_c[c] + n_c[kk]) / 2) ** 2

    d_o = sum(o[c][kk] * d2(c, kk) for c in range(k) for kk in range(k)) / n
    d_e = sum(n_c[c] * n_c[kk] * d2(c, kk) for c in range(k) for kk in range(k)) / (n * (n - 1))
    return None if d_e == 0 else round(1 - d_o / d_e, 4)


def agreement(label_rows: dict, metric: str) -> dict:
    """label_rows {key: {judge_id: label or None}}. Statistics over the keys with three usable judge labels; pairwise
    agreement over the keys where both judges of the pair are usable."""
    order = METRICS[metric]["labels"]
    ids = [j.judge_id for j in JUDGES]
    full = [[r[j] for j in ids] for r in label_rows.values() if all(r.get(j) is not None for j in ids)]
    pair = {}
    for a, b in combinations(ids, 2):
        both = [(r[a], r[b]) for r in label_rows.values() if r.get(a) is not None and r.get(b) is not None]
        pair[f"{a}_vs_{b}"] = {"n": len(both), "pct_agree": round(100 * sum(x == y for x, y in both) / len(both), 2) if both else None}
    vals = [v["pct_agree"] for v in pair.values() if v["pct_agree"] is not None]
    return {"n_keys_three_usable": len(full),
            "pct_all_three_agree": round(100 * sum(len(set(r)) == 1 for r in full) / len(full), 2) if full else None,
            "pairwise": pair, "mean_pairwise_pct": round(sum(vals) / len(vals), 2) if vals else None,
            "fleiss_kappa": fleiss_kappa(full, order) if full else None,
            "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(full, order) if full else None}


def label_rows_of(metric: str, recs: list[dict]) -> dict:
    """{key: {judge_id: label}} from the usable records (a judge with no usable record has no entry -> None)."""
    rows: dict = {}
    for r in recs:
        if r.get("parse_ok") and r.get("label") is not None:
            rows.setdefault(key_of(r, metric), {})[r["judge_id"]] = r["label"]
    return {k: {j.judge_id: v.get(j.judge_id) for j in JUDGES} for k, v in rows.items()}


def build_panel_records(metric: str, recs: list[dict], expected: set | None = None) -> list[dict]:
    rows = label_rows_of(metric, recs)
    keys = sorted(expected, key=str) if expected is not None else sorted(rows, key=str)
    out = []
    for k in keys:
        jl = rows.get(k, {j.judge_id: None for j in JUDGES})
        agg = aggregate_key(jl, metric)
        rec = {"metric": metric, "key": list(k), "judge_labels": jl, **agg,
               "panel_score": METRICS[metric]["score"].get(agg["panel_label"])}
        scores = [METRICS[metric]["score"][v] for v in jl.values() if v is not None]
        rec["mean_judge_score"] = round(sum(scores) / len(scores), 4) if len(scores) == 3 else None
        out.append(rec)
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(metric: str, panel: list[dict], recs: list[dict]) -> dict:
    """Pooled summary of one set of keys (a cell, a version, or everything)."""
    labels = METRICS[metric]["labels"]
    decided = [p for p in panel if p["panel_label"] not in (INCOMPLETE, UNRESOLVED)]
    dist = Counter(p["panel_label"] for p in decided)
    per_judge = {}
    keyset = {tuple(p["key"]) for p in panel}
    for j in JUDGES:
        js = [r for r in recs if r["judge_id"] == j.judge_id and r.get("label") is not None and key_of(r, metric) in keyset]
        per_judge[j.judge_id] = {"n": len(js), "mean_score": _mean([r["score"] for r in js]),
                                 "labels": dict(Counter(r["label"] for r in js))}
    full = [p["mean_judge_score"] for p in panel if p["mean_judge_score"] is not None]
    return {"n_keys": len(panel), "n_decided": len(decided),
            "n_incomplete": sum(p["panel_label"] == INCOMPLETE for p in panel),
            "n_unresolved": sum(p["panel_label"] == UNRESOLVED for p in panel),
            "decided_by_majority": sum(p["decided_by"] == "majority" for p in panel),
            "decided_by_median": sum(p["decided_by"] == "median" for p in panel),
            "panel_labels": {lab: dist.get(lab, 0) for lab in labels},
            "panel_mean_score": _mean([p["panel_score"] for p in decided]), "per_judge": per_judge,
            "mean_of_judge_scores": _mean(full), "n_keys_three_usable": len(full)}


def load_abstain_keys(results_dir: Path | None = None) -> set | None:
    """(condition, question_id) of the answers the hallucination panel labelled ABSTAIN, from the official panel files;
    None when no such file exists (the non-ABSTAIN version is then reported as not available)."""
    d = Path(results_dir) if results_dir else EVAL_DIR / "results"
    files = sorted(d.glob("panel_official_R*_judge-v2.3_panel.jsonl"))
    if not files:
        return None
    return {(p["condition"], int(p["question_id"])) for f in files for p in sm._read_tolerant(f)
            if p.get("panel_label") == "ABSTAIN"}


def report_ansrel(runs: dict, recs: list[dict], abstain: set | None, expected: set | None = None) -> dict:
    panel = build_panel_records("ansrel", recs, expected)
    cell_of = {str(archive_paths.canonical(p)): c for c, p in runs.values()}

    def cell(p):
        return cell_of.get(p["key"][0], "?")

    rows = label_rows_of("ansrel", recs)
    out = {"status": "run", "cells": {}, "all_cells": {}}
    for version in ("all_answers", "non_abstain"):
        if version == "non_abstain" and abstain is None:
            out["non_abstain_status"] = "not available (no official hallucination-panel files)"
            continue
        keep = (lambda p: True) if version == "all_answers" else (lambda p: (cell(p), int(p["key"][1])) not in abstain)
        for c in sorted({cell(p) for p in panel}):
            sel = [p for p in panel if cell(p) == c and keep(p)]
            ks = {tuple(p["key"]) for p in sel}
            out["cells"].setdefault(c, {})[version] = {
                **summarize("ansrel", sel, recs), "agreement": agreement({k: v for k, v in rows.items() if k in ks}, "ansrel")}
        sel = [p for p in panel if keep(p)]
        ks = {tuple(p["key"]) for p in sel}
        out["all_cells"][version] = {**summarize("ansrel", sel, recs),
                                     "agreement": agreement({k: v for k, v in rows.items() if k in ks}, "ansrel")}
    return out


def report_ctxrel(runs: dict, recs: list[dict], expected: set | None = None) -> dict:
    import llm_judge_context_relevance_v1 as cx

    panel = build_panel_records("ctxrel", recs, expected)
    by_key = {tuple(p["key"]): {"label": p["panel_label"]} for p in panel if p["panel_label"] not in (INCOMPLETE, UNRESOLVED)}
    rows = label_rows_of("ctxrel", recs)
    out = {"status": "run", "cells": {}, "all_cells": {**summarize("ctxrel", panel, recs), "agreement": agreement(rows, "ctxrel")}}
    for c, path in ((c, p) for c, p in runs.values() if c != "A"):
        # extras of the single-judge report, reused: % per label, questions with >= 1 RELEVANT, mean score rank 1 vs 2-5
        out["cells"][c] = {"panel_context_relevance": cx.compute_run_context_relevance(str(archive_paths.resolve(path)), by_key)}
    return out


def judge_states(metric: str, runs: dict, out: Path, judges=None) -> dict:
    """{judge_id: "not run" | "partial" | "complete"}. A judge outside `judges` (when a subset is given) or without any
    record is "not run"; "complete" = every expected key attempted and unusable <= 2%."""
    exp = expected_keys(metric, runs)
    recs = sm._read_tolerant(out)
    fails = sm._read_tolerant(failures_path_of(out))
    selected = {j.judge_id for j in (judges or JUDGES)}
    states = {}
    for j in JUDGES:
        if j.judge_id not in selected:
            states[j.judge_id] = "not run"
            continue
        usable = {key_of(r, metric) for r in recs if r.get("judge_id") == j.judge_id and r.get("parse_ok") and r.get("label") is not None}
        tried = usable | {key_of(r, metric) for r in fails if r.get("judge_id") == j.judge_id}
        if not tried:
            states[j.judge_id] = "not run"
        elif tried >= exp and len(exp - usable) <= UNUSABLE_LIMIT * len(exp):
            states[j.judge_id] = "complete"
        else:
            states[j.judge_id] = "partial"
    return states


def _incomplete_text(states: dict) -> str:
    return "; ".join(f"{j} {s}" for j, s in states.items() if s != "complete")


def report_partial(metric: str, runs: dict, recs: list[dict], states: dict, abstain: set | None) -> dict:
    """Fewer than three complete judges: per-judge figures and a labelled 2-judge interim agreement ONLY. No panel file,
    panel label, agreement among three, or panel mean."""
    from judge_binary import cohen_kappa

    done = [j for j, s in states.items() if s == "complete"]
    cell_of = {str(archive_paths.canonical(p)): c for c, p in runs.values()}
    order = METRICS[metric]["labels"]
    cells = sorted(set(cell_of.values())) if metric == "ansrel" else ["all"]

    def cell(r):
        return cell_of.get(r.get("source_file"), "?") if metric == "ansrel" else "all"

    def abst(r):  # (cell, question) labelled ABSTAIN by the hallucination panel
        return abstain is not None and metric == "ansrel" and (cell(r), int(r["question_id"])) in abstain

    out = {"status": "incomplete", "incomplete": _incomplete_text(states),
           "panel": "not computed: no panel label, 3-judge agreement or panel mean unless all three judges are complete",
           "judge_states": states, "per_judge": {}}
    usable = [r for r in recs if r.get("parse_ok") and r.get("label") is not None]
    for jid in done:
        mine = [r for r in usable if r["judge_id"] == jid]
        d = {}
        for version in ("all_answers", "non_abstain"):
            if version == "non_abstain" and (abstain is None or metric != "ansrel"):
                continue
            sel = [r for r in mine if version == "all_answers" or not abst(r)]
            d[version] = {c: {"n": len([r for r in sel if cell(r) == c]),
                              "mean_score": _mean([r["score"] for r in sel if cell(r) == c]),
                              "labels": {lab: sum(1 for r in sel if cell(r) == c and r["label"] == lab) for lab in order}}
                          for c in cells}
            d[version]["all"] = {"n": len(sel), "mean_score": _mean([r["score"] for r in sel]),
                                 "labels": {lab: sum(1 for r in sel if r["label"] == lab) for lab in order}}
        out["per_judge"][jid] = d
    if metric == "ansrel" and abstain is None:
        out["non_abstain_status"] = "not available (no official hallucination-panel files)"
    out["interim_two_judge_agreement"] = {"label": "2-judge interim figure, NOT panel agreement"}
    for a, b in combinations(done, 2):
        by = {}
        for r in usable:
            if r["judge_id"] in (a, b):
                by.setdefault(key_of(r, metric), {})[r["judge_id"]] = r["label"]
        both = {k: v for k, v in by.items() if len(v) == 2}
        la, lb = [v[a] for v in both.values()], [v[b] for v in both.values()]
        entry = {"n": len(both), "pct_agree": round(100 * sum(x == y for x, y in zip(la, lb)) / len(both), 2) if both else None,
                 "cohen_kappa": cohen_kappa(la, lb, order) if both else None}
        if metric == "ansrel":
            entry["per_cell"] = {}
            for c in cells:
                kc = {k: v for k, v in both.items() if cell_of.get(k[0]) == c}
                xs, ys = [v[a] for v in kc.values()], [v[b] for v in kc.values()]
                entry["per_cell"][c] = {"n": len(kc), "pct_agree": round(100 * sum(x == y for x, y in zip(xs, ys)) / len(kc), 2) if kc else None}
        out["interim_two_judge_agreement"][f"{a}_vs_{b}"] = entry
    return out


def report(runs: dict | None = None, out_dir=None, results_dir=None) -> dict:
    """Both stages; a stage with no output yet is reported as {"status": "not run"} (never as zero). A stage whose three
    judges are not all complete is {"status": "incomplete", ...} with per-judge figures only."""
    res = {}
    for metric in METRICS:
        f = out_paths(metric, out_dir)["records"]
        recs = sm._read_tolerant(f)
        if not recs:
            res[metric] = {"status": "not run"}
            continue
        if runs is None:
            res[metric] = {"status": "run", "note": "pass the run files: the panel / per-judge report needs the expected keys",
                           "records_per_judge": dict(Counter(r["judge_id"] for r in recs))}
            continue
        mruns = {k: v for k, v in runs.items() if v[0] in METRICS[metric]["conditions"]}
        states = judge_states(metric, mruns, f)
        if any(s != "complete" for s in states.values()):
            res[metric] = report_partial(metric, mruns, recs, states, load_abstain_keys(results_dir))
        elif metric == "ansrel":
            res[metric] = report_ansrel(runs, recs, load_abstain_keys(results_dir))
        else:
            res[metric] = report_ctxrel(runs, recs)
    return res


# ---------------------------------------------------------------------------
# Safeguards (same pattern as judge_panel.py / secondary_metrics.py)
# ---------------------------------------------------------------------------

def identity_report(metric: str, runs: dict, out: Path, judges=None) -> dict:
    """What the stage's output file holds, per judge, against the expected keys and the frozen id lists. A judge outside
    `judges` (when a subset is given) or without any record is reported as "not run" (information, not a failure); its
    stray records, if any, are only counted."""
    from llm.sample_split import load_frozen_ids

    test = {int(i) for i in load_frozen_ids("test")["ids"]}
    dev = {int(i) for i in load_frozen_ids("dev")["ids"]}
    exp = expected_keys(metric, runs)
    recs = sm._read_tolerant(out)
    fails = sm._read_tolerant(failures_path_of(out))
    qid = (lambda k: int(k[1])) if metric == "ansrel" else (lambda k: int(k[0]))
    selected = {j.judge_id for j in (judges or JUDGES)}
    per_judge, ok = {}, True
    for j in JUDGES:
        keys = [key_of(r, metric) for r in recs if r.get("judge_id") == j.judge_id]
        n_fail = sum(1 for r in fails if r.get("judge_id") == j.judge_id)
        if j.judge_id not in selected or (not keys and not n_fail):
            per_judge[j.judge_id] = {"status": "not run", "records_on_disk": len(keys), "failures_file_records": n_fail,
                                     "expected": len(exp)}
            continue
        unusable = len(exp) - len(set(keys) & exp)
        ids = {qid(k) for k in keys}
        d = {"status": "run", "records": len(keys), "unique_keys": len(set(keys)), "expected": len(exp),
             "keys_inside_expected": set(keys) <= exp, "keys_equal_expected": set(keys) == exp,
             "unusable": unusable, "unusable_pct": round(100 * unusable / len(exp), 3) if exp else None,
             "dev_ids": len(ids & dev), "ids_outside_test_list": len(ids - test),
             "failures_file_records": n_fail,
             "served_model_mismatch": sum(1 for r in recs if r.get("judge_id") == j.judge_id and r.get("served_model_mismatch"))}
        d["ok"] = (d["records"] == d["unique_keys"] and d["keys_inside_expected"] and d["dev_ids"] == 0
                   and d["ids_outside_test_list"] == 0 and unusable <= UNUSABLE_LIMIT * len(exp))
        ok &= d["ok"]
        per_judge[j.judge_id] = d
    ran = [d for d in per_judge.values() if d["status"] == "run"]
    return {"metric": metric, "per_judge": per_judge, "ok": bool(ok and ran),
            "panel_complete": len(ran) == len(JUDGES) and all(d["ok"] for d in ran)}


def assert_no_foreign_records(metric: str, runs: dict, out: Path) -> None:
    exp = expected_keys(metric, runs)
    foreign = {key_of(r, metric) for r in sm._read_tolerant(out)} - exp
    if foreign:
        raise sm.SecondaryGuardError(f"--run refused: {Path(out).name}: {len(foreign)} key(s) not among the current inputs")


def register_stage(status: str, metric: str, runs: dict, paths: dict, cost: float, started: str, n_calls: int, log=print,
                   judges=None) -> None:
    import subprocess
    from llm import run_registry as rr

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    run_id = f"secondary_panel_{metric}@{started[:19]}"
    try:
        rr.register({"run_id": run_id, "status": status, "kind": "judge", "condition": None, "grounding": None,
                     "alpha": None, "n": n_calls, "seed": None, "generator": None,
                     "judge_config": {"panel_id": "panel-v2", "metric": metric, "prompt_version": METRICS[metric]["prompt"],
                                      "temperature": 0.0, "max_retries": MAX_RETRIES,
                                      "judges": [{"judge_id": j.judge_id, "model": j.model, "extra_body": j.extra_body} for j in (judges or JUDGES)]},
                     "commit": head or None, "created_at": started,
                     "files": {"records": rr._rel(paths["records"]), "panel": rr._rel(paths["panel"])},
                     "cost_usd": cost, "cost_source": "sum of estimated_cost_usd of this run's records",
                     "notes": f"inputs: {sorted(runs)}; judges: {[j.judge_id for j in (judges or JUDGES)]}", "frozen_params_hash": rr.frozen_params_hash()})
        log(f"[registry] secondary panel run {run_id} registered ({status}); spend ${cost:.4f} recorded")
    except rr.RegistryError as e:
        log(f"[WARN] secondary panel run not registered: {e}")


def run_guarded(metric: str, runs: dict, files: list[str], args, clients=None) -> int:
    import judge_versions as jv
    import judge_panel as jp
    from llm.common.resume import repair_torn_tail

    if args.max_cost_usd is None:
        print("[ERROR] --run needs --max-cost-usd")
        return 2
    official = jv.run_purpose(files) == jv.PURPOSE_OFFICIAL
    if official and args.out_dir:
        print("[ERROR] official run files write to results/official_secondary_panel only (no --out-dir)")
        return 2
    paths = out_paths(metric, args.out_dir)
    out = paths["records"]
    ids = getattr(args, "judges", None)
    judges = tuple(j for j in JUDGES if ids is None or j.judge_id in ids)
    if not judges:
        print(f"[ERROR] --judges names no panel judge ({[j.judge_id for j in JUDGES]})")
        return 2
    bad = [c for c, _ in runs.values() if c not in METRICS[metric]["conditions"]]
    if bad:
        print(f"[ERROR] stage {metric} covers conditions {METRICS[metric]['conditions']}; got {bad}")
        return 2
    status, remaining = None, None
    if official:
        from llm import run_registry as rr

        try:
            jp.assert_official_inputs(files)
            entries = rr.assert_reportable(files, (rr.STATUS_OFFICIAL,))
            assert_no_foreign_records(metric, runs, out)
        except Exception as e:  # PanelGuardError, RegistryError, SecondaryGuardError
            print(f"[ERROR] {e}")
            return 2
        status, remaining = rr.STATUS_OFFICIAL, rr.deepinfra_remaining()
        print(f"[registry] {len(entries)} {status} run(s); DeepInfra budget remaining ${remaining:.4f}")
    for f in (out, failures_path_of(out), paths["panel"]):
        repair_torn_tail(f)
    if official:
        jp.reconcile_ledger(out, status, log)
    p = plan(metric, runs, out, judges)
    print(plan_text(p))
    projected = p["pending_projected_usd"]
    print(f"[plan] pending projected ${projected:.4f}; limit ${args.max_cost_usd:.4f}")
    if projected > args.max_cost_usd:
        print(f"[ABORT] projected cost ${projected:.4f} exceeds --max-cost-usd {args.max_cost_usd}")
        return 2
    if official and projected > remaining:
        print(f"[ABORT] projected ${projected:.4f} exceeds the remaining DeepInfra budget ${remaining:.4f}")
        return 2
    started = datetime.now(timezone.utc).isoformat()
    spec = RequestSpec(metric)
    guard = jp.BudgetGuard(args.max_cost_usd, spec.max_tokens,
                           lambda msgs: sum(len(m["content"]) for m in msgs) / jp.CHARS_PER_TOKEN_EST)
    items = build_items(metric, runs)
    before = _stored_cost(out)
    stats = run_stage(metric, items, clients or jp.make_clients(judges), out, guard, args.workers, judges=judges)
    cost = round(_stored_cost(out) - before, 6)
    print(json.dumps(stats))
    if guard.n_skipped:
        print(f"[BUDGET] {guard.n_skipped} call(s) not made: they could have pushed the spend over --max-cost-usd {args.max_cost_usd}")
    print(f"[cost] this run ${cost:.4f} (cap ${args.max_cost_usd:.4f})")
    recs = sm._read_tolerant(out)
    states = judge_states(metric, runs, out)
    if all(s == "complete" for s in states.values()):
        with open(paths["panel"], "w", encoding="utf-8") as f:
            for r in build_panel_records(metric, recs, expected_keys(metric, runs)):
                f.write(json.dumps(r) + "\n")
    else:
        print(f"[PANEL] incomplete: {_incomplete_text(states)}; no panel file, panel label, agreement or panel mean computed")
    if official:
        jp.reconcile_ledger(out, status, log)
        register_stage(status, metric, runs, paths, cost, started, stats["calls"], judges=judges)
        ident = identity_report(metric, runs, out, judges)
        print(f"[identity] {json.dumps(ident)}")
        print("[identity] OK" if ident["ok"] else "[identity] MISMATCH: do not use this file before it is explained")
    return 0


def _stored_cost(out: Path) -> float:
    return round(sum(float(r.get("estimated_cost_usd") or 0) for r in sm._read_tolerant(out) + sm._read_tolerant(failures_path_of(out))), 6)


def _runs_from_args(args) -> dict:
    runs: dict = {}
    if args.run_files:
        for spec in args.run_files:
            label, _, path = spec.partition("=")
            runs[label] = (label.strip()[0].upper(), path)
    else:
        from llm import run_registry as rr

        for e in rr.runs((rr.STATUS_OFFICIAL,)):
            if e["kind"] == "generation":
                runs[f"{e['condition']}-{e.get('run_id', '')[:8]}"] = (e["condition"], e["files"]["output"])
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=tuple(METRICS), default=None, help="the stage: ansrel (stage 1) or ctxrel (stage 2)")
    ap.add_argument("--run-files", nargs="*", default=None, help="LABEL=PATH (A=, B=, C=, D=)")
    ap.add_argument("--run", action="store_true", help="actually judge (default: plan only, nothing is called)")
    ap.add_argument("--report", action="store_true", help="aggregate whatever stages exist (no call)")
    ap.add_argument("--max-cost-usd", type=float, default=None)
    ap.add_argument("--out-dir", default=None, help="tests / smoke only; refused for official run files")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judges", default=None, help="comma-separated subset of the panel to call (default: all three), e.g. llama,deepseek")
    args = ap.parse_args(argv)
    if args.judges is not None:
        args.judges = [x.strip() for x in args.judges.split(",") if x.strip()]
        unknown = [x for x in args.judges if x not in {j.judge_id for j in JUDGES}]
        if unknown:
            print(f"[ERROR] unknown judge(s) {unknown}; panel: {[j.judge_id for j in JUDGES]}")
            return 2
    runs = _runs_from_args(args)
    if args.report:
        print(json.dumps(report(runs or None, args.out_dir), indent=1, default=str))
        return 0
    if not args.metric:
        print("[ERROR] --metric {ansrel,ctxrel} is required (one stage at a time)")
        return 2
    if not runs:
        print("[ERROR] no run files and no official generation run registered")
        return 2
    runs = {k: v for k, v in runs.items() if v[0] in METRICS[args.metric]["conditions"]}
    if not args.run:
        print(plan_text(plan(args.metric, runs, out_paths(args.metric, args.out_dir)["records"],
                             tuple(j for j in JUDGES if args.judges is None or j.judge_id in args.judges))))
        return 0
    files = [p for _, p in runs.values()]
    sm._gate(files)
    return run_guarded(args.metric, runs, files, args)


if __name__ == "__main__":
    sys.exit(main())
