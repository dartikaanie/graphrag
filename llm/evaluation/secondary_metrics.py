#!/usr/bin/env python3
"""
secondary_metrics.py
====================
Secondary metrics of the official evaluation (Amendment 16): CONTEXT RELEVANCE (B, C, D) and ANSWER RELEVANCE
(A, B, C, D). Both are judged by ONE judge, the registry judge "primary" (meta-llama/Llama-3.3-70B-Instruct, served
as -Turbo on DeepInfra), temperature 0.0, with the EXISTING prompts:

  context relevance  ctxrel-v2 (ctxrel_prompt_v2.py) through llm_judge_context_relevance_v1.run_batch: one
                     (question, context item) pair per call; the judge sees the question and the item text only
                     (no condition, scores, hop, trust, answer or reference).
  answer relevance   the prompt of llm_judge_answer_relevance.py (question + answer only; no reference, no context),
                     with the answer blinded by blind-v3 (citation markers and "the retrieved sources ..." meta
                     phrases removed), so the judge cannot tell the condition from the answer text.

These are reported next to the primary metrics; they never enter the hallucination rate or a pass/fail rule.

Default = PLAN (dry run): prints the number of judge calls and a cost estimate; NOTHING is called. Judging needs
--run, and an official (test-sample) run file additionally needs the same approval as the hallucination panel
(judge_versions.AMENDMENT14_APPROVED), so this module cannot be used to judge official runs early.

    python3 llm/evaluation/secondary_metrics.py                      # plan from the registered official runs
    python3 llm/evaluation/secondary_metrics.py --project            # plan from the matrix: 4 runs x 384 answers
    python3 llm/evaluation/secondary_metrics.py --run-files A=path B=path C=path D=path
    python3 llm/evaluation/secondary_metrics.py --run-files ... --run --max-cost-usd 1.2528   # judge (needs the key)

Outputs (written only with --run): results/official_secondary/{ctxrel,ansrel}_official.jsonl (+ *_failures.jsonl)
--run safeguards (2026-10-28): see the block "Operational safeguards" below the gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
for p in (EVAL_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm import archive_paths  # noqa: E402

JUDGE_ID = "primary"
CTXREL_PROMPT = "ctxrel-v2"
ANSREL_PROMPT = "ansrel-v1"
OUT_DIR = REPO_ROOT / "results" / "official_secondary"
CTXREL_FILE = OUT_DIR / "ctxrel_official.jsonl"
ANSREL_FILE = OUT_DIR / "ansrel_official.jsonl"
ANSREL_LABEL_SCORE = {"MENJAWAB": 1.0, "MENJAWAB_SEBAGIAN": 0.5, "TIDAK_MENJAWAB": 0.0}
# rough token sizes per call, from the judge-v1 / ctxrel-v1 cost plans (an ESTIMATE, labelled as such)
EST_TOKENS = {"ctxrel": (1500, 60), "ansrel": (1200, 80)}
_LOCK = threading.Lock()


def _read(path) -> list[dict]:
    path = archive_paths.resolve(path)
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Plan (dry run): counts only, no client is created
# ---------------------------------------------------------------------------

def plan(runs: dict[str, tuple[str, str]]) -> dict:
    """runs: label -> (condition, output file). Judge calls: context relevance = unique (question, answer, text)
    items over the B/C/D files; answer relevance = one call per answer of every file. Counts are exact for
    existing files."""
    ctx_items: set = set()
    per_run = {}
    for label, (condition, path) in runs.items():
        recs = _read(path)
        n_items = 0
        for r in recs:
            for it in r.get("retrieved_context") or []:
                if condition == "A":
                    continue
                n_items += 1
                ctx_items.add((r["question_id"], it.get("answer_id"), _text_hash(it.get("chunk_text", ""))))
        per_run[label] = {"condition": condition, "n_answers": len(recs),
                          "ansrel_calls": len(recs), "ctxrel_items_before_dedup": n_items}
    calls = {"ctxrel": len(ctx_items), "ansrel": sum(v["ansrel_calls"] for v in per_run.values())}
    return _with_cost({"source": "run files", "per_run": per_run, "calls": calls})


def project(n_sample: int = 384, k: int = 5) -> dict:
    """Upper-bound plan from the matrix alone (no output files yet): 4 runs x n answers; B, C, D each k items."""
    per_run = {lab: {"condition": c, "n_answers": n_sample, "ansrel_calls": n_sample,
                     "ctxrel_items_before_dedup": 0 if c == "A" else n_sample * k}
               for lab, c in (("A", "A"), ("B-grounded", "B"), ("C-primary-grounded", "C"), ("D-grounded", "D"))}
    calls = {"ctxrel": sum(v["ctxrel_items_before_dedup"] for v in per_run.values()),
             "ansrel": sum(v["ansrel_calls"] for v in per_run.values())}
    return _with_cost({"source": f"projection: 4 runs x {n_sample} answers x top-{k} (upper bound, before dedup)",
                       "per_run": per_run, "calls": calls})


def _with_cost(p: dict) -> dict:
    from judge_clients import JUDGE_REGISTRY

    cfg = JUDGE_REGISTRY[JUDGE_ID]
    cost = 0.0
    for kind, n in p["calls"].items():
        tin, tout = EST_TOKENS[kind]
        cost += n * (tin * cfg.price_per_m_input + tout * cfg.price_per_m_output) / 1e6
    p.update(judge={"id": JUDGE_ID, "model": cfg.model, "temperature": 0.0, "ctxrel_prompt": CTXREL_PROMPT,
                    "ansrel_prompt": ANSREL_PROMPT},
             total_calls=sum(p["calls"].values()), estimated_cost_usd=round(cost, 3),
             note="estimate from rough per-call token sizes; no call was made")
    return p


def plan_text(p: dict) -> str:
    lines = [f"Secondary metrics plan ({p['source']})", f"judge: {p['judge']}", ""]
    for label, v in p["per_run"].items():
        lines.append(f"  {label:<20} cond {v['condition']}  answers {v['n_answers']:>4}  answer-relevance calls "
                     f"{v['ansrel_calls']:>4}  context items {v['ctxrel_items_before_dedup']:>5}")
    c = p["calls"]
    lines += ["", f"context-relevance calls : {c['ctxrel']}", f"answer-relevance calls  : {c['ansrel']}",
              f"TOTAL judge calls       : {p['total_calls']}", f"estimated cost          : ~${p['estimated_cost_usd']} "
              f"({p['note']})", "NO CALL WAS MADE."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judging (only with --run)
# ---------------------------------------------------------------------------

def _gate(files: list[str]) -> None:
    import judge_versions as jv

    purpose = jv.run_purpose(files)
    if purpose == jv.PURPOSE_OFFICIAL and not jv.AMENDMENT14_APPROVED:
        raise jv.JudgeVersionError("official run files: judging is refused until Amendment 14 is approved "
                                   "(judge_versions.AMENDMENT14_APPROVED), same gate as the hallucination panel")


# ---------------------------------------------------------------------------
# Operational safeguards of --run (2026-10-28; same pattern as judge_panel.py). They change NOTHING about what is
# asked, to whom, or how an answer is scored: prompts, judge, temperature, blinding and scoring are untouched.
#   --max-cost-usd (required) + a runtime spend guard; ledger booking from the stored records (at start and end);
#   torn-last-line repair; foreign-records guard; run registration; [identity] check of the output files.
# ---------------------------------------------------------------------------

COST_RESERVE_OUT_TOKENS = 400   # guard assumption only: completion tokens reserved per call (the answers are tiny JSON)
CTXREL_CHUNK = 200              # context-relevance calls per guarded batch (run_batch itself is unchanged)


class SecondaryGuardError(RuntimeError):
    pass


def _price() -> tuple[float, float]:
    from judge_clients import JUDGE_REGISTRY

    cfg = JUDGE_REGISTRY[JUDGE_ID]
    return cfg.price_per_m_input, cfg.price_per_m_output


def record_cost(r: dict) -> float:
    """USD of one stored judge record, from its token counts at the registry price (failed calls carry none)."""
    pin, pout = _price()
    return ((r.get("prompt_tokens") or 0) * pin + (r.get("completion_tokens") or 0) * pout) / 1e6


def failures_path_of(out: Path) -> Path:
    return Path(out).with_name(Path(out).stem + "_failures.jsonl")


def stored_cost(out: Path) -> float:
    return round(sum(record_cost(r) for r in _read_tolerant(out) + _read_tolerant(failures_path_of(out))), 6)


def _read_tolerant(path) -> list[dict]:
    """Records of a JSONL file; a torn LAST line (the process died mid-write) is skipped, any other bad line raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in open(path) if line.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith("\n"):
                continue
            raise
    return out


class SpendGuard:
    """Runtime cap: a batch starts only if the spend so far plus the worst case of every batch in flight plus the worst
    case of this one stays within max_cost_usd. Skipped calls are counted and reported."""

    def __init__(self, max_cost_usd: float):
        self.max_cost_usd, self.spent, self.reserved, self.n_skipped = max_cost_usd, 0.0, 0.0, 0
        self._lock = threading.Lock()

    def worst_case(self, kind: str, n_calls: int = 1) -> float:
        pin, pout = _price()
        return n_calls * (2 * EST_TOKENS[kind][0] * pin + COST_RESERVE_OUT_TOKENS * pout) / 1e6

    def reserve(self, kind: str, n_calls: int = 1):
        w = self.worst_case(kind, n_calls)
        with self._lock:
            if self.spent + self.reserved + w > self.max_cost_usd:
                self.n_skipped += n_calls
                return None
            self.reserved += w
            return w

    def settle(self, reserved: float, cost: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += cost


def pending_calls(runs: dict[str, tuple[str, str]], ctx_file: Path, ans_file: Path) -> dict:
    """Calls still to make (not yet stored as usable), from the run files and the output files."""
    import llm_judge_context_relevance_v1 as cx

    model = _model()
    ctx_files = [p for c, p in runs.values() if c != "A"]
    done_ctx = cx.load_already_done(ctx_file)
    n_ctx = sum(1 for it in (cx.load_items(ctx_files, None) if ctx_files else [])
                if (it["question_id"], it["answer_id"], it["text_hash"], JUDGE_ID, model, CTXREL_PROMPT) not in done_ctx)
    done_ans = _ansrel_done(ans_file)
    n_ans = sum(1 for _, p in runs.values() for r in _read(p)
                if (str(archive_paths.canonical(p)), r["question_id"], model, ANSREL_PROMPT) not in done_ans)
    return {"ctxrel": n_ctx, "ansrel": n_ans}


def _model() -> str:
    from judge_clients import JUDGE_REGISTRY

    return JUDGE_REGISTRY[JUDGE_ID].model


def _ansrel_done(ans_file: Path) -> set:
    """Keys of USABLE answer-relevance records. Failed calls live in the failures file and are retried on resume."""
    return {_ansrel_key(r) for r in _read_tolerant(ans_file) if r.get("label") is not None}


def reconcile_ledger(out: Path, status: str, log=print) -> float:
    """Idempotent DeepInfra ledger update from the STORED records of `out` (and its failures file): books the part of
    their cost that no ledger entry tagged with this file covers yet. Safe after a killed run and safe to call twice."""
    from llm import run_registry as rr

    out = Path(out)
    cost = stored_cost(out)
    entries = rr.load().get("deepinfra_spend") or []
    booked = round(sum(float(e.get("cost_usd") or 0) for e in entries if e.get("file") == out.name), 6)
    delta = round(cost - booked, 6)
    if delta > 1e-6:
        rr.record_deepinfra_spend(f"{status} secondary metrics {out.name} (from the stored records)", delta, file=out.name)
        log(f"[ledger] {out.name}: stored records cost ${cost:.4f}, booked ${booked:.4f}: added ${delta:.4f}")
        return delta
    if delta < -1e-6:
        log(f"[WARN] ledger holds ${booked:.4f} for {out.name}, stored records only ${cost:.4f}; nothing changed")
    return 0.0


def _expected(runs: dict[str, tuple[str, str]]) -> dict:
    import llm_judge_context_relevance_v1 as cx

    ctx_files = [p for c, p in runs.values() if c != "A"]
    ctx = {(it["question_id"], it["answer_id"], it["text_hash"]) for it in (cx.load_items(ctx_files, None) if ctx_files else [])}
    ans = {(str(archive_paths.canonical(p)), r["question_id"]) for _, p in runs.values() for r in _read(p)}
    return {"ctxrel": ctx, "ansrel": ans}


def assert_no_foreign_records(runs: dict[str, tuple[str, str]], ctx_file: Path, ans_file: Path) -> None:
    """Official runs: an existing output file may hold only records of the current inputs (a stale or foreign file would
    otherwise be 'resumed' into and counted)."""
    exp = _expected(runs)
    ctx_keys = {(r.get("question_id"), r.get("answer_id"), r.get("text_hash")) for r in _read_tolerant(ctx_file)}
    ans_keys = {(r.get("source_file"), r.get("question_id")) for r in _read_tolerant(ans_file)}
    bad = []
    if ctx_keys - exp["ctxrel"]:
        bad.append(f"{Path(ctx_file).name}: {len(ctx_keys - exp['ctxrel'])} record(s) not among the current inputs")
    if ans_keys - exp["ansrel"]:
        bad.append(f"{Path(ans_file).name}: {len(ans_keys - exp['ansrel'])} record(s) not among the current inputs")
    if bad:
        raise SecondaryGuardError("--run refused: " + "; ".join(bad))


def identity_report(runs: dict[str, tuple[str, str]], ctx_file: Path, ans_file: Path) -> dict:
    """What the two official output files hold, against the inputs and the frozen id lists (printed after every run)."""
    from llm.sample_split import load_frozen_ids

    test = {int(i) for i in load_frozen_ids("test")["ids"]}
    dev = {int(i) for i in load_frozen_ids("dev")["ids"]}
    exp = _expected(runs)
    ctx = _read_tolerant(ctx_file)
    ans = _read_tolerant(ans_file)
    ctx_keys = [(r.get("question_id"), r.get("answer_id"), r.get("text_hash")) for r in ctx]
    ans_keys = [(r.get("source_file"), r.get("question_id")) for r in ans]
    ids = {int(k[0]) for k in ctx_keys} | {int(k[1]) for k in ans_keys}
    res = {"ctxrel": {"records": len(ctx), "unique_keys": len(set(ctx_keys)), "expected": len(exp["ctxrel"]),
                      "keys_equal_expected": set(ctx_keys) == exp["ctxrel"],
                      "parse_errors": sum(1 for r in ctx if r.get("parse_error")),
                      "failures_file_records": len(_read_tolerant(failures_path_of(ctx_file)))},
           "ansrel": {"records": len(ans), "unique_keys": len(set(ans_keys)), "expected": len(exp["ansrel"]),
                      "keys_equal_expected": set(ans_keys) == exp["ansrel"],
                      "unusable": sum(1 for r in ans if r.get("label") is None),
                      "failures_file_records": len(_read_tolerant(failures_path_of(ans_file)))},
           "dev_ids": len(ids & dev), "ids_outside_test_list": len(ids - test)}
    res["ok"] = (res["ctxrel"]["records"] == res["ctxrel"]["unique_keys"] == res["ctxrel"]["expected"]
                 and res["ctxrel"]["keys_equal_expected"]
                 and res["ansrel"]["records"] == res["ansrel"]["unique_keys"] == res["ansrel"]["expected"]
                 and res["ansrel"]["keys_equal_expected"] and res["dev_ids"] == 0 and res["ids_outside_test_list"] == 0)
    return res


def register_secondary_run(status: str, runs: dict, ctx_file: Path, ans_file: Path, cost: float, started: str,
                           n_calls: int, log=print) -> None:
    """Registers the secondary-metrics run (same status as its inputs, kind 'judge' like the panel runs)."""
    import subprocess
    from llm import run_registry as rr

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    run_id = f"secondary_metrics@{started[:19]}"
    try:
        rr.register({"run_id": run_id, "status": status, "kind": "judge", "condition": None, "grounding": None,
                     "alpha": None, "n": n_calls, "seed": None, "generator": None,
                     "judge_config": {"judge_id": JUDGE_ID, "model": _model(), "temperature": 0.0,
                                      "ctxrel_prompt": CTXREL_PROMPT, "ansrel_prompt": ANSREL_PROMPT,
                                      "metrics": "context relevance (B, C, D) + answer relevance (A, B, C, D); secondary"},
                     "commit": head or None, "created_at": started,
                     "files": {"ctxrel": rr._rel(ctx_file), "ansrel": rr._rel(ans_file)},
                     "cost_usd": cost, "cost_source": "sum of token counts x registry price of this run's records",
                     "notes": f"inputs: {sorted(runs)}", "frozen_params_hash": rr.frozen_params_hash()})
        log(f"[registry] secondary run {run_id} registered ({status}); spend ${cost:.4f} recorded")
    except rr.RegistryError as e:
        log(f"[WARN] secondary run not registered: {e}")


def run_ctxrel(files: list[str], workers: int = 4, guard: "SpendGuard | None" = None, ctx_file=None) -> dict:
    import llm_judge_context_relevance_v1 as cx
    import os

    out = Path(ctx_file or CTXREL_FILE)
    items = cx.load_items(files, None)
    out.parent.mkdir(parents=True, exist_ok=True)
    fail = failures_path_of(out)
    qp = os.getenv("QUESTIONS_PARQUET")
    if guard is None:
        return cx.run_batch(items, [JUDGE_ID], qp, out, fail, workers, prompt_version=CTXREL_PROMPT)
    # guarded: the same run_batch, called on batches of CTXREL_CHUNK pending items; the spend of every batch is read back
    # from the stored records before the next one may start
    model = _model()
    done = cx.load_already_done(out)
    pending = [it for it in items
               if (it["question_id"], it["answer_id"], it["text_hash"], JUDGE_ID, model, CTXREL_PROMPT) not in done]
    total = {"judged": 0, "skipped": len(items) - len(pending), "failed": 0, "parse_errors": 0, "budget_skipped": 0}
    for i in range(0, len(pending), CTXREL_CHUNK):
        chunk = pending[i:i + CTXREL_CHUNK]
        w = guard.reserve("ctxrel", len(chunk))
        if w is None:
            total["budget_skipped"] += len(chunk)
            continue
        before = stored_cost(out)
        res = cx.run_batch(chunk, [JUDGE_ID], qp, out, fail, workers, prompt_version=CTXREL_PROMPT)
        guard.settle(w, stored_cost(out) - before)
        for k in ("judged", "skipped", "failed", "parse_errors"):
            total[k] += res[k]
    return total


def _ansrel_key(r: dict) -> tuple:
    return (str(archive_paths.canonical(r.get("source_file", ""))), r.get("question_id"), r.get("judge_model"),
            r.get("prompt_version"))


def run_ansrel(files: list[str], workers: int = 4, guard: "SpendGuard | None" = None, ans_file=None) -> dict:
    import os
    import llm_judge_context_relevance_v1 as cx
    import llm_judge_answer_relevance as ar
    from _judge_common import html_to_text, load_question_bodies
    from judge_blinding import blind_candidate_v3
    from judge_clients import get_judge_client

    out = Path(ans_file or ANSREL_FILE)
    client, config = get_judge_client(JUDGE_ID)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _ansrel_done(out)
    tasks = []
    for path in files:
        recs = _read(path)
        bodies = load_question_bodies(os.getenv("QUESTIONS_PARQUET"), [r["question_id"] for r in recs])
        for r in recs:
            rec_key = (str(archive_paths.canonical(path)), r["question_id"], config.model, ANSREL_PROMPT)
            if rec_key not in done:
                tasks.append((path, r, html_to_text(bodies.get(r["question_id"]), 3000) if bodies.get(r["question_id"]) else ""))

    def one(task):
        path, r, body = task
        w = guard.reserve("ansrel") if guard else 0.0
        if w is None:
            return None  # not made: it could have pushed the spend over --max-cost-usd
        messages = ar.build_answer_relevance_prompt(r.get("title", ""), body, blind_candidate_v3(r.get("llm_answer", "")))
        payload, attempts = cx.call_judge_with_retry(client, config, messages)
        parsed = ar._parse_answer_relevance_output(payload["raw"]) if payload else None
        label = parsed["answer_relevance_label"] if parsed else None
        rec = {"source_file": str(archive_paths.canonical(path)), "question_id": r["question_id"],
               "judge_id": JUDGE_ID, "judge_model": config.model, "prompt_version": ANSREL_PROMPT, "temperature": 0.0,
               "label": label, "score": ANSREL_LABEL_SCORE.get(label), "parse_error": parsed is None,
               "call_failed": payload is None, "response_model": payload and payload["response_model"],
               "prompt_tokens": payload and payload["prompt_tokens"],
               "completion_tokens": payload and payload["completion_tokens"],
               "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        if guard:
            guard.settle(w, record_cost(rec))
        return rec

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            rec = fut.result()
            if rec is None:
                continue
            # usable records -> the output file; failed / unparseable calls -> the failures file (retried on resume)
            with _LOCK, open(out if rec["label"] is not None else failures_path_of(out), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            n_ok += rec["label"] is not None
            n_fail += rec["label"] is None
    return {"answer_relevance_judged": n_ok, "answer_relevance_failed": n_fail, "skipped": len(done),
            "budget_skipped": guard.n_skipped if guard else 0}


# ---------------------------------------------------------------------------
# Reading back for the report
# ---------------------------------------------------------------------------

def report_section(output_file, condition: str) -> dict:
    """Secondary metrics of one run from the stored judgments; None where nothing is judged yet."""
    out = {"judge": f"{JUDGE_ID} (Llama-3.3-70B-Instruct), temperature 0.0", "context_relevance": None,
           "answer_relevance": None}
    canon = str(archive_paths.canonical(output_file))
    if condition != "A" and CTXREL_FILE.exists():
        import llm_judge_context_relevance_v1 as cx

        by_key = cx.load_ctxrel_records([CTXREL_FILE], prompt_version=CTXREL_PROMPT)
        m = cx.compute_run_context_relevance(str(archive_paths.resolve(output_file)), by_key)
        out["context_relevance"] = m if m["n_items_judged"] else None
    recs = [r for r in _read(ANSREL_FILE) if str(archive_paths.canonical(r.get("source_file", ""))) == canon
            and r.get("prompt_version") == ANSREL_PROMPT and r.get("label")]
    if recs:
        n = len(recs)
        out["answer_relevance"] = {
            "n_judged": n, "mean_score": round(sum(r["score"] for r in recs) / n, 4),
            "label_share": {k: round(sum(1 for r in recs if r["label"] == k) / n, 4) for k in ANSREL_LABEL_SCORE}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-files", nargs="*", default=None, help="LABEL=PATH (A=, B=, C=, D= or cell ids)")
    ap.add_argument("--project", action="store_true", help="plan from the matrix (4 runs x 384), no files needed")
    ap.add_argument("--n-sample", type=int, default=384)
    ap.add_argument("--run", action="store_true", help="actually judge (default: plan only, nothing is called)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost-usd", type=float, default=None,
                    help="spend cap of this --run (required with --run); a batch that could exceed it is not started")
    ap.add_argument("--out-dir", default=None, help="output directory (default results/official_secondary); "
                    "refused for official run files")
    args = ap.parse_args(argv)

    runs: dict[str, tuple[str, str]] = {}
    if args.run_files:
        for spec in args.run_files:
            label, _, path = spec.partition("=")
            runs[label] = (label.strip()[0].upper() if label[0].upper() in "ABCD" else "?", path)
    elif not args.project:
        from llm import run_registry as rr

        for e in rr.runs((rr.STATUS_OFFICIAL,)):
            if e["kind"] == "generation":
                runs[f"{e['condition']}-{e.get('run_id', '')[:8]}"] = (e["condition"], e["files"]["output"])
    p = plan(runs) if runs else project(args.n_sample)
    if not runs and not args.project:
        p["source"] = "no official run is registered yet -> " + p["source"]
    print(plan_text(p))
    if not args.run:
        return 0
    if not runs:
        print("[ERROR] --run needs --run-files or registered official runs")
        return 2
    files = [path for _, path in runs.values()]
    _gate(files)
    return run_guarded(runs, files, args)


def run_guarded(runs: dict, files: list[str], args) -> int:
    """--run with the operational safeguards (cap, ledger, identity, registration, resume)."""
    import judge_versions as jv
    from llm.common.resume import repair_torn_tail

    if args.max_cost_usd is None:
        print("[ERROR] --run needs --max-cost-usd")
        return 2
    official = jv.run_purpose(files) == jv.PURPOSE_OFFICIAL
    if official and args.out_dir:
        print("[ERROR] official run files write to results/official_secondary only (no --out-dir)")
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    ctx_file, ans_file = (CTXREL_FILE, ANSREL_FILE) if not args.out_dir else (out_dir / CTXREL_FILE.name, out_dir / ANSREL_FILE.name)
    status = None
    if official:
        from llm import run_registry as rr
        from judge_panel import assert_official_inputs

        try:
            assert_official_inputs(files)  # test ids only, no dev id, no duplicate
            entries = rr.assert_reportable(files, (rr.STATUS_OFFICIAL,))
            assert_no_foreign_records(runs, ctx_file, ans_file)
        except Exception as e:  # PanelGuardError, RegistryError, SecondaryGuardError
            print(f"[ERROR] {e}")
            return 2
        status = rr.STATUS_OFFICIAL
        remaining = rr.deepinfra_remaining()
        print(f"[registry] {len(entries)} {status} run(s); DeepInfra budget remaining ${remaining:.4f}")
    for f in (ctx_file, ans_file, failures_path_of(ctx_file), failures_path_of(ans_file)):
        repair_torn_tail(f)  # a run killed while writing leaves a torn last line: cut it off
    if official:
        reconcile_ledger(ctx_file, status)
        reconcile_ledger(ans_file, status)
    pend = pending_calls(runs, ctx_file, ans_file)
    pin, pout = _price()
    projected = sum(n * (EST_TOKENS[k][0] * pin + EST_TOKENS[k][1] * pout) / 1e6 for k, n in pend.items())
    print(f"[plan] pending calls: context relevance {pend['ctxrel']}, answer relevance {pend['ansrel']}; projected "
          f"${projected:.4f} (no buffer); limit ${args.max_cost_usd:.4f}")
    if projected > args.max_cost_usd:
        print(f"[ABORT] projected cost ${projected:.4f} exceeds --max-cost-usd {args.max_cost_usd}")
        return 2
    if official and projected > remaining:
        print(f"[ABORT] projected ${projected:.4f} exceeds the remaining DeepInfra budget ${remaining:.4f}")
        return 2
    started = datetime.now(timezone.utc).isoformat()
    guard = SpendGuard(args.max_cost_usd)
    before = stored_cost(ctx_file) + stored_cost(ans_file)
    ctx_paths = [path for c, path in runs.values() if c != "A"]
    print(json.dumps(run_ctxrel(ctx_paths, args.workers, guard, ctx_file), indent=1))
    print(json.dumps(run_ansrel(files, args.workers, guard, ans_file), indent=1))
    cost = round(stored_cost(ctx_file) + stored_cost(ans_file) - before, 6)
    if guard.n_skipped:
        print(f"[BUDGET] {guard.n_skipped} call(s) not made: they could have pushed the spend over "
              f"--max-cost-usd {args.max_cost_usd}")
    print(f"[cost] this run ${cost:.4f} (cap ${args.max_cost_usd:.4f})")
    if official:
        reconcile_ledger(ctx_file, status)
        reconcile_ledger(ans_file, status)
        register_secondary_run(status, runs, ctx_file, ans_file, cost, started, sum(pend.values()))
        ident = identity_report(runs, ctx_file, ans_file)
        print(f"[identity] {json.dumps(ident)}")
        print("[identity] OK" if ident["ok"] else "[identity] MISMATCH: do not use these files before it is explained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
