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
    python3 llm/evaluation/secondary_metrics.py --run-files ... --run   # judge (needs DEEPINFRA_API_KEY)

Outputs (written only with --run): results/official_secondary/{ctxrel,ansrel}_official.jsonl
"""

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


def run_ctxrel(files: list[str], workers: int = 4) -> dict:
    import llm_judge_context_relevance_v1 as cx
    import os

    items = cx.load_items(files, None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return cx.run_batch(items, [JUDGE_ID], os.getenv("QUESTIONS_PARQUET"), CTXREL_FILE,
                        CTXREL_FILE.with_name(CTXREL_FILE.stem + "_failures.jsonl"), workers,
                        prompt_version=CTXREL_PROMPT)


def _ansrel_key(r: dict) -> tuple:
    return (str(archive_paths.canonical(r.get("source_file", ""))), r.get("question_id"), r.get("judge_model"),
            r.get("prompt_version"))


def run_ansrel(files: list[str], workers: int = 4) -> dict:
    import os
    import llm_judge_context_relevance_v1 as cx
    import llm_judge_answer_relevance as ar
    from _judge_common import html_to_text, load_question_bodies
    from judge_blinding import blind_candidate_v3
    from judge_clients import get_judge_client

    client, config = get_judge_client(JUDGE_ID)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {_ansrel_key(r) for r in _read(ANSREL_FILE)}
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
        messages = ar.build_answer_relevance_prompt(r.get("title", ""), body, blind_candidate_v3(r.get("llm_answer", "")))
        payload, attempts = cx.call_judge_with_retry(client, config, messages)
        parsed = ar._parse_answer_relevance_output(payload["raw"]) if payload else None
        label = parsed["answer_relevance_label"] if parsed else None
        return {"source_file": str(archive_paths.canonical(path)), "question_id": r["question_id"],
                "judge_id": JUDGE_ID, "judge_model": config.model, "prompt_version": ANSREL_PROMPT, "temperature": 0.0,
                "label": label, "score": ANSREL_LABEL_SCORE.get(label), "parse_error": parsed is None,
                "call_failed": payload is None, "response_model": payload and payload["response_model"],
                "prompt_tokens": payload and payload["prompt_tokens"],
                "completion_tokens": payload and payload["completion_tokens"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat()}

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tasks]):
            rec = fut.result()
            with _LOCK, open(ANSREL_FILE, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            n_ok += rec["label"] is not None
            n_fail += rec["label"] is None
    return {"answer_relevance_judged": n_ok, "answer_relevance_failed": n_fail, "skipped": len(done)}


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
    print(json.dumps(run_ctxrel([path for c, path in runs.values() if c != "A"], args.workers), indent=1))
    print(json.dumps(run_ansrel(files, args.workers), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
