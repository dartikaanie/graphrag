#!/usr/bin/env python3
"""
devlarge_alpha_stage1.py
========================
Stage 1 of the alpha selection for C-v2 (docs/DECISION_C_SCORING.md, Amendment 19) on dev-large: context relevance of
the four v5 retrieval-only runs (alpha 0, 0.1, 0.25, 0.5; 150 questions each), judged by the Amendment 18 panel.

A NEW entry point. secondary_panel.py, judge_panel.py and every official file are untouched: their judges, RequestSpec,
run_stage (call_judge, retries, record format), BudgetGuard, reconcile_ledger and the ctxrel-v2 prompt are imported and
reused. Each unique (question, answer, text) triple is judged ONCE by the 3 judges (2-of-3 majority, as in
secondary_panel.aggregate_key) and shared by every alpha that retrieved it.

Per alpha: primary = share of questions with >= 1 RELEVANT item (panel label); secondary = mean panel ctxrel score per
item (RELEVANT 1, PARTIAL 0.5, IRRELEVANT 0; items without a panel label are left out).
Selection (Amendment 19): alpha = 0 is a reference and NOT eligible. The best primary wins; alphas within 2 percentage
points of it are tied and decided by the secondary (best secondary, within 0.02 tied); ties go to the SMALLER alpha. The
two best eligible alphas advance (the winner is removed and the rule applied again).

    python3 llm/evaluation/devlarge_alpha_stage1.py F0 F01 F025 F05                     # dry run: triples and cost
    python3 llm/evaluation/devlarge_alpha_stage1.py F0 F01 F025 F05 --run --max-cost-usd X
    python3 llm/evaluation/devlarge_alpha_stage1.py F0 F01 F025 F05 --report            # from stored records, no call

Resumable (usable records are kept, failed calls are retried). Spend is booked in the ledger as a non-official
"dev-stage2" entry tagged with the output file. Refuses official runs, non-devlarge files and non-v5 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
for _p in (EVAL_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import secondary_panel as sp  # noqa: E402

METRIC = "ctxrel"
ALPHAS = (0.0, 0.1, 0.25, 0.5)
REFERENCE_ALPHA = 0.0
PRIMARY_MARGIN = 0.02      # 2 percentage points
SECONDARY_MARGIN = 0.02
N_QUESTIONS = 150
OUT_DIR = REPO_ROOT / "results" / "devlarge_alpha_stage1"
LEDGER_STATUS = "dev-stage2"
log = print


class Stage1Error(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_runs(files) -> dict:
    """{alpha: [record]} after the checks: devlarge, v5, retrieval-only contexts, 150 questions, one config_hash, the four
    alphas exactly once, and not an official run."""
    import judge_versions as jv

    runs = {}
    for f in files:
        recs = [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
        if not recs:
            raise Stage1Error(f"{f}: empty")
        cfg = recs[0].get("config") or {}
        if cfg.get("sample_split") != "devlarge":
            raise Stage1Error(f"{f}: not a devlarge run (sample_split={cfg.get('sample_split')!r})")
        if cfg.get("c_retrieval_version") != "v5":
            raise Stage1Error(f"{f}: not a v5 run (c_retrieval_version={cfg.get('c_retrieval_version')!r})")
        if len({r.get("config_hash") for r in recs}) != 1:
            raise Stage1Error(f"{f}: more than one config_hash")
        if jv.run_purpose([str(f)]) == jv.PURPOSE_OFFICIAL:
            raise Stage1Error(f"{f}: is an official run")
        alpha = float(cfg["alpha"])
        if alpha in runs:
            raise Stage1Error(f"{f}: alpha {alpha} given twice")
        qids = [r["question_id"] for r in recs]
        if len(qids) != len(set(qids)) or len(qids) != N_QUESTIONS:
            raise Stage1Error(f"{f}: {len(qids)} records / {len(set(qids))} unique questions, expected {N_QUESTIONS}")
        if not any(r.get("retrieved_context") for r in recs):
            raise Stage1Error(f"{f}: no retrieved_context")
        runs[alpha] = recs
    if set(runs) != set(ALPHAS):
        raise Stage1Error(f"alphas {sorted(runs)} != {list(ALPHAS)}")
    return runs


def keys_by_question(recs: list[dict]) -> dict:
    import llm_judge_context_relevance_v1 as cx

    return {r["question_id"]: sorted({(r["question_id"], it.get("answer_id"), cx._text_hash(it.get("chunk_text", "") or ""))
                                      for it in (r.get("retrieved_context") or [])}, key=str)
            for r in recs}


def _bodies(qids):
    import os
    from _judge_common import load_question_bodies

    return load_question_bodies(os.getenv("QUESTIONS_PARQUET"), qids)


def unique_triples(runs: dict) -> list[dict]:
    """One item (with the ctxrel-v2 messages every judge receives) per unique triple over all alphas."""
    import llm_judge_context_relevance_v1 as cx
    from _judge_common import html_to_text

    seen = {}
    for alpha in sorted(runs):
        for r in runs[alpha]:
            for it in r.get("retrieved_context") or []:
                chunk = it.get("chunk_text", "") or ""
                k = (r["question_id"], it.get("answer_id"), cx._text_hash(chunk))
                seen.setdefault(k, {"question_id": k[0], "answer_id": k[1], "text_hash": k[2], "title": r.get("title", ""),
                                    "tags": r.get("tags"), "chunk_text": chunk})
    bodies = _bodies(sorted({k[0] for k in seen}))
    build = cx.get_prompt_builder(sp.METRICS[METRIC]["prompt"])
    items = []
    for k in sorted(seen, key=str):
        s = seen[k]
        html = bodies.get(s["question_id"], "")
        items.append({"question_id": s["question_id"], "answer_id": s["answer_id"], "text_hash": s["text_hash"],
                      "messages": build(s["title"], html_to_text(html) if html else "", s["tags"], s["chunk_text"])})
    return items


def plan(n_triples: int, out: Path, judges=None) -> dict:
    judges = judges or sp.JUDGES
    tin, tout = sp.METRICS[METRIC]["est_tokens"]
    done = sp.done_keys(METRIC, out)
    per = {}
    for j in judges:
        pending = n_triples - sum(1 for (_, jid) in done if jid == j.judge_id)
        unit = (tin * j.price_per_m_input + tout * j.price_per_m_output) / 1e6
        per[j.judge_id] = {"calls": n_triples, "pending": pending, "cost_usd": round(n_triples * unit, 4),
                           "pending_cost_usd": round(pending * unit, 4)}
    return {"unique_triples": n_triples, "per_judge": per, "total_calls": n_triples * len(judges),
            "projected_usd": round(sum(v["cost_usd"] for v in per.values()), 4),
            "pending_projected_usd": round(sum(v["pending_cost_usd"] for v in per.values()), 4)}


def plan_text(p: dict) -> str:
    lines = [f"Dev-large alpha stage 1 plan: {p['unique_triples']} unique (question, answer, text) triples x "
             f"{len(p['per_judge'])} judges = {p['total_calls']} calls"]
    lines += [f"  {j:<12} pending {v['pending']:>6}  projected ${v['cost_usd']:.4f}" for j, v in p["per_judge"].items()]
    lines += [f"TOTAL projected ${p['projected_usd']:.4f} (pending ${p['pending_projected_usd']:.4f}); fixed-size token estimate",
              "NO CALL WAS MADE."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scores and the selection rule
# ---------------------------------------------------------------------------

def panel_labels(recs: list[dict], expected: set) -> tuple[dict, dict]:
    """({key: panel label or None}, per-judge stats) from the stored records. A key with < 2 usable judge records, or
    two that disagree, has no label (None)."""
    rows = sp.label_rows_of(METRIC, recs)
    labels = {}
    for k in expected:
        agg = sp.aggregate_key(rows.get(k, {j.judge_id: None for j in sp.JUDGES}), METRIC)
        labels[k] = None if agg["panel_label"] in (sp.INCOMPLETE, sp.UNRESOLVED) else agg["panel_label"]
    judges = {}
    for j in sp.JUDGES:
        ok = sum(1 for k in expected if rows.get(k, {}).get(j.judge_id) is not None)
        judges[j.judge_id] = {"usable": ok, "unusable": len(expected) - ok,
                              "unusable_pct": round(100 * (len(expected) - ok) / len(expected), 3) if expected else 0.0}
    return labels, judges


def alpha_scores(by_question: dict, labels: dict) -> dict:
    scorer = sp.METRICS[METRIC]["score"]
    with_rel = sum(1 for keys in by_question.values() if any(labels.get(k) == "RELEVANT" for k in keys))
    items = [scorer[labels[k]] for keys in by_question.values() for k in keys if labels.get(k)]
    return {"n_questions": len(by_question), "questions_with_relevant": with_rel,
            "primary": with_rel / len(by_question) if by_question else 0.0, "n_items_scored": len(items),
            "secondary": sum(items) / len(items) if items else 0.0}


def _pick(pool: dict) -> tuple[float, str]:
    """pool {alpha: {"primary", "secondary"}} -> (winner, rule branch)."""
    top = max(v["primary"] for v in pool.values())
    tied = {a: v for a, v in pool.items() if top - v["primary"] <= PRIMARY_MARGIN + 1e-9}
    if len(tied) == 1:
        return next(iter(tied)), "primary (clear best, outside the 2 pp margin)"
    best2 = max(v["secondary"] for v in tied.values())
    tied2 = [a for a, v in tied.items() if best2 - v["secondary"] <= SECONDARY_MARGIN + 1e-9]
    if len(tied2) == 1:
        return tied2[0], "secondary (primary tied within 2 pp; best mean score outside the 0.02 margin)"
    return min(tied2), "tie -> smaller alpha (primary within 2 pp and secondary within 0.02)"


def select(scores: dict) -> dict:
    """scores {alpha: {"primary", "secondary", ...}}. Alpha 0 is a reference, never eligible. Two advance."""
    pool = {a: v for a, v in scores.items() if a != REFERENCE_ALPHA}
    advancing, branches = [], []
    for _ in range(2):
        if not pool:
            break
        a, branch = _pick(pool)
        advancing.append(a)
        branches.append(branch)
        pool = {k: v for k, v in pool.items() if k != a}
    return {"advancing": advancing, "rule_branches": branches, "eligible": sorted(a for a in scores if a != REFERENCE_ALPHA),
            "reference_not_eligible": REFERENCE_ALPHA,
            "margins": {"primary_pp": PRIMARY_MARGIN * 100, "secondary": SECONDARY_MARGIN},
            "note": "ties go to the smaller alpha; the margin comparison is to the best value, so it is not transitive"}


def _direction(secondary: dict) -> str:
    """How the mean score moves with alpha: 'decreasing' (strictly, over every alpha in order), 'lower at the highest alpha
    than at the lowest (not monotone)', or 'not lower at the highest alpha'."""
    a = sorted(secondary)
    vals = [secondary[x] for x in a]
    if all(y < x for x, y in zip(vals, vals[1:])):
        return "decreasing"
    return "lower at highest alpha (not monotone)" if vals[-1] < vals[0] else "not lower at highest alpha"


def cohen_kappa(pairs: list[tuple]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    cats = {c for pr in pairs for c in pr}
    pe = sum((sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n) for c in cats)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def per_judge_detail(runs: dict, recs: list[dict], expected: set, panel_secondary: dict | None = None) -> dict:
    """Descriptive, post-hoc. Read-only on the stored records; the selection never reads it."""
    rows = sp.label_rows_of(METRIC, recs)  # {key: {judge_id: label or None}}
    scorer = sp.METRICS[METRIC]["score"]
    ids = [j.judge_id for j in sp.JUDGES]
    by_q = {a: keys_by_question(runs[a]) for a in runs}
    judges = {}
    for jid in ids:
        per_alpha, secondary = {}, {}
        for a in sorted(runs):
            lab = lambda k: rows.get(k, {}).get(jid)  # noqa: E731
            items = [k for keys in by_q[a].values() for k in keys]
            used = [lab(k) for k in items if lab(k) is not None]
            counts = {c: used.count(c) for c in ("RELEVANT", "PARTIAL", "IRRELEVANT")}
            with_rel = sum(1 for keys in by_q[a].values() if any(lab(k) == "RELEVANT" for k in keys))
            n_q = len(by_q[a])
            sec = sum(scorer[x] for x in used) / len(used) if used else 0.0
            per_alpha[str(a)] = {"primary": with_rel / n_q if n_q else 0.0, "questions_with_relevant": with_rel,
                                 "secondary": sec, "labels": counts, "n_items": len(items),
                                 "n_unusable_left_out": len(items) - len(used)}
            secondary[a] = sec
        prim = {a: per_alpha[str(a)]["primary"] for a in runs}
        judges[jid] = {"per_alpha": per_alpha,
                       "ranking_by_primary": sorted(runs, key=lambda a: (-prim[a], a)),
                       "ranking_by_secondary": sorted(runs, key=lambda a: (-secondary[a], a)),
                       "secondary_direction": _direction(secondary)}
    panel_dir = _direction(panel_secondary) if panel_secondary else None
    for v in judges.values():
        v["same_direction_as_panel"] = None if panel_dir is None else v["secondary_direction"] == panel_dir
    agreement = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pairs = [(rows[k][a], rows[k][b]) for k in expected if rows.get(k, {}).get(a) and rows.get(k, {}).get(b)]
            agreement[f"{a}-{b}"] = {"n": len(pairs), "percent_agreement": round(100 * sum(x == y for x, y in pairs) / len(pairs), 2) if pairs else None,
                                     "cohen_kappa": cohen_kappa(pairs)}
    full = [k for k in expected if all(rows.get(k, {}).get(j) for j in ids)]
    return {"judges": judges, "panel_secondary_direction": panel_dir, "agreement": agreement,
            "unanimous_3way": {"n_with_3_usable": len(full), "n_unique_triples": len(expected),
                               "percent": round(100 * sum(len({rows[k][j] for j in ids}) == 1 for k in full) / len(full), 2) if full else None}}


def report(runs: dict, out: Path, report_dir: Path) -> dict:
    recs = sp.sm._read_tolerant(out)
    expected = {k for a in runs for keys in keys_by_question(runs[a]).values() for k in keys}
    labels, judges = panel_labels(recs, expected)
    bad = {j: v for j, v in judges.items() if v["unusable"] > sp.UNUSABLE_LIMIT * len(expected)}
    per_alpha = {a: alpha_scores(keys_by_question(runs[a]), labels) for a in sorted(runs)}
    res = {"metric": METRIC, "prompt_version": sp.METRICS[METRIC]["prompt"], "unique_triples": len(expected),
           "judges": judges, "completion_rule_ok": not bad, "per_alpha": {str(a): v for a, v in per_alpha.items()}}
    if bad:
        res["selection"] = None
        res["error"] = (f"completion rule failed (> {sp.UNUSABLE_LIMIT:.0%} unusable): {sorted(bad)}; no selection made")
    else:
        res["selection"] = select(per_alpha)
    res["per_judge_detail"] = per_judge_detail(runs, recs, expected, {a: v["secondary"] for a, v in per_alpha.items()})
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage1_report.json").write_text(json.dumps(res, indent=1))
    (report_dir / "stage1_report.md").write_text(report_md(res))
    return res


def report_md(res: dict) -> str:
    L = ["# Dev-large alpha selection, stage 1 (Amendment 19)", "",
         f"Context relevance by the panel ({res['prompt_version']}), {res['unique_triples']} unique triples judged once and shared.", "",
         "| alpha | questions with >=1 RELEVANT | primary | mean ctxrel score | items scored |", "|---|---|---|---|---|"]
    for a, v in res["per_alpha"].items():
        ref = " (reference, not eligible)" if float(a) == REFERENCE_ALPHA else ""
        L.append(f"| {a}{ref} | {v['questions_with_relevant']}/{v['n_questions']} | {v['primary']:.1%} | "
                 f"{v['secondary']:.3f} | {v['n_items_scored']} |")
    L += ["", "Unusable per judge: " + ", ".join(f"{j} {v['unusable_pct']}%" for j, v in res["judges"].items()), ""]
    sel = res.get("selection")
    if sel is None:
        L.append(f"**No selection:** {res.get('error')}")
    else:
        L.append(f"**Advancing to stage 2: alpha {sel['advancing']}**")
        L += [f"- pick {i + 1}: alpha {a}, rule branch: {b}" for i, (a, b) in enumerate(zip(sel["advancing"], sel["rule_branches"]))]
        L += ["", "Rule: alpha 0 is a reference (not eligible); primary margin 2 pp, secondary margin 0.02; ties go to the "
                  "smaller alpha. " + sel["note"] + "."]
    if res.get("per_judge_detail"):
        L += per_judge_md(res["per_judge_detail"], res.get("selection"))
    return "\n".join(L) + "\n"


def per_judge_md(d: dict, sel: dict | None) -> list[str]:
    L = ["", "## Per-judge detail (descriptive, post-hoc; does not affect the selection)", "",
         "| judge | alpha | primary | mean score | RELEVANT | PARTIAL | IRRELEVANT | left out (unusable) |",
         "|---|---|---|---|---|---|---|---|"]
    for jid, v in d["judges"].items():
        for a, x in v["per_alpha"].items():
            c = x["labels"]
            L.append(f"| {jid} | {a} | {x['primary']:.1%} | {x['secondary']:.3f} | {c['RELEVANT']} | {c['PARTIAL']} | "
                     f"{c['IRRELEVANT']} | {x['n_unusable_left_out']}/{x['n_items']} |")
    L += ["", f"Alpha ranking implied by each judge (best first); panel secondary direction: {d['panel_secondary_direction']}.", ""]
    for jid, v in d["judges"].items():
        L.append(f"- {jid}: by primary {v['ranking_by_primary']}, by secondary {v['ranking_by_secondary']}; secondary "
                 f"{v['secondary_direction']}; same direction as the panel: {v['same_direction_as_panel']}")
    L += ["", "Agreement between judges on the unique triples:", "", "| pair | n | agreement | Cohen's kappa |", "|---|---|---|---|"]
    for pair, x in d["agreement"].items():
        L.append(f"| {pair} | {x['n']} | {x['percent_agreement']}% | {x['cohen_kappa']} |")
    u = d["unanimous_3way"]
    L += ["", f"3-way unanimous: {u['percent']}% of {u['n_with_3_usable']} triples with 3 usable labels "
              f"({u['n_unique_triples']} unique triples).", ""]
    advancing = sel["advancing"] if sel else None
    L.append(f"The panel (2-of-3) selection is unchanged: advancing alphas {advancing}.")
    return L


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(runs: dict, out: Path, args, clients=None) -> int:
    import judge_panel as jp
    from llm.common.resume import repair_torn_tail

    if args.max_cost_usd is None:
        print("[ERROR] --run needs --max-cost-usd")
        return 2
    for f in (out, sp.failures_path_of(out)):
        repair_torn_tail(f)
    jp.reconcile_ledger(out, LEDGER_STATUS, log)
    items = unique_triples(runs)
    p = plan(len(items), out)
    print(plan_text(p))
    if p["pending_projected_usd"] > args.max_cost_usd:
        print(f"[ABORT] pending projected ${p['pending_projected_usd']:.4f} exceeds --max-cost-usd {args.max_cost_usd}")
        return 2
    spec = sp.RequestSpec(METRIC)
    guard = jp.BudgetGuard(args.max_cost_usd, spec.max_tokens,
                           lambda msgs: sum(len(m["content"]) for m in msgs) / jp.CHARS_PER_TOKEN_EST)
    before = sp._stored_cost(out)
    stats = sp.run_stage(METRIC, items, clients or jp.make_clients(sp.JUDGES), out, guard, args.workers)
    print(json.dumps(stats))
    if guard.n_skipped:
        print(f"[BUDGET] {guard.n_skipped} call(s) not made (--max-cost-usd {args.max_cost_usd})")
    print(f"[cost] this run ${sp._stored_cost(out) - before:.4f} (cap ${args.max_cost_usd:.4f})")
    jp.reconcile_ledger(out, LEDGER_STATUS, log)
    return 0


def main(argv=None, clients=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs=4, help="the four v5 retrieval-only JSONL files (alpha 0, 0.1, 0.25, 0.5, any order)")
    ap.add_argument("--run", action="store_true", help="actually judge (default: dry run, nothing is called)")
    ap.add_argument("--report", action="store_true", help="write the report from the stored records (no call)")
    ap.add_argument("--max-cost-usd", type=float, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    try:
        runs = read_runs(args.files)
    except Stage1Error as e:
        print(f"[ERROR] {e}")
        return 2
    out_dir = Path(args.out_dir)
    out = out_dir / "ctxrel_panel_devlarge_stage1.jsonl"
    if args.run:
        rc = run(runs, out, args, clients)
        if rc:
            return rc
    elif not args.report:
        print(plan_text(plan(len(unique_triples(runs)), out)))
        return 0
    res = report(runs, out, out_dir)
    print(json.dumps(res.get("selection") or {"error": res.get("error")}, indent=1))
    print(f"[report] {out_dir / 'stage1_report.md'}")
    return 0 if res.get("selection") else 1


if __name__ == "__main__":
    sys.exit(main())
