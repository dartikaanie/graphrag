"""Read-only dashboard views over the 3-judge panel's OFFICIAL outputs.

Two sources, both plain JSONL on disk (never written here):
  - Hallucination: llm/evaluation/results/panel_official_*_judge-v2.3_panel.jsonl
    (one aggregated record per question; run_id == History's history_id).
    Stage-2 candidate/exploratory panel files are NOT official and are skipped.
  - Secondary metrics (answer/context relevance): results/official_secondary_panel/
    <metric>_panel_official.jsonl (per-judge raw records) + *_failures.jsonl.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HALL_PANEL_DIR = REPO_ROOT / "llm" / "evaluation" / "results"
SECONDARY_DIR = REPO_ROOT / "results" / "official_secondary_panel"

HALL_LABELS = ["FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def hallucination_by_run() -> dict[str, dict[str, Any]]:
    """run_id -> panel hallucination summary. Same rate definitions as
    judge_agreement.compute_run_label_summary: hall = SEBAGIAN + PENUH;
    'excl' drops ABSTAIN from the denominator, 'incl' counts ABSTAIN as non-factual."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(HALL_PANEL_DIR.glob("panel_official_*_judge-v2.3_panel.jsonl")):
        by_run: dict[str, list[dict]] = defaultdict(list)
        for r in _read_jsonl(path):
            by_run[r["run_id"]].append(r)
        for run_id, recs in by_run.items():
            n = len(recs)
            dist: dict[str, int] = defaultdict(int)
            for r in recs:
                dist[r.get("panel_label") or "INCOMPLETE"] += 1
            n_abstain = dist.get("ABSTAIN", 0)
            n_hall = dist.get("HALUSINASI_SEBAGIAN", 0) + dist.get("HALUSINASI_PENUH", 0)
            n_non_abstain = n - n_abstain
            out[run_id] = {
                "run_id": run_id,
                "run_label": recs[0].get("run_label"),
                "panel_id": recs[0].get("panel_id"),
                "prompt_version": recs[0].get("prompt_version"),
                "source": path.name,
                "n": n,
                "label_distribution": dict(dist),
                "abstention_rate": round(n_abstain / n, 3) if n else None,
                "hall_rate_excl_abstain": round(n_hall / n_non_abstain, 3) if n_non_abstain else None,
                "hall_rate_incl_abstain": round((n_hall + n_abstain) / n, 3) if n else None,
                "unanimous_rate": round(sum(1 for r in recs if r.get("unanimous")) / n, 3) if n else None,
                "n_incomplete": sum(1 for r in recs if r.get("panel_incomplete")),
            }
    return out


def secondary_panel() -> dict[str, Any]:
    """metric -> per-judge x per-run-file summary (n, label counts, mean score,
    cost) + failure counts. A judge with fewer records than the others is
    simply shown with its own n -- completeness is the reader's call, and the
    CLI report (secondary_panel.py) still owns the 'all three judges done' gate."""
    metrics: dict[str, Any] = {}
    if not SECONDARY_DIR.exists():
        return {"metrics": metrics}
    for path in sorted(SECONDARY_DIR.glob("*_panel_official.jsonl")):
        recs = _read_jsonl(path)
        if not recs:
            continue
        metric = recs[0].get("metric") or path.name.split("_")[0]
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for r in recs:
            src = Path(r.get("source_file", "")).name
            key = (r.get("judge_id", "?"), src)
            row = rows.setdefault(key, {
                "judge_id": key[0], "judge_model": r.get("judge_model"), "source_file": src,
                "prompt_version": r.get("prompt_version"), "n": 0, "labels": defaultdict(int),
                "scores": [], "cost_usd": 0.0, "n_unparsed": 0,
            })
            row["n"] += 1
            row["labels"][r.get("label") or "UNPARSED"] += 1
            if r.get("score") is not None:
                row["scores"].append(r["score"])
            if not r.get("parse_ok", True) or r.get("call_failed"):
                row["n_unparsed"] += 1
            row["cost_usd"] += r.get("estimated_cost_usd") or 0.0
        out_rows = []
        for row in rows.values():
            scores = row.pop("scores")
            row["labels"] = dict(row["labels"])
            row["mean_score"] = round(sum(scores) / len(scores), 4) if scores else None
            row["cost_usd"] = round(row["cost_usd"], 4)
            out_rows.append(row)
        out_rows.sort(key=lambda x: (x["source_file"], x["judge_id"]))
        failures_path = path.with_name(path.name.replace(".jsonl", "_failures.jsonl"))
        failures = _read_jsonl(failures_path) if failures_path.exists() else []
        fail_by_judge: dict[str, int] = defaultdict(int)
        for fr in failures:
            fail_by_judge[fr.get("judge_id", "?")] += 1
        metrics[metric] = {
            "metric": metric,
            "file": path.name,
            "judges": sorted({r["judge_id"] for r in out_rows}),
            "total_cost_usd": round(sum(r["cost_usd"] for r in out_rows), 4),
            "rows": out_rows,
            "failures_by_judge": dict(fail_by_judge),
        }
    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# Per-run evaluation view: every panel metric, panel summary + per-judge
# breakdown + judge identity. Used by History detail and Compare.
# ---------------------------------------------------------------------------

_ORDINAL = {
    "ansrel": ("TIDAK_MENJAWAB", "MENJAWAB_SEBAGIAN", "MENJAWAB"),
    "ctxrel": ("IRRELEVANT", "PARTIAL", "RELEVANT"),
}
_METRIC_TITLE = {"ansrel": "Answer Relevance", "ctxrel": "Context Relevance"}
N_PANEL_JUDGES = 3


def _dist_stats(labels: list[str | None]) -> dict[str, Any]:
    dist: dict[str, int] = defaultdict(int)
    for lab in labels:
        dist[lab or "UNPARSED"] += 1
    return dict(dist)


def _hall_rates(dist: dict[str, int], n: int) -> dict[str, Any]:
    n_abs = dist.get("ABSTAIN", 0)
    n_hall = dist.get("HALUSINASI_SEBAGIAN", 0) + dist.get("HALUSINASI_PENUH", 0)
    non_abs = n - n_abs
    return {
        "n": n, "label_distribution": dist,
        "abstention_rate": round(n_abs / n, 3) if n else None,
        "hall_rate_excl_abstain": round(n_hall / non_abs, 3) if non_abs else None,
        "hall_rate_incl_abstain": round((n_hall + n_abs) / n, 3) if n else None,
    }


def _hallucination_detail() -> tuple[dict[str, dict], dict[str, str]]:
    """run_id -> {panel, per_judge, judges}; plus source basename -> run_id."""
    summary = hallucination_by_run()
    detail: dict[str, dict] = {}
    basename_to_run: dict[str, str] = {}
    for path in sorted(HALL_PANEL_DIR.glob("panel_official_*_judge-v2.3.jsonl")):
        if path.name.endswith("_panel.jsonl") or path.name.endswith("_failures.jsonl"):
            continue
        by_run_judge: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in _read_jsonl(path):
            by_run_judge[r["run_id"]][r["judge_id"]].append(r)
            if r.get("source_file"):
                basename_to_run[Path(r["source_file"]).name] = r["run_id"]
        for run_id, judges in by_run_judge.items():
            per_judge = []
            for jid, recs in sorted(judges.items()):
                per_judge.append({
                    "judge_id": jid, "judge_model": recs[0].get("judge_model"),
                    "provider": recs[0].get("provider"), "temperature": recs[0].get("temperature"),
                    **_hall_rates(_dist_stats([r.get("label") for r in recs]), len(recs)),
                })
            detail[run_id] = {"panel": summary.get(run_id), "per_judge": per_judge}
    return detail, basename_to_run


def _secondary_by_run(basename_to_run: dict[str, str]) -> dict[str, dict[str, dict]]:
    """metric -> run_id -> {panel, per_judge, judges_present, complete}."""
    out: dict[str, dict[str, dict]] = {}
    if not SECONDARY_DIR.exists():
        return out
    for path in sorted(SECONDARY_DIR.glob("*_panel_official.jsonl")):
        recs = _read_jsonl(path)
        if not recs:
            continue
        metric = recs[0].get("metric") or path.name.split("_")[0]
        if metric == "ctxrel":
            continue  # keyed by (question, answer, text_hash), not source_file -- see _ctxrel_by_run
        order = _ORDINAL.get(metric, ())
        score_of = {r["label"]: r["score"] for r in recs if r.get("label") and r.get("score") is not None}
        by_run: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            run_id = basename_to_run.get(Path(r.get("source_file", "")).name)
            if run_id:
                by_run[run_id].append(r)
        out[metric] = {}
        for run_id, rr in by_run.items():
            per_judge, items = [], defaultdict(dict)
            for jid in sorted({r["judge_id"] for r in rr}):
                jr = [r for r in rr if r["judge_id"] == jid]
                ok = [r for r in jr if r.get("parse_ok") and r.get("label") is not None]
                scores = [r["score"] for r in ok if r.get("score") is not None]
                per_judge.append({
                    "judge_id": jid, "judge_model": jr[0].get("judge_model"), "provider": jr[0].get("provider"),
                    "temperature": jr[0].get("temperature"), "n": len(jr), "n_unparsed": len(jr) - len(ok),
                    "label_distribution": _dist_stats([r["label"] for r in ok]),
                    "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                })
                for r in ok:
                    items[r["question_id"]][jid] = r["label"]
            panel_labels, n_incomplete, n_unresolved = [], 0, 0
            for labs in items.values():
                votes = list(labs.values())
                if len(votes) < 2:
                    n_incomplete += 1
                    continue
                top, cnt = max(((v, votes.count(v)) for v in set(votes)), key=lambda x: x[1])
                if cnt >= 2:
                    panel_labels.append(top)
                elif len(votes) == 3 and order:
                    panel_labels.append(sorted(votes, key=order.index)[1])
                else:
                    n_unresolved += 1
            ps = [score_of[lab] for lab in panel_labels if lab in score_of]
            judges_present = sorted({p["judge_id"] for p in per_judge})
            # All three judges present => a panel verdict exists. Items one judge
            # failed on are decided by the other two when they agree (otherwise
            # 'unresolved'); the counts below keep that visible, never hidden.
            n_missing_votes = sum(1 for labs in items.values() if len(labs) < len(judges_present))
            complete = len(judges_present) >= N_PANEL_JUDGES
            out[metric][run_id] = {
                "metric": metric, "title": _METRIC_TITLE.get(metric, metric),
                "judges_present": judges_present,
                "complete": complete,
                "per_judge": per_judge,
                # A combined verdict from a partial panel would mislead (mostly
                # 'unresolved' pairs), so it is withheld until all 3 judges finish.
                "panel": None if not complete else {
                    "n_decided": len(panel_labels), "n_incomplete": n_incomplete, "n_unresolved": n_unresolved,
                    "n_missing_votes": n_missing_votes,
                    "label_distribution": _dist_stats(panel_labels),
                    "mean_score": round(sum(ps) / len(ps), 4) if ps else None,
                },
            }
    return out


CTX_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}


def _panel_label(votes: list[str], order: tuple[str, ...]) -> str | None:
    """Majority of >=2 agreeing judges; median on the ordinal scale if all
    three differ; None if undecidable (<2 usable votes, or 2 that disagree)."""
    if len(votes) < 2:
        return None
    top, cnt = max(((v, votes.count(v)) for v in set(votes)), key=lambda x: x[1])
    if cnt >= 2:
        return top
    if len(votes) == 3 and order:
        return sorted(votes, key=order.index)[1]
    return None


def _ctx_stats(items: list[tuple], label_of: dict[tuple, str]) -> dict[str, Any]:
    """items: [(key, question_id, rank)] for ONE run's retrieved context items;
    label_of: key -> label (this judge, or the panel). Mirrors
    llm_judge_context_relevance_v1.compute_run_context_relevance: items with no
    label are counted as unjudged, never as IRRELEVANT."""
    dist: dict[str, int] = defaultdict(int)
    scores, r1, r2 = [], [], []
    questions, q_relevant = set(), set()
    for key, qid, rank in items:
        questions.add(qid)
        lab = label_of.get(key)
        if lab not in CTX_SCORE:
            continue
        dist[lab] += 1
        sc = CTX_SCORE[lab]
        scores.append(sc)
        (r1 if rank == 1 else r2).append(sc)
        if lab == "RELEVANT":
            q_relevant.add(qid)
    n = len(scores)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "n": n, "n_unjudged": len(items) - n, "label_distribution": dict(dist), "mean_score": mean(scores),
        "mean_score_rank1": mean(r1), "mean_score_rank2plus": mean(r2),
        "n_questions": len(questions),
        "pct_questions_with_relevant": round(100 * len(q_relevant) / len(questions), 2) if questions else None,
    }


def _run_context_items(run_id: str) -> list[tuple] | None:
    """[(key, question_id, rank)] from the run's output file; None if no file."""
    import hashlib
    from app.services import history_service

    try:
        detail = history_service.get_history_detail(run_id)
    except Exception:
        return None
    path = history_service.resolve_output_path((detail or {}).get("output_path"), run_id.split("-", 1)[0])
    if not path:
        return None
    items = []
    for rec in _read_jsonl(path):
        for rank, it in enumerate(rec.get("retrieved_context") or [], start=1):
            h = hashlib.sha256((it.get("chunk_text", "") or "").encode("utf-8")).hexdigest()
            items.append(((rec.get("question_id"), it.get("answer_id"), h), rec.get("question_id"), rank))
    return items


def _ctxrel_by_run(run_ids: list[str]) -> dict[str, dict | None]:
    path = SECONDARY_DIR / "ctxrel_panel_official.jsonl"
    recs = _read_jsonl(path) if path.exists() else []
    order = _ORDINAL["ctxrel"]
    by_judge: dict[str, dict[tuple, dict]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for r in recs:
        if r.get("parse_ok") and r.get("label") is not None:
            by_judge[r["judge_id"]][(r.get("question_id"), r.get("answer_id"), r.get("text_hash"))] = r
        meta.setdefault(r.get("judge_id", "?"), r)

    out: dict[str, dict | None] = {}
    for run_id in run_ids:
        if run_id.startswith("A-"):
            out[run_id] = {"per_judge": [], "panel": None, "complete": False, "judges_present": [],
                           "note": "Not applicable: condition A has no retrieved context."}
            continue
        if not recs:
            out[run_id] = None
            continue
        items = _run_context_items(run_id)
        if items is None:
            out[run_id] = None
            continue
        keys = {k for k, _, _ in items}
        per_judge = []
        for jid in sorted(by_judge):
            st = _ctx_stats(items, {k: v["label"] for k, v in by_judge[jid].items() if k in keys})
            if st["n"] == 0:
                continue
            m = meta[jid]
            per_judge.append({
                "judge_id": jid, "judge_model": m.get("judge_model"), "provider": m.get("provider"),
                "temperature": m.get("temperature"), "n_unparsed": 0, **st,
            })
        if not per_judge:
            out[run_id] = None
            continue
        judges_present = [j["judge_id"] for j in per_judge]
        complete = len(per_judge) >= N_PANEL_JUDGES
        panel = None
        if complete:
            labels = {}
            for k in keys:
                votes = [by_judge[j][k]["label"] for j in judges_present if k in by_judge[j]]
                lab = _panel_label(votes, order)
                if lab:
                    labels[k] = lab
            n_missing = sum(1 for k in keys if sum(k in by_judge[j] for j in judges_present) < len(judges_present))
            panel = {**_ctx_stats(items, labels), "n_incomplete": 0, "n_missing_votes": n_missing,
                     "n_unresolved": len(keys) - len(labels) - sum(
                         1 for k in keys if k not in labels and sum(k in by_judge[j] for j in judges_present) < 2)}
        out[run_id] = {"metric": "ctxrel", "title": _METRIC_TITLE["ctxrel"], "judges_present": judges_present,
                       "complete": complete, "per_judge": per_judge, "panel": panel}
    return out


def _claim_counts(recs: list[dict]) -> dict[str, Any]:
    """Faithfulness + citation support from one judge's per-claim labels.
    Mirrors llm/evaluation/official_report.claim_stats: usable + attempted
    answers only; claims with a null value are left out, never imputed."""
    n_total = len(recs)
    n_unusable = n_abstain = 0
    f_yes = f_no = c_yes = c_no = 0
    f_macro, c_macro = [], []
    for r in recs:
        if r.get("call_failed") or r.get("parse_error") or not isinstance(r.get("answer_attempted"), bool):
            n_unusable += 1
            continue
        if not r["answer_attempted"]:
            n_abstain += 1
            continue
        fy = fn = cy = cn = 0
        for cl in r.get("claims") or []:
            sup = str(cl.get("supported_by_context") or "").upper()
            if sup in ("YES", "NO"):
                fn += 1
                fy += sup == "YES"
            cs = str(cl.get("cited_item_supports") or "").upper()
            if cs in ("YES", "NO"):
                cn += 1
                cy += cs == "YES"
        f_yes, f_no, c_yes, c_no = f_yes + fy, f_no + (fn - fy), c_yes + cy, c_no + (cn - cy)
        if fn:
            f_macro.append(fy / fn)
        if cn:
            c_macro.append(cy / cn)
    ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    f_n, c_n = f_yes + f_no, c_yes + c_no
    return {
        "n": n_total, "n_unusable": n_unusable, "n_abstain": n_abstain,
        "n_attempted": n_total - n_unusable - n_abstain,
        "n_answers_rated": len(f_macro), "claims_supported": f_yes, "claims_unsupported": f_no, "claims_rated": f_n,
        "faithfulness_micro": ratio(f_yes, f_n), "faithfulness_macro": mean(f_macro),
        "n_answers_cited": len(c_macro), "cit_supported": c_yes, "cit_unsupported": c_no, "cit_rated": c_n,
        "citation_micro": ratio(c_yes, c_n), "citation_macro": mean(c_macro),
    }


def _faithfulness_by_run(run_ids: list[str]) -> dict[str, dict | None]:
    by_run_judge: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(HALL_PANEL_DIR.glob("panel_official_*_judge-v2.3.jsonl")):
        if path.name.endswith("_panel.jsonl") or path.name.endswith("_failures.jsonl"):
            continue
        for r in _read_jsonl(path):
            by_run_judge[r["run_id"]][r["judge_id"]].append(r)
    out: dict[str, dict | None] = {}
    for run_id in run_ids:
        if run_id.startswith("A-"):
            out[run_id] = {"per_judge": [], "panel": None, "complete": False, "judges_present": [],
                           "note": "Not applicable: condition A has no retrieved context."}
            continue
        judges = by_run_judge.get(run_id)
        if not judges:
            out[run_id] = None
            continue
        per_judge = []
        for jid, recs in sorted(judges.items()):
            per_judge.append({
                "judge_id": jid, "judge_model": recs[0].get("judge_model"), "provider": recs[0].get("provider"),
                "temperature": recs[0].get("temperature"), **_claim_counts(recs),
            })
        complete = len(per_judge) >= N_PANEL_JUDGES
        panel = None
        if complete:
            tot = lambda k: sum(j[k] for j in per_judge)  # noqa: E731
            ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
            macro = lambda k: ([j[k] for j in per_judge if j[k] is not None])  # noqa: E731
            fm, cm = macro("faithfulness_macro"), macro("citation_macro")
            panel = {
                "claims_supported": tot("claims_supported"), "claims_unsupported": tot("claims_unsupported"),
                "claims_rated": tot("claims_rated"),
                "faithfulness_micro": ratio(tot("claims_supported"), tot("claims_rated")),
                "faithfulness_macro": round(sum(fm) / len(fm), 4) if fm else None,
                "cit_supported": tot("cit_supported"), "cit_rated": tot("cit_rated"),
                "citation_micro": ratio(tot("cit_supported"), tot("cit_rated")),
                "citation_macro": round(sum(cm) / len(cm), 4) if cm else None,
                "n_answers": per_judge[0]["n"], "n_abstain": per_judge[0]["n_abstain"],
            }
        out[run_id] = {"metric": "faithfulness", "title": "Faithfulness", "judges_present": [j["judge_id"] for j in per_judge],
                       "complete": complete, "per_judge": per_judge, "panel": panel}
    return out


METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "hallucination": {
        "unit": "one answer per question (reference-based: judged against the accepted answer + the judge's own knowledge)",
        "labels": [
            {"label": "FAKTUAL", "meaning": "answer is factually correct"},
            {"label": "HALUSINASI_SEBAGIAN", "meaning": "partly hallucinated"},
            {"label": "HALUSINASI_PENUH", "meaning": "fully hallucinated"},
            {"label": "ABSTAIN", "meaning": "answer declines / does not attempt"},
        ],
        "calculation": "Hallucination = HALUSINASI_SEBAGIAN + HALUSINASI_PENUH. 'excl. ABSTAIN' = hallucinated / (N - ABSTAIN); "
                       "'incl. ABSTAIN' = (hallucinated + ABSTAIN) / N (abstain counted as non-factual). "
                       "Panel label = majority vote of the 3 judges.",
    },
    "faithfulness": {
        "unit": "one factual claim extracted from an attempted (non-ABSTAIN) answer, checked against the retrieved context only "
                "(not the accepted answer). Conditions B, C, D; condition A has no context.",
        "labels": [
            {"label": "YES", "meaning": "claim is supported by the retrieved context (counts as faithful)"},
            {"label": "NO", "meaning": "claim is not supported by the retrieved context"},
        ],
        "calculation": "Per judge: faithfulness (micro) = claims with supported_by_context = YES / claims rated YES or NO, pooled over "
                       "all answers of the run. Macro = mean over answers of that same per-answer ratio (answers with ≥1 rated claim). "
                       "ABSTAIN answers, unusable records and claims with a null value are left out, never counted as 0 or 1. "
                       "Panel (micro) = the three judges' claim counts pooled: ΣYES / Σrated. Panel (macro) = mean of the three judges' "
                       "macro values. Citation support uses the same recipe over cited claims (cited_item_supports).",
    },
    "ansrel": {
        "unit": "one answer per question (answer vs. the question)",
        "labels": [
            {"label": "MENJAWAB", "meaning": "answers the question", "score": 1.0},
            {"label": "MENJAWAB_SEBAGIAN", "meaning": "answers it partly", "score": 0.5},
            {"label": "TIDAK_MENJAWAB", "meaning": "does not answer it", "score": 0.0},
        ],
        "calculation": "Mean score = average of label scores. Panel label = majority of the 3 judges; if all three differ, "
                       "the median on the ordinal scale (TIDAK_MENJAWAB < MENJAWAB_SEBAGIAN < MENJAWAB). "
                       "Two judges that disagree = unresolved.",
    },
    "ctxrel": {
        "unit": "one retrieved context item (chunk) per question, judged against the question; condition A has none",
        "labels": [
            {"label": "RELEVANT", "meaning": "context item is relevant to the question", "score": 1.0},
            {"label": "PARTIAL", "meaning": "partly relevant", "score": 0.5},
            {"label": "IRRELEVANT", "meaning": "not relevant", "score": 0.0},
        ],
        "calculation": "Mean score = average of item scores. '≥1 RELEVANT' = % of questions with at least one RELEVANT item. "
                       "Rank 1 / Rank 2+ = mean score of the top-ranked item vs. the rest. Items without a judgment are "
                       "counted as unjudged, never as IRRELEVANT. Panel label per item = majority of 3 judges, median if all differ.",
    },
}


def evaluation_for_runs(run_ids: list[str]) -> dict[str, Any]:
    hall, basename_to_run = _hallucination_detail()
    secondary = _secondary_by_run(basename_to_run)
    ctx = _ctxrel_by_run(run_ids)
    faith = _faithfulness_by_run(run_ids)
    metrics = [
        {"metric": "hallucination", "title": "Hallucination (reference-based, judge-v2.3)",
         "definition": METRIC_DEFINITIONS["hallucination"], "runs": {i: hall.get(i) for i in run_ids}},
        {"metric": "faithfulness", "title": "Faithfulness & Citation Support (grounding in retrieved context)",
         "definition": METRIC_DEFINITIONS["faithfulness"], "runs": faith},
        {"metric": "ansrel", "title": _METRIC_TITLE["ansrel"], "definition": METRIC_DEFINITIONS["ansrel"],
         "runs": {i: secondary.get("ansrel", {}).get(i) for i in run_ids}},
        {"metric": "ctxrel", "title": _METRIC_TITLE["ctxrel"] + " (stage 2)", "definition": METRIC_DEFINITIONS["ctxrel"],
         "runs": ctx},
    ]
    return {"run_ids": run_ids, "metrics": metrics}
