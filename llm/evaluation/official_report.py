"""
official_report.py
==================
End-to-end report of the official evaluation (Amendment 14 draft, plan
step 8), from the run registry and STORED outputs only -- no API call:

  per official run (and condition; the four runs of Amendment 16: A, B-grounded, C-primary-grounded, D-grounded):
    - the hallucination report for the panel AND each judge: (a) rate excl. ABSTAIN (primary), (b) rate with
      ABSTAIN counted as failed = (hallucinated + ABSTAIN) / all usable answers, ABSTAIN rate, the severity split
      (full / partial) and the share of unverifiable claims
    - faithfulness and citation support (B, C, D) aggregated from the per-claim labels of the panel records
    - secondary metrics (context relevance B/C/D, answer relevance A-D), if judged -- see secondary_metrics.py
    - inter-judge Fleiss' kappa and Krippendorff's alpha (nominal, 3 categories), and the share of answers on which
      all three judges gave the same label
    - cosine similarity (mean, median), citation validity (NF2), latency
      (retrieval p50/p95 per answer; run seconds per answer vs the NF3 target)
    - retrieval / generation / judging statistics
  the pre-registered comparisons C vs A, C vs B, C vs D (rate differences, the rule-(e) consistent/inconclusive
  flag, and per-question paired counts: C better / equal / worse),
  the alpha stage-2 result, the budget, and the limitations (Amendment 14 text
  + archived calibration evidence, labelled as from the archive).

Rates are "panel-detected hallucination rates" (Amendment 14 (a)).

    python3 llm/evaluation/official_report.py            # results/official_report.{json,md}
    python3 llm/evaluation/official_report.py --mock     # synthetic data, every output labelled MOCK
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
for p in (EVAL_DIR, REPO_ROOT, REPO_ROOT / "llm" / "c_graphrag"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from judge_binary import (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED,  # noqa: E402
                          category_of, is_unusable, severity_of)
from judge_panel_aggregate import UNRESOLVED, aggregate_labels, panel_category  # noqa: E402
import secondary_metrics  # noqa: E402
from llm import archive_paths  # noqa: E402

PANEL_ID = "panel-v2"
PROMPT_VERSION = "judge-v2.3"
JUDGES = ("llama", "deepseek", "qwen3-235b")
SENSITIVE_JUDGE = "deepseek"
CATS = (CATEGORY_ABSTAIN, CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED)
NF3_TARGET_SEC = 15.0   # NF3: p95 of per-answer end-to-end latency (record schema r3), Condition C
NF3_CONDITION = "C"
# Cell ids of llm/official_run_matrix.py: R1 = A, R3 = B-grounded, R5 = C-primary-grounded, R9 = D-grounded.
COMPARISONS = (("R5", "R1", "C vs A"), ("R5", "R3", "C vs B"), ("R5", "R9", "C vs D"))
# Legacy comparisons need legacy cells (C-ablation, plain); evaluated only if both runs exist, else omitted quietly.
LEGACY_COMPARISONS = (("R5", "R7", "C vs C-ablation (legacy)"), ("R4", "R2", "legacy"), ("R4", "R8", "legacy"),
                      ("R4", "R6", "legacy"))
OUT = REPO_ROOT / "results" / "official_report"
ARCHIVE_EVIDENCE = {
    "agreement_report": "llm/evaluation/results/panel_agreement_calibration_judge-v2.3.md",
    "interjudge_summary": "llm/evaluation/results/interjudge_summary_calibration.md",
    "interjudge_patterns": "llm/evaluation/results/interjudge_patterns_C_calibration.md",
    "stage1_of_record": "llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.json",
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def three_numbers(categories: list) -> dict:
    """categories: one 3-category label (or None = no usable label) per answer."""
    c = Counter(categories)
    h, n_ok, a = c[CATEGORY_HALLUCINATED], c[CATEGORY_NOT_HALLUCINATED], c[CATEGORY_ABSTAIN]
    usable = h + n_ok + a
    return {"n": len(categories), "n_usable": usable, "n_missing": c[None], "hallucinated": h, "not_hallucinated": n_ok,
            "abstain": a,
            # (a) primary: ABSTAIN leaves the denominator
            "rate_excl_abstain": _ratio(h, h + n_ok),
            # (b) ABSTAIN counted as failed: (hallucinated + ABSTAIN) / all usable answers (Amendment 14 (i))
            "rate_abstain_as_failed": _ratio(h + a, usable),
            "abstain_rate": _ratio(a, usable)}


def severity_split(severities: list) -> dict:
    """severities: one of "PENUH" / "SEBAGIAN" / "UNSPECIFIED" per HALLUCINATED answer. Shares are of the
    hallucinated answers; UNSPECIFIED = an error claim without severity (never imputed)."""
    c = Counter(severities)
    n = len(severities)
    return {"n_hallucinated": n, "full": c["PENUH"], "partial": c["SEBAGIAN"], "unspecified": c["UNSPECIFIED"],
            "full_share": _ratio(c["PENUH"], n), "partial_share": _ratio(c["SEBAGIAN"], n)}


def claim_stats(records: list[dict]) -> dict:
    """From usable, attempted judge records: unverifiable-claim share and the faithfulness / citation-support
    aggregates. Denominators (docs/OFFICIAL_RUN_MATRIX.md):
      unverifiable  = UNVERIFIABLE claims / all claims
      faithfulness  = claims with supported_by_context == YES / claims with a YES or NO value (micro, over claims);
                      macro = mean over answers of the same ratio (answers with >= 1 rated claim)
      citation_sup. = cited claims with cited_item_supports == YES / cited claims with a YES or NO value
    Claims with a null value are left out of the denominator, never imputed."""
    n_claims = n_unv = f_yes = f_n = c_yes = c_n = 0
    f_macro, c_macro = [], []
    for r in records:
        if is_unusable(r) or not r.get("answer_attempted"):
            continue
        fy = fn = cy = cn = 0
        for cl in r.get("claims") or []:
            n_claims += 1
            n_unv += str(cl.get("verdict", "")).upper() == "UNVERIFIABLE"
            sup = str(cl.get("supported_by_context") or "").upper()
            if sup in ("YES", "NO"):
                fn += 1
                fy += sup == "YES"
            cs = str(cl.get("cited_item_supports") or "").upper()
            if cs in ("YES", "NO"):
                cn += 1
                cy += cs == "YES"
        f_yes, f_n, c_yes, c_n = f_yes + fy, f_n + fn, c_yes + cy, c_n + cn
        if fn:
            f_macro.append(fy / fn)
        if cn:
            c_macro.append(cy / cn)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"n_claims": n_claims, "n_unverifiable": n_unv, "unverifiable_share": _ratio(n_unv, n_claims),
            "faithfulness": {"micro": _ratio(f_yes, f_n), "n_claims_rated": f_n, "macro": mean(f_macro),
                             "n_answers": len(f_macro)},
            "citation_support": {"micro": _ratio(c_yes, c_n), "n_claims_cited": c_n, "macro": mean(c_macro),
                                 "n_answers": len(c_macro)}}


def pool_claim_stats(per_judge: dict) -> dict:
    """Pool the three judges' claim counts (they judge the same answers, each with their own claim list)."""
    n_claims = sum(v["n_claims"] for v in per_judge.values())
    n_unv = sum(v["n_unverifiable"] for v in per_judge.values())
    out = {"n_claims": n_claims, "n_unverifiable": n_unv, "unverifiable_share": _ratio(n_unv, n_claims)}
    for key, num_key in (("faithfulness", "n_claims_rated"), ("citation_support", "n_claims_cited")):
        den = sum(v[key][num_key] for v in per_judge.values())
        num = sum((v[key]["micro"] or 0) * v[key][num_key] for v in per_judge.values())
        out[key] = {"micro": round(num / den, 4) if den else None, num_key: den}
    return out


def _ratio(x, n):
    return round(x / n, 4) if n else None


def fleiss_kappa(rows: list[list[str]], cats=CATS) -> float | None:
    rows = [r for r in rows if all(x is not None for x in r)]
    if not rows:
        return None
    m, n = len(rows[0]), len(rows)
    p_j = {c: sum(r.count(c) for r in rows) / (n * m) for c in cats}
    p_i = [(sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in rows]
    p_bar, p_e = sum(p_i) / n, sum(v * v for v in p_j.values())
    return None if p_e == 1 else round((p_bar - p_e) / (1 - p_e), 4)


def krippendorff_alpha_nominal(rows: list[list[str | None]]) -> float | None:
    """Nominal Krippendorff's alpha; rows = units, values = raters' labels (None = missing)."""
    units = [[x for x in r if x is not None] for r in rows]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    coinc = Counter()
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    coinc[(a, b)] += 1 / (m - 1)
    n_c = Counter()
    for (a, _b), v in coinc.items():
        n_c[a] += v
    n = sum(n_c.values())
    d_o = sum(v for (a, b), v in coinc.items() if a != b)
    d_e = sum(n_c[a] * n_c[b] for a in n_c for b in n_c if a != b) / (n - 1)
    return None if d_e == 0 else round(1 - d_o / d_e, 4)


def _pcts(xs: list[float]) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"p50": None, "p95": None, "n": 0}
    return {"p50": round(st.median(xs), 3), "p95": round(xs[max(0, int(round(0.95 * len(xs))) - 1)], 3), "n": len(xs)}


# ---------------------------------------------------------------------------
# Stores: where runs, answers and judge records come from
# ---------------------------------------------------------------------------

class RegistryStore:
    """Official runs from results/run_registry.json, outputs read through the archive resolver."""
    mock = False

    def __init__(self, include_dev_stage2: bool = False):
        from llm import run_registry as rr
        self.rr = rr
        self.reg = rr.load()
        statuses = (rr.STATUS_OFFICIAL,) + ((rr.STATUS_DEV_STAGE2,) if include_dev_stage2 else ())
        self.entries = rr.runs(statuses, reg=self.reg)

    def generation_runs(self) -> list[dict]:
        return [e for e in self.entries if e["kind"] == "generation"]

    def answers(self, run: dict) -> list[dict]:
        return _read(run["files"]["output"])

    def manifest(self, run: dict) -> dict:
        m = run["files"].get("manifest")
        return json.loads(archive_paths.resolve(m).read_text()) if m and archive_paths.resolve(m).exists() else {}

    def judge_records(self, run: dict) -> list[dict]:
        target = str(archive_paths.canonical(run["files"]["output"]))
        out = []
        for e in self.entries:
            if e["kind"] != "judge":
                continue
            for r in _read(e["files"]["output"]):
                if (r.get("panel_id") == PANEL_ID and r.get("prompt_version") == PROMPT_VERSION
                        and str(archive_paths.canonical(r.get("source_file", ""))) == target):
                    out.append(r)
        return out

    def unresolved_failures(self, run: dict) -> int:
        target = str(archive_paths.canonical(run["files"]["output"]))
        n = 0
        for e in self.entries:
            if e["kind"] == "judge":
                f = archive_paths.resolve(e["files"]["output"])
                f = f.with_name(f.stem + "_failures.jsonl")
                if f.exists():
                    n += sum(1 for r in _read(f) if str(archive_paths.canonical(r.get("source_file", ""))) == target)
        return n

    def budget(self) -> dict:
        return {"budget": self.reg.get("budget"), "spent_usd": self.rr.deepinfra_spent(self.reg),
                "remaining_usd": self.rr.deepinfra_remaining(self.reg), "ledger": self.reg.get("deepinfra_spend", [])}

    def judge_cost(self) -> float:
        return round(sum(e.get("cost_usd") or 0 for e in self.entries if e["kind"] == "judge"), 6)


def _read(rel) -> list[dict]:
    path = archive_paths.resolve(rel)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class MockStore:
    """Synthetic official evaluation (the four cells R1, R3, R5, R9; 24 answers each) for the dashboard
    walkthrough. Nothing here is a result; every output built from it carries mock = true."""
    mock = True
    CELLS = (("R1", "A", None, None), ("R3", "B", True, None), ("R5", "C", True, 0.25), ("R9", "D", True, None))
    RATES = {"R1": 0.35, "R3": 0.22, "R5": 0.15, "R9": 0.2}

    def __init__(self, n: int = 24, seed: int = 7):
        self.n, self.rng = n, random.Random(seed)
        self.runs = [{"run_id": f"MOCK-{r}", "status": "official", "kind": "generation", "condition": c, "grounding": g,
                      "alpha": a, "n": n, "seed": 42, "generator": "gpt-4o-mini", "judge_config": None,
                      "commit": "mock000", "created_at": "2026-10-20T00:00:00+00:00",
                      "files": {"output": f"MOCK/{r}.jsonl"}, "cost_usd": round(n * 0.0005, 4), "cost_source": "mock",
                      "notes": "MOCK", "_label": r} for r, c, g, a in self.CELLS]
        self._answers = {run["run_id"]: self._make_answers(run) for run in self.runs}
        self._judged = {run["run_id"]: self._make_judges(run) for run in self.runs}

    def generation_runs(self):
        return self.runs

    def _make_answers(self, run):
        out = []
        for q in range(self.n):
            ctx = [] if run["condition"] == "A" else [{"answer_id": 9000 + k, "chunk_text": f"[mock context {k}]"}
                                                      for k in range(self.rng.choice((5, 5, 5, 4, 0)))]
            out.append({"question_id": 1000 + q, "title": f"MOCK question {q}", "ground_truth_answer": "<p>mock reference</p>",
                        "llm_answer": "mock candidate answer", "retrieved_context": ctx,
                        "cosine_similarity": round(self.rng.uniform(0.4, 0.8), 4),
                        "has_valid_citation": None if run["condition"] == "A" else self.rng.random() < 0.85,
                        "retrieval_latency_sec": None if run["condition"] == "A" else round(self.rng.uniform(0.1, 2.5), 3),
                        "generation_latency_sec": round(self.rng.uniform(3.0, 9.0), 3),
                        "end_to_end_latency_sec": round(self.rng.uniform(3.5, 12.0), 3),
                        "usage_full": {"prompt_tokens": 2100, "completion_tokens": 400}})
        return out

    def _make_judges(self, run):
        recs, base = [], self.RATES[run["_label"]]
        for q in range(self.n):
            truth = self.rng.random()
            for j, bias in (("llama", -0.08), ("deepseek", 0.08), ("qwen3-235b", 0.0)):
                if truth < 0.08:
                    label = "ABSTAIN"
                else:
                    label = "HALUSINASI_PENUH" if self.rng.random() < base + bias else "FAKTUAL"
                recs.append({"panel_id": PANEL_ID, "prompt_version": PROMPT_VERSION, "judge_id": j, "question_id": 1000 + q,
                             "label": label, "derived_label": label, "parse_ok": True, "call_failed": False,
                             "parse_error": False, "answer_attempted": label != "ABSTAIN",
                             "claims": ([{"claim": "mock claim", "verdict": "CONTRADICTED", "severity": "CORE",
                                          "supported_by_context": "NO", "cited_items": ["K1"],
                                          "cited_item_supports": "NO",
                                          "evidence": '[R] "mock reference"'}] if label.startswith("HALU") else
                                        [{"claim": "mock claim", "verdict": "SUPPORTED", "severity": None,
                                          "supported_by_context": "YES", "cited_items": ["K1"],
                                          "cited_item_supports": "YES", "evidence": '[R] "mock reference"'},
                                         {"claim": "mock claim 2", "verdict": "UNVERIFIABLE", "severity": None,
                                          "supported_by_context": "YES", "cited_items": [],
                                          "cited_item_supports": None, "evidence": "mock"}]),
                             "estimated_cost_usd": 0.001, "source_file": run["files"]["output"]})
        return recs

    def answers(self, run):
        return self._answers[run["run_id"]]

    def manifest(self, run):
        return {"started_at_utc": "2026-10-20T00:00:00+00:00", "finished_at_utc": "2026-10-20T00:04:00+00:00",
                "item_counts": {"attempted": self.n, "succeeded": self.n, "failed": 0}}

    def judge_records(self, run):
        return self._judged[run["run_id"]]

    def unresolved_failures(self, run):
        return 0

    def budget(self):
        return {"budget": {"deepinfra_budget_usd": 8.0, "set_by": "MOCK"}, "spent_usd": 1.23, "remaining_usd": 6.77,
                "ledger": [{"label": "MOCK", "cost_usd": 1.23}]}

    def judge_cost(self):
        return round(sum(r["estimated_cost_usd"] for v in self._judged.values() for r in v), 4)


# ---------------------------------------------------------------------------
# Run labels (cell ids R1, R3, R5, R9 of the official run matrix; legacy cells R2, R4, R6-R8)
# ---------------------------------------------------------------------------

def matrix_label(run: dict, official: dict | None) -> str:
    if run.get("_label"):
        return run["_label"]
    if run["status"] == "dev-stage2":
        return f"S2 a={run['alpha']} grounding {'on' if run['grounding'] else 'off'}"
    from llm.official_run_matrix import build_matrix
    for row in build_matrix(official, include_legacy=True):
        if row["condition"] != run["condition"]:
            continue
        if row["condition"] != "A" and row["grounding"] != run["grounding"]:
            continue
        if row["condition"] == "C" and row["alpha"] != run["alpha"]:
            continue
        return row["run"]
    return f"{run['condition']}?"


# ---------------------------------------------------------------------------
# Per run
# ---------------------------------------------------------------------------

def run_section(store, run: dict, official: dict | None) -> dict:
    answers = store.answers(run)
    by_q = {}
    for r in store.judge_records(run):
        by_q.setdefault(int(r["question_id"]), {})[r["judge_id"]] = r
    qids = [int(a["question_id"]) for a in answers]
    judge_cats = {j: [] for j in JUDGES}
    judge_sev = {j: [] for j in JUDGES}
    panel_labels, rows, strict_rows = [], [], []
    for q in qids:
        recs = by_q.get(q, {})
        cats = {j: (category_of(recs[j]) if j in recs else None) for j in JUDGES}
        for j in JUDGES:
            judge_cats[j].append(cats[j])
            if cats[j] == CATEGORY_HALLUCINATED:
                judge_sev[j].append(severity_of(recs[j]))
        labels = {j: (recs[j]["label"] if j in recs and category_of(recs[j]) is not None else None) for j in JUDGES}
        panel_labels.append(aggregate_labels(labels, PANEL_ID))
        rows.append([cats[j] for j in JUDGES])
        strict_rows.append([labels[j] for j in JUDGES])
    panel_cats = [panel_category(p["panel_label"]) for p in panel_labels]
    panel_sev = [{"HALUSINASI_PENUH": "PENUH", "HALUSINASI_SEBAGIAN": "SEBAGIAN"}.get(p["panel_label"], "UNSPECIFIED")
                 for p in panel_labels if p["panel_label"] in ("HALUSINASI_PENUH", "HALUSINASI_SEBAGIAN",
                                                               "HALUSINASI_SEVERITY_UNSPECIFIED")]
    judge_claims = {j: claim_stats([by_q.get(q, {}).get(j) for q in qids if j in by_q.get(q, {})]) for j in JUDGES}
    complete = [(c, st_) for c, st_ in zip(rows, strict_rows) if None not in c]
    unanimous = {"n_complete": len(complete),
                 "same_label": _ratio(sum(1 for _, sr in complete if len(set(sr)) == 1), len(complete)),
                 "same_3cat_label": _ratio(sum(1 for c, _ in complete if len(set(c)) == 1), len(complete))}
    manifest = store.manifest(run)
    secs = None
    if manifest.get("started_at_utc") and manifest.get("finished_at_utc") and answers:
        dt = (datetime.fromisoformat(manifest["finished_at_utc"]) - datetime.fromisoformat(manifest["started_at_utc"]))
        secs = round(dt.total_seconds() / len(answers), 3)
    ctx_counts = [len(a.get("retrieved_context") or []) for a in answers]
    cos = [a.get("cosine_similarity") for a in answers if a.get("cosine_similarity") is not None]
    cited = [a.get("has_valid_citation") for a in answers if a.get("has_valid_citation") is not None]
    usage = [a.get("usage_full") or {} for a in answers]
    label_dist = {j: dict(Counter((by_q.get(q, {}).get(j) or {}).get("label", "missing") for q in qids)) for j in JUDGES}
    has_context = run["condition"] != "A"
    return {
        "run_id": run["run_id"], "label": matrix_label(run, official), "status": run["status"],
        "condition": run["condition"], "grounding": run["grounding"], "alpha": run["alpha"], "n": len(answers),
        "commit": run.get("commit"), "config_hash": run.get("config_hash"),
        "retrieval": {"context_items_mean": round(st.mean(ctx_counts), 2) if ctx_counts else None,
                      "empty_context": sum(1 for c in ctx_counts if c == 0) if run["condition"] != "A" else None,
                      "latency_sec": _pcts([a.get("retrieval_latency_sec") for a in answers])},
        "generation": {"n": len(answers), "abstain_panel": sum(1 for c in panel_cats if c == CATEGORY_ABSTAIN),
                       "tokens_in": sum(u.get("prompt_tokens", 0) for u in usage),
                       "tokens_out": sum(u.get("completion_tokens", 0) for u in usage),
                       "cost_usd": run.get("cost_usd"), "sec_per_answer": secs,
                       "failures": (manifest.get("item_counts") or {}).get("failed")},
        "judging": {"label_distribution": {**label_dist, "panel": dict(Counter(p["panel_label"] for p in panel_labels))},
                    "unresolved": sum(1 for p in panel_labels if p["panel_label"] == UNRESOLVED),
                    "incomplete_panels": sum(1 for p in panel_labels if p["panel_incomplete"]),
                    "missing_judge_records": {j: sum(1 for c in judge_cats[j] if c is None) for j in JUDGES},
                    "failures_unresolved": store.unresolved_failures(run)},
        "latency": latency_section(answers, secs, run["condition"]),
        "results": {"panel": three_numbers(panel_cats), "judges": {j: three_numbers(judge_cats[j]) for j in JUDGES},
                    "severity": {"panel": severity_split(panel_sev),
                                 "judges": {j: severity_split(judge_sev[j]) for j in JUDGES}},
                    "unverifiable_claims": {"judges": {j: {k: judge_claims[j][k] for k in
                                                          ("n_claims", "n_unverifiable", "unverifiable_share")}
                                                      for j in JUDGES},
                                            "pooled": {k: pool_claim_stats(judge_claims)[k] for k in
                                                       ("n_claims", "n_unverifiable", "unverifiable_share")}},
                    # B, C, D only: condition A has no context to be faithful to or to cite
                    "faithfulness_citation": ({"judges": {j: {k: judge_claims[j][k] for k in
                                                              ("faithfulness", "citation_support")} for j in JUDGES},
                                               "pooled": {k: pool_claim_stats(judge_claims)[k] for k in
                                                          ("faithfulness", "citation_support")}}
                                              if has_context else None),
                    "cosine": {"mean": round(st.mean(cos), 4) if cos else None,
                               "median": round(st.median(cos), 4) if cos else None},
                    "nf2_citation_validity": _ratio(sum(1 for c in cited if c), len(cited)) if cited else None,
                    "latency_vs_nf3": nf3(answers, secs, run["condition"])},
        "reliability": {"fleiss_kappa_3cat": fleiss_kappa([r for r in rows if None not in r]),
                        "n_complete": sum(1 for r in rows if None not in r),
                        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(rows),
                        "all_three_judges_same": unanimous},
        "per_question_panel": {str(q): c for q, c in zip(qids, panel_cats)},
    }


def latency_section(answers: list[dict], sec_per_answer, condition: str) -> dict:
    """Per-answer latencies (record schema r3): end-to-end (retrieval + generation) and generation only."""
    return {"end_to_end_sec": _pcts([a.get("end_to_end_latency_sec") for a in answers]),
            "generation_sec": _pcts([a.get("generation_latency_sec") for a in answers]),
            "retrieval_sec": _pcts([a.get("retrieval_latency_sec") for a in answers]),
            "run_duration_per_answer_sec": sec_per_answer,
            "schema_r3_answers": sum(1 for a in answers if a.get("end_to_end_latency_sec") is not None)}


def nf3(answers: list[dict], sec_per_answer, condition: str) -> dict:
    """NF3 = p95 of per-answer end-to-end latency, Condition C, target <= 15 s. Records without the r3 timing
    (archived runs) fall back to run duration / answers, labelled as a fallback (no p95 is possible from it)."""
    e2e = [a.get("end_to_end_latency_sec") for a in answers]
    applies = condition == NF3_CONDITION
    if answers and all(x is not None for x in e2e):
        p95 = _pcts(e2e)["p95"]
        return {"method": "p95 of end_to_end_latency_sec (per answer, schema r3)", "fallback": False,
                "value_sec": p95, "target_sec": NF3_TARGET_SEC, "nf3_applies": applies,
                "meets": (p95 <= NF3_TARGET_SEC) if applies else None}
    return {"method": "FALLBACK: run duration / answers (records without per-answer end-to-end timing)",
            "fallback": True, "value_sec": sec_per_answer, "target_sec": NF3_TARGET_SEC, "nf3_applies": applies,
            "meets": None}


def paired_counts(cats_c: dict, cats_x: dict, abstain_as_failed: bool = False) -> dict:
    """Per-question comparison of C (first) with another condition on the binary hallucination label.
    better = C not hallucinated and the other hallucinated; worse = the reverse; equal = same label.
    Primary (abstain_as_failed=False): only pairs where both answers are usable and neither is ABSTAIN count;
    the rest are reported as n_excluded, not imputed. With abstain_as_failed=True an ABSTAIN counts as failed
    (= hallucinated), so only unusable pairs are left out. `cells` is the 2x2 table: only C hallucinated,
    only the other, both, neither."""
    def label(c):
        if c is None:
            return None
        if c == CATEGORY_ABSTAIN:
            return True if abstain_as_failed else None
        return c == CATEGORY_HALLUCINATED

    better = worse = equal = excluded = 0
    cells = {"only_c": 0, "only_other": 0, "both": 0, "neither": 0}
    for q in sorted(set(cats_c) & set(cats_x)):
        hc, hx = label(cats_c[q]), label(cats_x[q])
        if hc is None or hx is None:
            excluded += 1
            continue
        if hc and hx:
            cells["both"] += 1
            equal += 1
        elif not hc and not hx:
            cells["neither"] += 1
            equal += 1
        elif hc:
            cells["only_c"] += 1
            worse += 1
        else:
            cells["only_other"] += 1
            better += 1
    return {"c_better": better, "equal": equal, "c_worse": worse, "n_pairs": better + worse + equal,
            "n_excluded": excluded, "cells": cells}


def paired_rate_difference(cats_a: dict, cats_b: dict) -> dict:
    """Hallucination-rate difference (a - b, ABSTAIN excluded) on the questions BOTH runs answered with a usable,
    non-ABSTAIN panel label. Same pairs as the primary paired_counts. rate_* = hallucinated / n_pairs on those pairs.
    difference_panel (each run's own non-ABSTAIN answers) is a separate figure and is not changed."""
    ha = hb = n = 0
    for q in sorted(set(cats_a) & set(cats_b)):
        ca, cb = cats_a[q], cats_b[q]
        if ca not in (CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED) or cb not in (CATEGORY_HALLUCINATED,
                                                                                       CATEGORY_NOT_HALLUCINATED):
            continue
        n += 1
        ha += ca == CATEGORY_HALLUCINATED
        hb += cb == CATEGORY_HALLUCINATED
    if n == 0:
        return {"difference": None, "rate_a": None, "rate_b": None, "n_pairs": 0}
    return {"difference": round(ha / n - hb / n, 4), "rate_a": round(ha / n, 4), "rate_b": round(hb / n, 4),
            "n_pairs": n}


def comparisons(runs: list[dict]) -> list[dict]:
    """The pre-registered comparisons C vs A, C vs B, C vs D (always listed; available=false if a run is missing).
    Legacy comparisons (needing legacy cells) appear only when both runs exist."""
    by_label = {r["label"]: r for r in runs}
    out = []
    for a, b, name in COMPARISONS + tuple(c for c in LEGACY_COMPARISONS if c[0] in {r["label"] for r in runs}
                                           and c[1] in {r["label"] for r in runs}):
        if a not in by_label or b not in by_label:
            out.append({"comparison": f"{a} vs {b}", "name": name, "available": False})
            continue
        ra, rb = by_label[a]["results"], by_label[b]["results"]

        def diff(x, y):
            return None if x is None or y is None else round(x - y, 4)

        d_panel = diff(ra["panel"]["rate_excl_abstain"], rb["panel"]["rate_excl_abstain"])
        d_panel_failed = diff(ra["panel"]["rate_abstain_as_failed"], rb["panel"]["rate_abstain_as_failed"])
        d_j = {j: diff(ra["judges"][j]["rate_excl_abstain"], rb["judges"][j]["rate_excl_abstain"]) for j in JUDGES}
        sens = d_j[SENSITIVE_JUDGE]
        consistent = d_panel is not None and sens is not None and d_panel != 0 and sens != 0 and \
            (d_panel > 0) == (sens > 0)
        pq_a, pq_b = by_label[a].get("per_question_panel") or {}, by_label[b].get("per_question_panel") or {}
        out.append({"comparison": f"{a} vs {b}", "name": name, "available": True, "difference_panel": d_panel,
                    "difference_panel_abstain_as_failed": d_panel_failed,
                    "difference_panel_paired": paired_rate_difference(pq_a, pq_b),
                    "difference_by_judge": d_j, "flag": "consistent" if consistent else "inconclusive",
                    "paired": {"primary_excl_abstain": paired_counts(pq_a, pq_b),
                               "abstain_as_failed": paired_counts(pq_a, pq_b, abstain_as_failed=True)}})
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _git_head() -> str | None:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return r.stdout.strip() or None


def configuration() -> dict:
    frozen_path = REPO_ROOT / "results" / "frozen_parameters.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    decision = (REPO_ROOT / "docs" / "DECISION_C_SCORING.md").read_text()
    amendments = [line.lstrip("# ").strip() for line in decision.splitlines() if line.startswith("## Amendment")]
    import judge_versions as jv
    return {"frozen_parameters": frozen.get("parameters", []), "table_hash": frozen.get("table_hash"),
            "amendments_in_decision_record": amendments,
            "amendment14": {"approved": jv.AMENDMENT14_APPROVED, "draft": "reports/amendment14_draft.md"},
            "git_head": _git_head()}


def data_and_sampling() -> dict:
    from llm import official_run_matrix as orm
    meta_path = REPO_ROOT / "results" / "leakage" / "exclusion_sets.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"test_sample": {"n": orm.N_SAMPLE, "seed": orm.SEED, "positions": "1-384",
                            "frozen_list": "llm/sample_ids/test_ids_v1.json"},
            "dev_set": {"n": 50, "positions": "385-434", "frozen_list": "llm/sample_ids/dev_ids_v1.json"},
            "leakage": {"mode": orm.LEAKAGE_MODE, "n_pool": meta.get("n_pool"), "modes_built": meta.get("modes"),
                        "built_at_utc": meta.get("built_at_utc"), "sha256": meta.get("sha256"),
                        "rules": "own question + all its answers excluded; Duplicate, Linked and near-duplicate "
                                 "questions are NOT excluded (Amendment 15, supersedes Amendment 8)"}}


def alpha_selection() -> dict:
    s1_path = archive_paths.resolve(ARCHIVE_EVIDENCE["stage1_of_record"])
    s1 = json.loads(s1_path.read_text()) if s1_path.exists() else {}
    stage1 = {"from_archive": True, "source": ARCHIVE_EVIDENCE["stage1_of_record"],
              "table": [{"alpha": float(a), "pct_questions_with_relevant": m.get("pct_questions_with_relevant"),
                         "mean_relevance_score": m.get("mean_relevance_score")} for a, m in sorted((s1.get("new") or {}).items(), key=lambda x: float(x[0]))],
              "top_two": (s1.get("selection_new") or {}).get("top_two"),
              "reasoning": (s1.get("selection_new") or {}).get("reasoning")}
    stage2_dir = REPO_ROOT / "llm" / "c_graphrag" / "results" / "stage2"
    reports = sorted(stage2_dir.glob("stage2_report_*.json")) if stage2_dir.is_dir() else []
    stage2 = None
    if reports:
        d = json.loads(reports[-1].read_text())
        stage2 = {"source": str(reports[-1].relative_to(REPO_ROOT)), "panel": d.get("panel"), "pooled": d.get("pooled"),
                  "per_judge_pooled": d.get("per_judge_pooled"), "winner": (d.get("selection") or {}).get("winner"),
                  "alpha_star": (d.get("draft_official_params") or {}).get("alpha_primary"),
                  "alpha_abl": (d.get("draft_official_params") or {}).get("alpha_ablation"),
                  "tie_rule_used": d.get("tie_rule_used")}
    official = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    return {"stage1": stage1, "stage2": stage2, "official_params_status": official.get("status"),
            "alpha_star": official.get("alpha_primary"), "alpha_abl": official.get("alpha_ablation")}


def limitations() -> dict:
    draft = REPO_ROOT / "reports" / "amendment14_draft.md"
    evidence = {}
    for key, rel in ARCHIVE_EVIDENCE.items():
        if key == "stage1_of_record":
            continue
        p = archive_paths.resolve(rel)
        evidence[key] = {"path": str(p.relative_to(REPO_ROOT)) if p.exists() else rel, "from_archive": True,
                         "text": p.read_text() if p.exists() else None}
    return {"amendment14_text": draft.read_text() if draft.exists() else None, "amendment14_status": "draft",
            "archived_evidence": evidence,
            "key_numbers": {"kappa_3cat_panel_v1": 0.529, "kappa_binary_panel_v1": 0.180, "recall_panel_v1": "2/15",
                            "recall_per_condition_panel_v1": {"A [dev]": "1/8", "B-plain [dev]": "1/6",
                                                              "B-grounded [dev]": "0/1"},
                            "panel_v2_calibration": {"kappa_3cat": 0.566, "kappa_binary": 0.329, "recall": "5/15"}},
            "wording": "panel-detected hallucination rate"}


def build(store=None, include_dev_stage2: bool = False) -> dict:
    store = store or RegistryStore(include_dev_stage2)
    official = json.loads((REPO_ROOT / "llm" / "c_graphrag" / "official_params.json").read_text())
    gen = store.generation_runs()
    runs = [run_section(store, r, official) for r in gen]
    for sec, r in zip(runs, gen):  # secondary metrics (context / answer relevance), if judged; never mocked
        sec["secondary"] = None if store.mock else secondary_metrics.report_section(r["files"]["output"], r["condition"])
    runs.sort(key=lambda r: r["label"])
    return {
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mock": store.mock,
                 "panel": PANEL_ID, "judge_version": PROMPT_VERSION, "n_runs": len(runs),
                 "wording": "panel-detected hallucination rates"},
        "configuration": configuration(), "data": data_and_sampling(), "alpha_selection": alpha_selection(),
        "runs": runs, "comparisons": comparisons(runs),
        "budget": {**store.budget(), "judging_cost_usd": store.judge_cost()},
        "limitations": limitations(),
    }


def item_detail(store, run_id: str, question_id: int) -> dict | None:
    """Item drill-down: question, reference, context, candidate, each judge's claims and verdicts, panel label."""
    run = next((r for r in store.generation_runs() if r["run_id"] == run_id), None)
    if run is None:
        return None
    ans = next((a for a in store.answers(run) if int(a["question_id"]) == int(question_id)), None)
    if ans is None:
        return None
    recs = {r["judge_id"]: r for r in store.judge_records(run) if int(r["question_id"]) == int(question_id)}
    labels = {j: (recs[j]["label"] if j in recs and category_of(recs[j]) is not None else None) for j in JUDGES}
    return {"run_id": run_id, "question_id": int(question_id), "title": ans.get("title"),
            "reference_html": ans.get("ground_truth_answer"), "context": ans.get("retrieved_context") or [],
            "candidate": ans.get("llm_answer"),
            "judges": {j: {"label": (recs.get(j) or {}).get("label"), "claims": (recs.get(j) or {}).get("claims"),
                           "reasoning": (recs.get(j) or {}).get("reasoning")} for j in JUDGES},
            "panel": aggregate_labels(labels, PANEL_ID), "mock": store.mock}


def item_list(store, run_id: str) -> list[dict] | None:
    """Per answer of one run: question id, title, each judge's label and the panel label (drill-down index)."""
    run = next((r for r in store.generation_runs() if r["run_id"] == run_id), None)
    if run is None:
        return None
    by_q = {}
    for r in store.judge_records(run):
        by_q.setdefault(int(r["question_id"]), {})[r["judge_id"]] = r
    out = []
    for a in store.answers(run):
        q = int(a["question_id"])
        recs = by_q.get(q, {})
        labels = {j: (recs[j]["label"] if j in recs and category_of(recs[j]) is not None else None) for j in JUDGES}
        out.append({"question_id": q, "title": a.get("title"), "judge_labels": labels,
                    "panel_label": aggregate_labels(labels, PANEL_ID)["panel_label"]})
    return out


def _pct(x):
    return "--" if x is None else f"{100 * x:.1f}%"


def markdown(rep: dict) -> str:
    m = rep["meta"]
    tag = " (MOCK DATA -- not results)" if m["mock"] else ""
    lines = [f"# Official evaluation report{tag}", "",
             f"Generated {m['generated_at']}; {m['panel']}, {m['judge_version']}; rates are {m['wording']}. "
             "Four official runs: A, B-grounded, C-primary-grounded, D-grounded.", "",
             "## Hallucination rate (panel primary; DeepSeek-only sensitive, others supporting)", "",
             "(a) excl. ABSTAIN = hallucinated / (hallucinated + not hallucinated), primary. "
             "(b) ABSTAIN as failed = (hallucinated + ABSTAIN) / all usable answers.", "",
             "| Run | Cond. | n | (a) Panel | (b) Panel | ABSTAIN | full / partial | unverifiable claims | "
             "DeepSeek (a) | Llama (a) | Qwen3 (a) | Cosine mean / median | NF2 | end-to-end p95 s (NF3: C <= 15 s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        res, p, sev = r["results"], r["results"]["panel"], r["results"]["severity"]["panel"]
        lines.append(f"| {r['label']} | {r['condition']} | {r['n']} | "
                     f"{p['rate_excl_abstain']} | {p['rate_abstain_as_failed']} | {p['abstain_rate']} | "
                     f"{sev['full']} / {sev['partial']} | {_pct(res['unverifiable_claims']['pooled']['unverifiable_share'])} | "
                     f"{res['judges']['deepseek']['rate_excl_abstain']} | {res['judges']['llama']['rate_excl_abstain']} | "
                     f"{res['judges']['qwen3-235b']['rate_excl_abstain']} | {res['cosine']['mean']} / {res['cosine']['median']} | "
                     f"{res['nf2_citation_validity']} | {res['latency_vs_nf3']['value_sec']}"
                     f"{' (fallback)' if res['latency_vs_nf3']['fallback'] else ''} |")
    if not rep["runs"]:
        lines.append("| -- | no official run registered yet | | | | | | | | | | | | |")
    lines += ["", "## Faithfulness and citation support (B, C, D; from the per-claim labels of the panel records)", "",
              "Faithfulness = claims with supported_by_context = YES / claims rated YES or NO. Citation support = cited "
              "claims with cited_item_supports = YES / cited claims rated YES or NO. Pooled over the three judges' claim "
              "lists (micro); null values are left out of the denominator.", "",
              "| Run | Faithfulness (pooled) | claims rated | Citation support (pooled) | cited claims | "
              "Faithfulness llama / deepseek / qwen3 |", "|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        fc = r["results"]["faithfulness_citation"]
        if fc is None:
            continue
        pj, jj = fc["pooled"], fc["judges"]
        lines.append(f"| {r['label']} | {pj['faithfulness']['micro']} | {pj['faithfulness']['n_claims_rated']} | "
                     f"{pj['citation_support']['micro']} | {pj['citation_support']['n_claims_cited']} | "
                     f"{jj['llama']['faithfulness']['micro']} / {jj['deepseek']['faithfulness']['micro']} / "
                     f"{jj['qwen3-235b']['faithfulness']['micro']} |")
    lines += ["", "## Secondary metrics (one judge: Llama-3.3-70B-Instruct, temperature 0; context relevance B/C/D, "
              "answer relevance A-D)", "",
              "| Run | Context: % questions with >=1 RELEVANT | mean item score | items judged | "
              "Answer relevance mean | answers judged |", "|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        sec = r.get("secondary") or {}
        cx, ar = sec.get("context_relevance") or {}, sec.get("answer_relevance") or {}
        lines.append(f"| {r['label']} | {cx.get('pct_questions_with_relevant', 'not judged')} | "
                     f"{cx.get('mean_relevance_score', '--')} | {cx.get('n_items_judged', '--')} | "
                     f"{ar.get('mean_score', 'not judged')} | {ar.get('n_judged', '--')} |")
    lines += ["", "## Pre-registered comparisons (C vs A, C vs B, C vs D)", "",
              "Rule (e): consistent only if the panel and DeepSeek-only differences (excl. ABSTAIN) share a sign. "
              "Paired counts are per question on the binary hallucination label of the panel: "
              "better = C not hallucinated and the other hallucinated.", "",
              "| Comparison | Panel diff (a) | Panel diff (b) | DeepSeek diff | Flag | C better / equal / worse (a) | "
              "pairs left out | C better / equal / worse (b, ABSTAIN = failed) |", "|---|---|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        if not c["available"]:
            lines.append(f"| {c['name']} ({c['comparison']}) | -- | -- | -- | not available | -- | -- | -- |")
        else:
            d, pa, pb = c["difference_by_judge"], c["paired"]["primary_excl_abstain"], c["paired"]["abstain_as_failed"]
            lines.append(f"| {c['name']} ({c['comparison']}) | {c['difference_panel']} | "
                         f"{c['difference_panel_abstain_as_failed']} | {d['deepseek']} | **{c['flag']}** | "
                         f"{pa['c_better']} / {pa['equal']} / {pa['c_worse']} | {pa['n_excluded']} | "
                         f"{pb['c_better']} / {pb['equal']} / {pb['c_worse']} |")
    lines += ["", "## Judge reliability", "",
              "| Run | Fleiss' kappa (3 cat., complete) | n complete | Krippendorff's alpha | all 3 judges same label | "
              "all 3 same (3 categories) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['label']} | {r['reliability']['fleiss_kappa_3cat']} | {r['reliability']['n_complete']} | "
              f"{r['reliability']['krippendorff_alpha_nominal']} | "
              f"{_pct(r['reliability']['all_three_judges_same']['same_label'])} | "
              f"{_pct(r['reliability']['all_three_judges_same']['same_3cat_label'])} |" for r in rep["runs"]]
    a = rep["alpha_selection"]
    lines += ["", "## Alpha selection", "", f"- stage 1 (from the archive): top_two {a['stage1']['top_two']}",
              f"- stage 2: {('alpha* = ' + str(a['stage2']['alpha_star']) + ' (the second candidate, alpha_abl = ' + str(a['stage2']['alpha_abl']) + ', is used by no official run)') if a['stage2'] else 'not run yet'}",
              f"- official_params.json: {a['official_params_status']}", "",
              f"## Budget", "", f"- DeepInfra budget {rep['budget']['budget']}, spent ${rep['budget']['spent_usd']}, "
              f"remaining {rep['budget']['remaining_usd']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="synthetic data (walkthrough); outputs are labelled MOCK")
    ap.add_argument("--include-dev-stage2", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rep = build(MockStore() if args.mock else None, args.include_dev_stage2)
    out = Path(args.out) if args.out else OUT.with_name(OUT.name + ("_MOCK" if args.mock else ""))
    out.with_name(out.name + ".json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    out.with_name(out.name + ".md").write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
