"""
official_report.py
==================
End-to-end report of the official evaluation (Amendment 14 draft, plan
step 8), from the run registry and STORED outputs only -- no API call:

  per official run (and condition):
    - the three-number report (hallucination rate excl. ABSTAIN, ABSTAIN rate,
      hallucination rate over all answers) for the panel AND each judge
    - inter-judge Fleiss' kappa and Krippendorff's alpha (nominal, 3 categories)
    - cosine similarity (mean, median), citation validity (NF2), latency
      (retrieval p50/p95 per answer; run seconds per answer vs the NF3 target)
    - retrieval / generation / judging statistics
  the direction-agreement table of the pre-registered comparisons (rule (e)),
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
                          category_of)
from judge_panel_aggregate import UNRESOLVED, aggregate_labels, panel_category  # noqa: E402
from llm import archive_paths  # noqa: E402

PANEL_ID = "panel-v2"
PROMPT_VERSION = "judge-v2.3"
JUDGES = ("llama", "deepseek", "qwen3-235b")
SENSITIVE_JUDGE = "deepseek"
CATS = (CATEGORY_ABSTAIN, CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED)
NF3_TARGET_SEC = 15.0   # NF3: p95 of per-answer end-to-end latency (record schema r3), Condition C
NF3_CONDITION = "C"
COMPARISONS = (("R5", "R1"), ("R5", "R3"), ("R5", "R9"), ("R5", "R7"),  # official stage 1
               ("R4", "R2"), ("R4", "R8"), ("R4", "R6"))              # official stage 2, when available
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
            "rate_excl_abstain": _ratio(h, h + n_ok), "abstain_rate": _ratio(a, usable),
            "rate_all_answers": _ratio(h, usable)}


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
    """Synthetic official stage 1 (R1, R3, R5, R7, R9; 24 answers each) for the dashboard walkthrough. Nothing here
    is a result; every output built from it carries mock = true."""
    mock = True
    CELLS = (("R1", "A", None, None), ("R3", "B", True, None), ("R5", "C", True, 0.25), ("R7", "C", True, 0.0),
             ("R9", "D", True, None))
    RATES = {"R1": 0.35, "R3": 0.22, "R5": 0.15, "R7": 0.18, "R9": 0.2}

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
                                          "evidence": '[R] "mock reference"'}] if label.startswith("HALU") else
                                        [{"claim": "mock claim", "verdict": "SUPPORTED", "severity": None,
                                          "evidence": '[R] "mock reference"'}]),
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
# Run labels (R1..R9) from the official run matrix
# ---------------------------------------------------------------------------

def matrix_label(run: dict, official: dict | None) -> str:
    if run.get("_label"):
        return run["_label"]
    if run["status"] == "dev-stage2":
        return f"S2 a={run['alpha']} grounding {'on' if run['grounding'] else 'off'}"
    from llm.official_run_matrix import build_matrix
    for row in build_matrix(official):
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
    panel_labels, rows = [], []
    for q in qids:
        recs = by_q.get(q, {})
        cats = {j: (category_of(recs[j]) if j in recs else None) for j in JUDGES}
        for j in JUDGES:
            judge_cats[j].append(cats[j])
        labels = {j: (recs[j]["label"] if j in recs and category_of(recs[j]) is not None else None) for j in JUDGES}
        panel_labels.append(aggregate_labels(labels, PANEL_ID))
        rows.append([cats[j] for j in JUDGES])
    panel_cats = [panel_category(p["panel_label"]) for p in panel_labels]
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
                    "cosine": {"mean": round(st.mean(cos), 4) if cos else None,
                               "median": round(st.median(cos), 4) if cos else None},
                    "nf2_citation_validity": _ratio(sum(1 for c in cited if c), len(cited)) if cited else None,
                    "latency_vs_nf3": nf3(answers, secs, run["condition"])},
        "reliability": {"fleiss_kappa_3cat": fleiss_kappa([r for r in rows if None not in r]),
                        "n_complete": sum(1 for r in rows if None not in r),
                        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(rows)},
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


def comparisons(runs: list[dict]) -> list[dict]:
    by_label = {r["label"]: r for r in runs}
    out = []
    for a, b in COMPARISONS:
        if a not in by_label or b not in by_label:
            out.append({"comparison": f"{a} vs {b}", "available": False})
            continue
        ra, rb = by_label[a]["results"], by_label[b]["results"]

        def diff(x, y):
            return None if x is None or y is None else round(x - y, 4)

        d_panel = diff(ra["panel"]["rate_excl_abstain"], rb["panel"]["rate_excl_abstain"])
        d_j = {j: diff(ra["judges"][j]["rate_excl_abstain"], rb["judges"][j]["rate_excl_abstain"]) for j in JUDGES}
        sens = d_j[SENSITIVE_JUDGE]
        consistent = d_panel is not None and sens is not None and d_panel != 0 and sens != 0 and \
            (d_panel > 0) == (sens > 0)
        out.append({"comparison": f"{a} vs {b}", "available": True, "difference_panel": d_panel,
                    "difference_by_judge": d_j, "flag": "consistent" if consistent else "inconclusive"})
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
                        "rules": "own thread + Duplicate-linked + near-duplicate (cosine >= 0.95) excluded; "
                                 "Linked kept (Amendment 8)"}}


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
    runs = [run_section(store, r, official) for r in store.generation_runs()]
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


def markdown(rep: dict) -> str:
    m = rep["meta"]
    tag = " (MOCK DATA -- not results)" if m["mock"] else ""
    lines = [f"# Official evaluation report{tag}", "",
             f"Generated {m['generated_at']}; {m['panel']}, {m['judge_version']}; rates are {m['wording']}.", "",
             "## Results (three numbers; panel primary, DeepSeek-only sensitive, others supporting)", "",
             "| Run | Cond. | Grounding | alpha | n | Panel excl. ABSTAIN | Panel ABSTAIN | Panel all | DeepSeek excl. | "
             "Llama excl. | Qwen3 excl. | Cosine mean / median | NF2 | end-to-end p95 s (NF3: C <= 15 s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["runs"]:
        res, p = r["results"], r["results"]["panel"]
        lines.append(f"| {r['label']} | {r['condition']} | {r['grounding']} | {r['alpha']} | {r['n']} | "
                     f"{p['rate_excl_abstain']} | {p['abstain_rate']} | {p['rate_all_answers']} | "
                     f"{res['judges']['deepseek']['rate_excl_abstain']} | {res['judges']['llama']['rate_excl_abstain']} | "
                     f"{res['judges']['qwen3-235b']['rate_excl_abstain']} | {res['cosine']['mean']} / {res['cosine']['median']} | "
                     f"{res['nf2_citation_validity']} | {res['latency_vs_nf3']['value_sec']}"
                     f"{' (fallback)' if res['latency_vs_nf3']['fallback'] else ''} |")
    if not rep["runs"]:
        lines.append("| -- | no official run registered yet | | | | | | | | | | | | |")
    lines += ["", "## Comparisons (rule (e): consistent only if panel and DeepSeek-only differences share a sign)", "",
              "| Comparison | Panel diff | DeepSeek diff | Llama diff | Qwen3 diff | Flag |", "|---|---|---|---|---|---|"]
    for c in rep["comparisons"]:
        if not c["available"]:
            lines.append(f"| {c['comparison']} | -- | -- | -- | -- | not available |")
        else:
            d = c["difference_by_judge"]
            lines.append(f"| {c['comparison']} | {c['difference_panel']} | {d['deepseek']} | {d['llama']} | "
                         f"{d['qwen3-235b']} | **{c['flag']}** |")
    lines += ["", "## Judge reliability", "", "| Run | Fleiss' kappa (3 cat., complete) | n complete | Krippendorff's alpha |",
              "|---|---|---|---|"]
    lines += [f"| {r['label']} | {r['reliability']['fleiss_kappa_3cat']} | {r['reliability']['n_complete']} | "
              f"{r['reliability']['krippendorff_alpha_nominal']} |" for r in rep["runs"]]
    a = rep["alpha_selection"]
    lines += ["", "## Alpha selection", "", f"- stage 1 (from the archive): top_two {a['stage1']['top_two']}",
              f"- stage 2: {('alpha* = ' + str(a['stage2']['alpha_star']) + ', alpha_abl = ' + str(a['stage2']['alpha_abl'])) if a['stage2'] else 'not run yet'}",
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
