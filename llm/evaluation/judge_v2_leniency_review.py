"""
judge_v2_leniency_review.py
=====================================
Section 3 of the 2026-10-07 judge-v2 follow-up: a PILOT DIAGNOSTIC (n=90
per judge, NOT a result) comparing judge-v1 and judge-v2 on the same
9-run pilot batch, computed on the reparsed judge-v2 files ONLY (see
reparse_judge_v2.py) -- analysis only, no prompt changes, no API calls.

Reads:
    judge-v1: results/jv1_dashboard_{primary,secondary}_judge-v1.jsonl
    judge-v2: results/jv2_dashboard_{primary,secondary}_judge-v2_reparsed.jsonl

Writes: docs/JUDGE_V2_LENIENCY_REVIEW.md

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 judge_v2_leniency_review.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_agreement import (  # noqa: E402
    ALL_LABELS,
    build_confusion_matrix,
    compute_kappas,
    index_by_key,
    load_judge_records,
    pair_two_judges,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = REPO_ROOT / "docs" / "JUDGE_V2_LENIENCY_REVIEW.md"

V1_FILES = [str(RESULTS_DIR / "jv1_dashboard_primary_judge-v1.jsonl"),
            str(RESULTS_DIR / "jv1_dashboard_secondary_judge-v1.jsonl")]
V2_REPARSED_FILES = [str(RESULTS_DIR / "jv2_dashboard_primary_judge-v2_reparsed.jsonl"),
                      str(RESULTS_DIR / "jv2_dashboard_secondary_judge-v2_reparsed.jsonl")]
V2_ORIGINAL_FILES = [str(RESULTS_DIR / "jv2_dashboard_primary_judge-v2.jsonl"),
                      str(RESULTS_DIR / "jv2_dashboard_secondary_judge-v2.jsonl")]

LABEL_ORDER = ("FAKTUAL", "HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH", "ABSTAIN", "PARSE_ERROR")


def _label_or_parse_error(r: dict) -> str:
    return r.get("label") if not r.get("parse_error") else "PARSE_ERROR"


# ---------------------------------------------------------------------
# Label distribution before/after reparse (primary only -- the 25
# recovered items are all primary; secondary had 0 parse errors).
# ---------------------------------------------------------------------

def label_distribution_before_after(judge_id: str) -> dict:
    before_path = RESULTS_DIR / f"jv2_dashboard_{judge_id}_judge-v2.jsonl"
    after_path = RESULTS_DIR / f"jv2_dashboard_{judge_id}_judge-v2_reparsed.jsonl"
    before_by_run_label: dict[str, Counter] = defaultdict(Counter)
    after_by_run_label: dict[str, Counter] = defaultdict(Counter)
    recovered: list[dict] = []  # items that were PARSE_ERROR before, a real label after

    before_by_key = {}
    with open(before_path) as f:
        for line in f:
            r = json.loads(line)
            before_by_key[(r["run_id"], r["question_id"])] = r
            before_by_run_label[r.get("run_label")][_label_or_parse_error(r)] += 1

    with open(after_path) as f:
        for line in f:
            r = json.loads(line)
            after_by_run_label[r.get("run_label")][_label_or_parse_error(r)] += 1
            before_r = before_by_key.get((r["run_id"], r["question_id"]))
            if before_r is not None and before_r.get("parse_error") and not r.get("parse_error"):
                recovered.append({
                    "question_id": r["question_id"], "run_label": r.get("run_label"),
                    "label": r.get("label"),
                })

    return {
        "before_by_run_label": {k: dict(v) for k, v in before_by_run_label.items()},
        "after_by_run_label": {k: dict(v) for k, v in after_by_run_label.items()},
        "recovered_items": recovered,
    }


# ---------------------------------------------------------------------
# v1 -> v2 transition matrix (per judge_id), joined by (run_id, question_id)
# ---------------------------------------------------------------------

def build_transition_data(v1_records: list[dict], v2_records: list[dict], judge_id: str) -> dict:
    v1_by_key = {(r["run_id"], r["question_id"]): r for r in v1_records if r.get("judge_id") == judge_id}
    v2_by_key = {(r["run_id"], r["question_id"]): r for r in v2_records if r.get("judge_id") == judge_id}
    common_keys = set(v1_by_key) & set(v2_by_key)

    matrix: dict[str, Counter] = defaultdict(Counter)
    drilldown = []
    for key in common_keys:
        v1_r, v2_r = v1_by_key[key], v2_by_key[key]
        v1_label = _label_or_parse_error(v1_r)
        v2_label = _label_or_parse_error(v2_r)
        matrix[v1_label][v2_label] += 1

        if v1_label in ("HALUSINASI_PENUH", "HALUSINASI_SEBAGIAN") and v2_label == "FAKTUAL":
            v1_errors = [c for c in v1_r.get("claims", [])
                         if str(c.get("verdict", "")).upper() in ("CONTRADICTED", "FABRICATED")]
            v2_claims = v2_r.get("claims", [])
            drilldown.append({
                "question_id": v1_r["question_id"], "run_label": v1_r.get("run_label"), "judge_id": judge_id,
                "v1_label": v1_label,
                "v1_error_claims": [{"claim": c.get("claim"), "verdict": c.get("verdict"),
                                      "severity": c.get("severity")} for c in v1_errors],
                "v2_claims": [{"claim": c.get("claim"), "verdict": c.get("verdict"),
                               "severity": c.get("severity"), "reference_conflict": c.get("reference_conflict")}
                              for c in v2_claims],
                "reference_conflict_used": any(c.get("reference_conflict") for c in v2_claims),
            })

    return {"matrix": {k: dict(v) for k, v in matrix.items()}, "drilldown": drilldown, "n_pairs": len(common_keys)}


# ---------------------------------------------------------------------
# reference_conflict stats (v2 reparsed only)
# ---------------------------------------------------------------------

def reference_conflict_stats(v2_records: list[dict], judge_id: str) -> dict:
    by_run_label: dict[str, dict] = {}
    records = [r for r in v2_records if r.get("judge_id") == judge_id and not r.get("parse_error")]
    totals = Counter(run_label=0)
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r.get("run_label")].append(r)

    for run_label, items in by_label.items():
        n_items = len(items)
        n_items_with_conflict = 0
        n_claims_with_conflict = 0
        for r in items:
            claims = r.get("claims", [])
            has_conflict = any(c.get("reference_conflict") for c in claims)
            if has_conflict:
                n_items_with_conflict += 1
            n_claims_with_conflict += sum(1 for c in claims if c.get("reference_conflict"))
        by_run_label[run_label] = {
            "n_items": n_items, "n_items_with_conflict": n_items_with_conflict,
            "pct_items_with_conflict": round(100 * n_items_with_conflict / n_items, 1) if n_items else None,
            "n_claims_with_conflict": n_claims_with_conflict,
        }
    return by_run_label


# ---------------------------------------------------------------------
# kappa helpers -- pre vs post reparse, and v1's own baseline
# ---------------------------------------------------------------------

def kappa_report(records: list[dict]) -> dict:
    pairs = pair_two_judges(records, "primary", "secondary")
    kappas = compute_kappas(pairs)
    return {"n_pairs": len(pairs), **kappas}


def main():
    v1_records = load_judge_records(V1_FILES)
    v2_reparsed_records = load_judge_records(V2_REPARSED_FILES)

    # Pre-reparse v2 records are read WITHOUT load_judge_records()'s
    # parse_version guard (those files predate the parse_version field
    # entirely, and that's expected/fine for a "before" snapshot) --
    # read directly instead.
    v2_original_records = []
    for path in V2_ORIGINAL_FILES:
        with open(path) as f:
            v2_original_records.extend(json.loads(line) for line in f if line.strip())

    lines = [
        "# Judge-v2 leniency review (pilot diagnostic, n=90 per judge)",
        "",
        "**This is a pilot diagnostic, not a result.** Computed on the 9-run pilot batch "
        "(n=90 generation items per judge) after the parser fix (2026-10-07) and the offline "
        "reparse (see `reparse_judge_v2.py`) -- the `*_reparsed.jsonl` files are the ONLY "
        "judge-v2 source used throughout this document. No prompt changes were made for this "
        "analysis.",
        "",
        "## 1. Label distribution before vs after reparse (primary judge)",
        "",
        "Secondary judge had 0 parse errors before reparse, so its distribution is unchanged "
        "and omitted here.",
        "",
    ]

    dist = label_distribution_before_after("primary")
    lines.append("| run_label | " + " | ".join(f"before {lbl}" for lbl in LABEL_ORDER) +
                  " | " + " | ".join(f"after {lbl}" for lbl in LABEL_ORDER) + " |")
    lines.append("|" + "---|" * (1 + 2 * len(LABEL_ORDER)))
    run_labels = sorted(set(dist["before_by_run_label"]) | set(dist["after_by_run_label"]))
    for run_label in run_labels:
        before = dist["before_by_run_label"].get(run_label, {})
        after = dist["after_by_run_label"].get(run_label, {})
        row = [run_label] + [str(before.get(lbl, 0)) for lbl in LABEL_ORDER] + \
            [str(after.get(lbl, 0)) for lbl in LABEL_ORDER]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append(f"### Where the {len(dist['recovered_items'])} recovered primary items landed")
    lines.append("")
    recovered_by_label = Counter(item["label"] for item in dist["recovered_items"])
    lines.append("| landed as | count |")
    lines.append("|---|---|")
    for lbl, count in recovered_by_label.most_common():
        lines.append(f"| {lbl} | {count} |")
    lines.append("")
    lines.append("| question_id | run_label | landed as |")
    lines.append("|---|---|---|")
    for item in sorted(dist["recovered_items"], key=lambda x: (x["run_label"] or "", x["question_id"])):
        lines.append(f"| {item['question_id']} | {item['run_label']} | {item['label']} |")
    lines.append("")

    # -------------------------------------------------------------
    # 2. Agreement: primary vs secondary, pre- vs post-reparse
    # -------------------------------------------------------------
    lines.append("## 2. Primary vs secondary agreement (Cohen's kappa)")
    lines.append("")
    pre = kappa_report(v2_original_records)
    post = kappa_report(v2_reparsed_records)
    v1_kappa = kappa_report(v1_records)
    lines.append("| | n_pairs | n_used (ordinal) | n_excluded | weighted kappa | unweighted kappa |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| judge-v1 (baseline) | {v1_kappa['n_pairs']} | {v1_kappa['n_used']} | "
                 f"{v1_kappa['n_excluded']} | {v1_kappa['weighted']} | {v1_kappa['unweighted']} |")
    lines.append(f"| judge-v2, pre-reparse | {pre['n_pairs']} | {pre['n_used']} | "
                 f"{pre['n_excluded']} | {pre['weighted']} | {pre['unweighted']} |")
    lines.append(f"| judge-v2, post-reparse | {post['n_pairs']} | {post['n_used']} | "
                 f"{post['n_excluded']} | {post['weighted']} | {post['unweighted']} |")
    lines.append("")
    lines.append(
        "Note: `n_excluded` counts pairs where either side's label is ABSTAIN or invalid "
        "(parse error) -- pre-reparse, the 25 recovered primary items were excluded here; "
        "post-reparse they contribute a real label and can enter the ordinal kappa "
        "computation (unless ABSTAIN)."
    )
    lines.append("")

    # -------------------------------------------------------------
    # 3. v1 -> v2 transition matrix, per judge
    # -------------------------------------------------------------
    lines.append("## 3. judge-v1 -> judge-v2 label transition matrix")
    lines.append("")
    for judge_id in ("primary", "secondary"):
        transition = build_transition_data(v1_records, v2_reparsed_records, judge_id)
        lines.append(f"### {judge_id} (n_pairs={transition['n_pairs']})")
        lines.append("")
        lines.append("| v1 vs v2 | " + " | ".join(LABEL_ORDER) + " |")
        lines.append("|---" * (1 + len(LABEL_ORDER)) + "|")
        for v1_label in LABEL_ORDER:
            row = transition["matrix"].get(v1_label, {})
            lines.append(f"| {v1_label} | " + " | ".join(str(row.get(v2_label, 0)) for v2_label in LABEL_ORDER) + " |")
        lines.append("")

    # -------------------------------------------------------------
    # 4. reference_conflict stats
    # -------------------------------------------------------------
    lines.append("## 4. `reference_conflict` usage (judge-v2, reparsed)")
    lines.append("")
    for judge_id in ("primary", "secondary"):
        stats = reference_conflict_stats(v2_reparsed_records, judge_id)
        lines.append(f"### {judge_id}")
        lines.append("")
        lines.append("| run_label | n items | n items with >=1 conflict | % items | n claims with conflict |")
        lines.append("|---|---|---|---|---|")
        for run_label in sorted(stats):
            s = stats[run_label]
            lines.append(f"| {run_label} | {s['n_items']} | {s['n_items_with_conflict']} | "
                          f"{s['pct_items_with_conflict']} | {s['n_claims_with_conflict']} |")
        lines.append("")

    # -------------------------------------------------------------
    # 5. HALUSINASI -> FAKTUAL drill-down
    # -------------------------------------------------------------
    lines.append("## 5. Items that were v1 HALUSINASI_* and are v2 FAKTUAL")
    lines.append("")
    for judge_id in ("primary", "secondary"):
        transition = build_transition_data(v1_records, v2_reparsed_records, judge_id)
        lines.append(f"### {judge_id} ({len(transition['drilldown'])} item(s))")
        lines.append("")
        for item in sorted(transition["drilldown"], key=lambda x: (x["run_label"] or "", x["question_id"])):
            lines.append(f"**question_id={item['question_id']} run_label={item['run_label']} "
                         f"v1_label={item['v1_label']} reference_conflict_used={item['reference_conflict_used']}**")
            lines.append("")
            lines.append("v1 CORE/MINOR error claim(s):")
            for c in item["v1_error_claims"]:
                lines.append(f"- [{c['verdict']}/{c['severity']}] {c['claim']}")
            lines.append("")
            lines.append("v2 claim(s):")
            for c in item["v2_claims"]:
                lines.append(f"- [{c['verdict']}/{c['severity']}, reference_conflict={c['reference_conflict']}] {c['claim']}")
            lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
