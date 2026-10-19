"""
judge_panel_agreement.py
------------------------
Judge-vs-human agreement for the three panel judges and the panel
(docs/DECISION_C_SCORING.md, Amendment 12 (e)).

ACCEPTANCE (Amendment 11, 12 (e), 13): the PANEL's unweighted 3-category
Cohen's kappa (ABSTAIN / hallucinated / not hallucinated) vs the frozen human
labels >= 0.6 AND the panel's binary kappa (ABSTAIN excluded) >= 0.4. The
3-category kappa is computed for every judge and the panel.

Supporting statistics (do not decide acceptance), on the binary
"hallucinated" outcome, reported twice:
  - ABSTAIN excluded: items that either side labels ABSTAIN are left out;
  - ABSTAIN counted:  ABSTAIN counts as "not hallucinated" on both sides.
Per rater (llama, deepseek, qwen, panel) vs the human, each with counts x/n:
  raw agreement, Cohen's kappa (unweighted; bootstrap 95% CI as in
  Amendment 11), hallucination recall (judge H | human H), unsafe pass rate
  (judge not H | human H), false alarm rate (judge H | human not H), and the
  confusion matrix (rows: human, columns: judge).
Also: ABSTAIN agreement (ABSTAIN yes/no) separately, n per condition, and
pairwise kappa between the judges.

Items a rater has no usable label for (failed call, parse error, panel
UNRESOLVED) are counted as missing for that rater, never silently dropped.

The panel composition is fixed in advance. Per-judge results are
informational and are never used to drop or swap a judge; the kappa >= 0.6
target applies to the panel.

CLI (only after the human annotation is complete; joins through the
verified key outside the repo):
    python3 judge_panel_agreement.py --purpose calibration \
        --judge-output results/panel_<name>.jsonl --panel-output results/panel_<name>_panel.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_binary import (BOOTSTRAP_RESAMPLES, CATEGORIES, CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED,  # noqa: E402
                          CATEGORY_NOT_HALLUCINATED, bootstrap_kappa_ci, category_of, cohen_kappa, confusion_matrix)
from judge_panel_aggregate import panel_category  # noqa: E402

JUDGE_RATERS = ("llama", "deepseek", "qwen")
RATERS = JUDGE_RATERS + ("panel",)
MODES = ("abstain_excluded", "abstain_counted")
BINARY = (CATEGORY_NOT_HALLUCINATED, CATEGORY_HALLUCINATED)
ABSTAIN_YN = ("ABSTAIN", "NOT_ABSTAIN")
ACCEPTANCE_KAPPA = 0.6
ACCEPTANCE_BINARY_KAPPA = 0.4  # Amendment 13: panel binary kappa, ABSTAIN excluded (judge-v2.3 and later)

# Post-freeze label edits (docs/ANNOTATION_POSTFREEZE_RECONSIDERATIONS.md): saved after the iteration-1 run, post-dating
# the freeze; never used for acceptance; every report adds a clearly labelled SENSITIVITY row computed with them.
SENSITIVITY_LABELS = {
    "calibration": {"CAL-010": CATEGORY_NOT_HALLUCINATED, "CAL-028": CATEGORY_HALLUCINATED,
                    "CAL-041": CATEGORY_HALLUCINATED},
}
SENSITIVITY_NOTE = ("SENSITIVITY: human labels with the post-freeze edits (docs/ANNOTATION_POSTFREEZE_RECONSIDERATIONS.md); "
                    "saved after the iteration-1 run, post-dating the freeze; not used for acceptance")


def _frac(x: int, n: int) -> dict:
    return {"x": x, "n": n, "rate": round(x / n, 4) if n else None}


def _kappa(a: list, b: list, cats, n_resamples: int) -> dict:
    k = cohen_kappa(a, b, cats)
    ci = bootstrap_kappa_ci(a, b, cats, n_resamples)["ci95"] if n_resamples else None
    return {"kappa": round(k, 4) if k is not None else None, "kappa_ci95": ci}


def binary_pairs(rater: dict, human: dict, mode: str) -> tuple[list, list, dict]:
    """Paired binary labels (rater, human) over the human's items. Inputs: {item_id: 3-category label or None}."""
    a, b = [], []
    counts = Counter()
    for i, h in sorted(human.items()):
        r = rater.get(i)
        if h is None or r is None:
            counts["n_missing"] += 1
            continue
        if mode == "abstain_excluded":
            if CATEGORY_ABSTAIN in (h, r):
                counts["n_abstain_excluded"] += 1
                continue
        else:
            h = CATEGORY_NOT_HALLUCINATED if h == CATEGORY_ABSTAIN else h
            r = CATEGORY_NOT_HALLUCINATED if r == CATEGORY_ABSTAIN else r
        a.append(r)
        b.append(h)
    return a, b, dict(counts)


def binary_metrics(rater_labels: list, human_labels: list, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    n = len(rater_labels)
    pairs = list(zip(rater_labels, human_labels))
    h_pos = [r for r, h in pairs if h == CATEGORY_HALLUCINATED]
    h_neg = [r for r, h in pairs if h == CATEGORY_NOT_HALLUCINATED]
    return {
        "n": n,
        "raw_agreement": _frac(sum(1 for r, h in pairs if r == h), n),
        **_kappa(rater_labels, human_labels, BINARY, n_resamples),
        "hallucination_recall": _frac(sum(1 for r in h_pos if r == CATEGORY_HALLUCINATED), len(h_pos)),
        "unsafe_pass_rate": _frac(sum(1 for r in h_pos if r == CATEGORY_NOT_HALLUCINATED), len(h_pos)),
        "false_alarm_rate": _frac(sum(1 for r in h_neg if r == CATEGORY_HALLUCINATED), len(h_neg)),
        # rows: human, columns: judge
        "confusion_human_rows": confusion_matrix(human_labels, rater_labels, BINARY),
    }


def three_category(rater: dict, human: dict, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Unweighted 3-category kappa (ABSTAIN / NOT_HALLUCINATED / HALLUCINATED) on items both label."""
    a, b, n_missing = [], [], 0
    for i, h in sorted(human.items()):
        r = rater.get(i)
        if h is None or r is None:
            n_missing += 1
            continue
        a.append(r)
        b.append(h)
    return {"n": len(a), "n_missing": n_missing,
            "raw_agreement": _frac(sum(1 for x, y in zip(a, b) if x == y), len(a)),
            **_kappa(a, b, CATEGORIES, n_resamples),
            "confusion_human_rows": confusion_matrix(b, a, CATEGORIES)}


def abstain_agreement(rater: dict, human: dict, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    a, b, n_missing = [], [], 0
    for i, h in sorted(human.items()):
        r = rater.get(i)
        if h is None or r is None:
            n_missing += 1
            continue
        a.append("ABSTAIN" if r == CATEGORY_ABSTAIN else "NOT_ABSTAIN")
        b.append("ABSTAIN" if h == CATEGORY_ABSTAIN else "NOT_ABSTAIN")
    return {"n": len(a), "n_missing": n_missing,
            "raw_agreement": _frac(sum(1 for x, y in zip(a, b) if x == y), len(a)),
            **_kappa(a, b, ABSTAIN_YN, n_resamples),
            "n_abstain_human": b.count("ABSTAIN"), "n_abstain_rater": a.count("ABSTAIN"),
            "confusion_human_rows": confusion_matrix(b, a, ABSTAIN_YN)}


def agreement_report(human: dict, raters: dict, conditions: dict, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """human: {item_id: category | None}; raters: {rater: {item_id: category | None}};
    conditions: {item_id: run_label}."""
    out = {"three_category": {}, "vs_human": {}, "abstain_agreement": {}, "pairwise_judges": {},
           "n_per_condition": {}}
    for name, labels in raters.items():
        out["three_category"][name] = three_category(labels, human, n_resamples)
        out["vs_human"][name] = {}
        for mode in MODES:
            a, b, counts = binary_pairs(labels, human, mode)
            out["vs_human"][name][mode] = {**binary_metrics(a, b, n_resamples), **counts}
        out["abstain_agreement"][name] = abstain_agreement(labels, human, n_resamples)
    judges = [r for r in JUDGE_RATERS if r in raters]
    for x, y in combinations(judges, 2):
        out["pairwise_judges"][f"{x}_vs_{y}"] = {}
        for mode in MODES:
            a, b, counts = binary_pairs(raters[x], raters[y], mode)
            out["pairwise_judges"][f"{x}_vs_{y}"][mode] = {
                "n": len(a), **counts, "raw_agreement": _frac(sum(1 for p, q in zip(a, b) if p == q), len(a)),
                **_kappa(a, b, BINARY, n_resamples)}
    for cond in sorted(set(conditions.values())):
        ids = [i for i in human if conditions.get(i) == cond]
        out["n_per_condition"][cond] = {
            "n_items": len(ids),
            "n_human_labelled": sum(1 for i in ids if human[i] is not None),
            "human_labels": dict(Counter(human[i] for i in ids if human[i] is not None)),
            **{f"n_{r}_usable": sum(1 for i in ids if raters[r].get(i) is not None) for r in raters},
        }
    k3 = out["three_category"].get("panel", {}).get("kappa")
    kb = out["vs_human"].get("panel", {}).get("abstain_excluded", {}).get("kappa")
    meets_3cat = k3 is not None and k3 >= ACCEPTANCE_KAPPA
    meets_bin = kb is not None and kb >= ACCEPTANCE_BINARY_KAPPA
    out["acceptance"] = {
        "rule": f"panel 3-category kappa (ABSTAIN / hallucinated / not hallucinated) >= {ACCEPTANCE_KAPPA} "
                f"(Amendment 11, 12 (e)) AND panel binary kappa, ABSTAIN excluded, >= {ACCEPTANCE_BINARY_KAPPA} "
                "(Amendment 13); the other binary metrics and per-judge results are supporting statistics",
        "panel_kappa_3cat": k3, "meets_3cat": meets_3cat,
        "panel_kappa_binary_abstain_excluded": kb, "meets_binary": meets_bin,
        "meets_target": meets_3cat and meets_bin,
    }
    return out


def sensitivity_report(human: dict, raters: dict, conditions: dict, overrides: dict,
                       n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The same metrics with `overrides` ({item_id: category}) applied to the human labels. Never decides."""
    edited = {i: overrides.get(i, c) for i, c in human.items()}
    r = agreement_report(edited, raters, conditions, n_resamples)
    return {"note": SENSITIVITY_NOTE, "overrides": dict(overrides),
            "changed_items": sorted(i for i in overrides if human.get(i) != overrides[i]),
            "three_category": r["three_category"], "vs_human": r["vs_human"]}


def _fmt(f: dict) -> str:
    return f"{f['x']}/{f['n']} ({f['rate']:.1%})" if f["rate"] is not None else f"{f['x']}/{f['n']} (-)"


def _ci(ci) -> str:
    return f"[{ci[0]}, {ci[1]}]" if ci else "-"


def report_markdown(purpose: str, rep: dict) -> str:
    lines = [f"# Panel agreement report -- {purpose} set", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}. Unweighted Cohen's kappa, 95% CI = percentile "
             "bootstrap (2,000 resamples, fixed seed). Confusion matrices: rows = human, columns = judge. "
             "The panel composition is fixed; per-judge rows are informational.", "",
             "## Acceptance: 3-category kappa (ABSTAIN / hallucinated / not hallucinated)", "",
             f"Rule (Amendment 11, Amendment 12 (e)): {rep['acceptance']['rule']}.", "",
             f"**Panel 3-category kappa = {rep['acceptance']['panel_kappa_3cat']} (>= {ACCEPTANCE_KAPPA}: "
             f"{rep['acceptance']['meets_3cat']}); panel binary kappa, ABSTAIN excluded = "
             f"{rep['acceptance']['panel_kappa_binary_abstain_excluded']} (>= {ACCEPTANCE_BINARY_KAPPA}: "
             f"{rep['acceptance']['meets_binary']}). Accepted: {rep['acceptance']['meets_target']}.**", "",
             "| rater | n | missing | raw agreement | 3-category kappa | 95% CI |", "|---|---|---|---|---|---|"]
    sens = rep.get("sensitivity")
    rows3 = list(rep["three_category"].items())
    if sens:
        rows3 += [(f"SENSITIVITY: {n}", r) for n, r in sens["three_category"].items()]
    for name, r in rows3:
        lines.append(f"| {name} | {r['n']} | {r['n_missing']} | {_fmt(r['raw_agreement'])} | {r['kappa']} | "
                     f"{_ci(r['kappa_ci95'])} |")
    lines.append("")
    if sens:
        lines += [f"SENSITIVITY rows: {sens['note']}. Edited items: {', '.join(sens['changed_items'])}.", ""]
    for name, r in rep["three_category"].items():
        m = r["confusion_human_rows"]
        lines += [f"{name}:", "", "| human \\ judge | " + " | ".join(CATEGORIES) + " |", "|---|---|---|---|"]
        for row in CATEGORIES:
            lines.append(f"| {row} | " + " | ".join(str(m[row][c]) for c in CATEGORIES) + " |")
        lines.append("")
    lines += ["# Supporting statistics (do not decide acceptance)", ""]
    for mode in MODES:
        title = "ABSTAIN excluded" if mode == "abstain_excluded" else "ABSTAIN counted as not hallucinated"
        lines += [f"## Judge vs human -- {title}", "",
                  "| rater | n | missing | raw agreement | kappa | 95% CI | recall | unsafe pass | false alarm |",
                  "|---|---|---|---|---|---|---|---|---|"]
        rows = list(rep["vs_human"].items())
        if sens:
            rows += [(f"SENSITIVITY: {n}", b) for n, b in sens["vs_human"].items()]
        for name, by_mode in rows:
            r = by_mode[mode]
            lines.append(f"| {name} | {r['n']} | {r.get('n_missing', 0)} | {_fmt(r['raw_agreement'])} | {r['kappa']} | "
                         f"{_ci(r['kappa_ci95'])} | {_fmt(r['hallucination_recall'])} | "
                         f"{_fmt(r['unsafe_pass_rate'])} | {_fmt(r['false_alarm_rate'])} |")
        lines.append("")
        for name, by_mode in rep["vs_human"].items():
            m = by_mode[mode]["confusion_human_rows"]
            lines += [f"{name}:", "", "| human \\ judge | NOT_H | H |", "|---|---|---|"]
            for row in BINARY:
                lines.append(f"| {row} | {m[row][CATEGORY_NOT_HALLUCINATED]} | {m[row][CATEGORY_HALLUCINATED]} |")
            lines.append("")
    lines += ["## ABSTAIN agreement (ABSTAIN yes/no)", "",
              "| rater | n | raw agreement | kappa | 95% CI | ABSTAIN human | ABSTAIN rater |", "|---|---|---|---|---|---|---|"]
    for name, r in rep["abstain_agreement"].items():
        lines.append(f"| {name} | {r['n']} | {_fmt(r['raw_agreement'])} | {r['kappa']} | {_ci(r['kappa_ci95'])} | "
                     f"{r['n_abstain_human']} | {r['n_abstain_rater']} |")
    lines += ["", "## Pairwise judge agreement (binary)", "",
              "| pair | mode | n | raw agreement | kappa | 95% CI |", "|---|---|---|---|---|---|"]
    for pair, by_mode in rep["pairwise_judges"].items():
        for mode, r in by_mode.items():
            lines.append(f"| {pair} | {mode} | {r['n']} | {_fmt(r['raw_agreement'])} | {r['kappa']} | "
                         f"{_ci(r['kappa_ci95'])} |")
    lines += ["", "## n per condition", "", "```json", json.dumps(rep["n_per_condition"], indent=2), "```", ""]
    return "\n".join(lines)


def raters_from_records(key_items: list[dict], judge_records: list[dict], panel_records: list[dict]) -> dict:
    """{rater: {item_id: category | None}} joined through the key by (run_id, question_id)."""
    from judge_panel_config import PANEL_ID, PANEL_JUDGES

    by_rq = {(k["run_id"], int(k["question_id"])): k["item_id"] for k in key_items}
    model_to_id = {j.model: j.judge_id for j in PANEL_JUDGES}
    raters = {r: {k["item_id"]: None for k in key_items} for r in RATERS}
    for rec in judge_records:
        item = by_rq.get((rec.get("run_id"), int(rec.get("question_id", -1))))
        jid = model_to_id.get(rec.get("judge_model"))
        if item is not None and jid and rec.get("panel_id") == PANEL_ID:
            raters[jid][item] = category_of(rec)
    for rec in panel_records:
        item = by_rq.get((rec.get("run_id"), int(rec.get("question_id", -1))))
        if item is not None and rec.get("panel_id") == PANEL_ID:
            raters["panel"][item] = panel_category(rec["panel_label"])
    return raters


def main(argv=None) -> int:
    from annotation_tool import AnnotationError, assert_annotation_frozen, key_dir_from, load_annotator, load_verified_key
    from judge_panel import read_jsonl

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--purpose", choices=["calibration", "confirmation"], required=True)
    p.add_argument("--judge-output", required=True, help="Per-judge panel JSONL (judge_panel.py --out)")
    p.add_argument("--panel-output", required=True, help="Panel JSONL (<out>_panel.jsonl)")
    p.add_argument("--key-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    try:  # the human labels must be exactly the frozen sheet (Amendment 12)
        frozen = assert_annotation_frozen(args.purpose, "annotator1")
    except AnnotationError as e:
        print(f"[ERROR] {e}")
        return 1
    print(f"[freeze] {frozen['file_name']} matches its freeze record (sha256 {frozen['sha256'][:12]}...)")
    key = load_verified_key(args.purpose, key_dir_from(args.key_dir))
    researcher, problems = load_annotator(args.purpose, "annotator1", key)
    if problems:
        print(f"[STOP] {len(problems)} problem(s) in the annotation sheet -- fix them first")
        for pr in problems[:20]:
            print(f"  - {pr}")
        return 1
    human = {k["item_id"]: (researcher.get(k["item_id"]) or {}).get("category") for k in key["items"]}
    conditions = {k["item_id"]: k["run_label"] for k in key["items"]}
    judge_records, panel_records = read_jsonl(Path(args.judge_output)), read_jsonl(Path(args.panel_output))
    versions = {r.get("prompt_version") for r in judge_records} | {r.get("prompt_version") for r in panel_records}
    if len(versions) != 1:
        print(f"[ERROR] the judge and panel files must hold exactly one prompt version, found {sorted(map(str, versions))}")
        return 1
    print(f"[judge version] {versions.pop()}")
    raters = raters_from_records(key["items"], judge_records, panel_records)
    rep = agreement_report(human, raters, conditions)
    if SENSITIVITY_LABELS.get(args.purpose):
        rep["sensitivity"] = sensitivity_report(human, raters, conditions, SENSITIVITY_LABELS[args.purpose])
    out = Path(args.out) if args.out else Path(args.judge_output).with_name(f"panel_agreement_{args.purpose}.md")
    out.write_text(report_markdown(args.purpose, rep))
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    for name, by_mode in rep["vs_human"].items():
        print(f"{name}: 3-category kappa={rep['three_category'][name]['kappa']} | binary kappa "
              f"excl={by_mode['abstain_excluded']['kappa']} counted={by_mode['abstain_counted']['kappa']}")
    if "sensitivity" in rep:
        s3 = rep["sensitivity"]["three_category"]["panel"]["kappa"]
        sb = rep["sensitivity"]["vs_human"]["panel"]["abstain_excluded"]["kappa"]
        print(f"SENSITIVITY (post-freeze edits, not for acceptance): panel 3-category kappa={s3}, binary excl={sb}")
    a = rep["acceptance"]
    print(f"Acceptance (Amendment 13): 3-category {a['panel_kappa_3cat']} >= {ACCEPTANCE_KAPPA}: {a['meets_3cat']}; "
          f"binary excl {a['panel_kappa_binary_abstain_excluded']} >= {ACCEPTANCE_BINARY_KAPPA}: {a['meets_binary']} "
          f"-> accepted: {a['meets_target']}")
    print(f"Report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
