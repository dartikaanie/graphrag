"""
summarize_claude_sensitivity.py
-------------------------------
4-judge SENSITIVITY panel (llama, deepseek, qwen3-235b + claude) on the official n=384 runs A-D, judge-v2.3, and a
per-judge table in the layout of the official report. Read-only on every official file; writes only
results/sensitivity_claude/summary_<tie_rule>.{md,json}. The official 3-judge panel is recomputed alongside (aggregate_labels)
as the reference column.

THE TIE RULE IS A REQUIRED PARAMETER (--tie-rule); fix it before labelling. Four judges can split 2-2.
Per judge: ordinal score ABSTAIN 0, FAKTUAL 0, SEBAGIAN 1, PENUH 2 (as judge_panel_aggregate); UNSPECIFIED = 1 or 2, both
tried (the panel label is it only if the result differs between the two, as in the official aggregation).

  Abstain: ABSTAIN iff a strict majority of the valid judges abstained (>= 3 of 4). Exactly half (2 of 4) is an abstain TIE.
  Label:   median of the valid judges' scores. With 4 judges the two middle scores s[1], s[2] may differ: a label TIE.
           (0 -> FAKTUAL, 1 -> HALUSINASI_SEBAGIAN, 2 -> HALUSINASI_PENUH.) A label tie that straddles 0|1 is also
           a tie on the binary hallucinated outcome; one that straddles 1|2 only a tie on severity.

  --tie-rule   resolves BOTH kinds of tie:
    lower        take the lower middle score (a hallucination tie -> NOT hallucinated); an abstain tie -> not ABSTAIN
    upper        take the upper middle score (a hallucination tie -> hallucinated);     an abstain tie -> not ABSTAIN
    unresolved   a tie -> UNRESOLVED (never dropped; counted)
    tiebreak:<judge_id>   the named judge (e.g. tiebreak:claude, tiebreak:deepseek) decides: the upper middle score if
                 that judge's own score >= it, else the lower one; for an abstain tie that judge's abstain vote decides
    official3    control: ignore claude, the official 3-judge panel
  With only 3 valid judges the median has no tie. Fewer than 3 valid -> UNRESOLVED.

Usage:
    python llm/evaluation/summarize_claude_sensitivity.py --tie-rule lower
    python llm/evaluation/summarize_claude_sensitivity.py --tie-rule tiebreak:claude --sens-dir results/sensitivity_claude
"""
import argparse
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import build_claude_judge_inputs as bci  # noqa: E402
from judge_binary import (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED,  # noqa: E402
                          category_of, severity_of)
from judge_panel_aggregate import (ABSTAIN, FAKTUAL, HALLUCINATED_LABELS, PENUH, SEBAGIAN, UNRESOLVED,  # noqa: E402
                                   UNSPECIFIED, LABEL_OF_SCORE, SCORE, aggregate_labels, panel_category)
from official_report import fleiss_kappa, three_numbers  # noqa: E402

OFFICIAL_JUDGES = ("llama", "deepseek", "qwen3-235b")
CLAUDE = "claude"
ALL_JUDGES = OFFICIAL_JUDGES + (CLAUDE,)
SENS_PANEL_ID = "panel-v2-claude-sensitivity"
FIXED_RULES = ("lower", "upper", "unresolved", "official3")


def parse_tie_rule(rule: str) -> str:
    if rule in FIXED_RULES:
        return rule
    if rule.startswith("tiebreak:") and rule.split(":", 1)[1] in ALL_JUDGES:
        return rule
    raise SystemExit(f"--tie-rule must be one of {FIXED_RULES} or tiebreak:<{'|'.join(ALL_JUDGES)}>, got {rule!r}")


def _scores(label):
    return (1, 2) if label == UNSPECIFIED else (SCORE[label],)


def aggregate_sensitivity(labels: dict, tie_rule: str) -> dict:
    """labels: {judge_id: label or None}. -> {"panel_label", "tie": None|"abstain"|"label", ...}."""
    if tie_rule == "official3":
        a = aggregate_labels({j: labels.get(j) for j in OFFICIAL_JUDGES})
        return {"panel_label": a["panel_label"], "tie": None}
    valid = {j: lab for j, lab in labels.items() if lab is not None}
    if len(valid) < 3:
        return {"panel_label": UNRESOLVED, "tie": None}
    n = len(valid)
    n_abs = sum(1 for lab in valid.values() if lab == ABSTAIN)
    breaker = tie_rule.split(":", 1)[1] if tie_rule.startswith("tiebreak:") else None
    tie = None
    if 2 * n_abs > n:
        return {"panel_label": ABSTAIN, "tie": None}
    if 2 * n_abs == n:  # abstain tie (n = 4, 2 abstain)
        tie = "abstain"
        if tie_rule == "unresolved":
            return {"panel_label": UNRESOLVED, "tie": tie}
        if breaker:
            if valid.get(breaker) is None:
                return {"panel_label": UNRESOLVED, "tie": tie}
            if valid[breaker] == ABSTAIN:
                return {"panel_label": ABSTAIN, "tie": tie}
        # lower / upper / tiebreak-with-an-answering-judge: not ABSTAIN; abstaining judges score 0 below
    results = set()
    for combo in product(*(_scores(lab) for lab in valid.values())):
        s = sorted(combo)
        lo, hi = (s[(n - 1) // 2], s[n // 2])  # n=3: both the median; n=4: the two middle scores
        if lo == hi:
            results.add(lo)
            continue
        tie = tie or "label"
        if tie_rule == "unresolved":
            return {"panel_label": UNRESOLVED, "tie": tie}
        if tie_rule == "lower":
            results.add(lo)
        elif tie_rule == "upper":
            results.add(hi)
        else:
            if valid.get(breaker) is None:
                return {"panel_label": UNRESOLVED, "tie": tie}
            b = min(_scores(valid[breaker])) if valid[breaker] != UNSPECIFIED else 1
            results.add(hi if b >= hi else lo)
    label = LABEL_OF_SCORE[results.pop()] if len(results) == 1 else UNSPECIFIED
    return {"panel_label": label, "tie": tie}


def load_records():
    """({cond: {judge: {question_id: record}}}, {cond: cell id}); official records filtered like the official report."""
    judge_files = bci.official_judge_files()
    data = {c: {j: {} for j in ALL_JUDGES} for c in bci.RUN_LABELS}
    for c, (_cell, _path, recs) in judge_files.items():
        for r in recs:
            if r["judge_id"] in OFFICIAL_JUDGES:
                data[c][r["judge_id"]][int(r["question_id"])] = r
    return data, {c: judge_files[c][0] for c in bci.RUN_LABELS}


def load_claude(sens_dir: Path, data: dict) -> dict:
    n = {}
    for c in bci.RUN_LABELS:
        p = sens_dir / f"{c}.jsonl"
        n[c] = 0
        if not p.exists():
            continue
        for r in bci.read_jsonl(p):
            if r.get("judge_id") == CLAUDE and r.get("panel_id") == SENS_PANEL_ID and r["prompt_version"] == bci.PROMPT_VERSION:
                data[c][CLAUDE][int(r["question_id"])] = r
                n[c] += 1
    return n


def label_of(rec):
    return rec["label"] if rec is not None and category_of(rec) is not None else None


def condition_summary(c: str, recs: dict, tie_rule: str) -> dict:
    qids = sorted(recs[OFFICIAL_JUDGES[0]])
    judge_cats = {j: [(category_of(recs[j][q]) if q in recs[j] else None) for q in qids] for j in ALL_JUDGES}
    judge_sev = {j: [severity_of(recs[j][q]) for q in qids if q in recs[j] and category_of(recs[j][q]) == CATEGORY_HALLUCINATED]
                 for j in ALL_JUDGES}
    off3, sens, ties = [], [], Counter()
    for q in qids:
        labs = {j: (label_of(recs[j].get(q))) for j in ALL_JUDGES}
        off3.append(aggregate_labels({j: labs[j] for j in OFFICIAL_JUDGES})["panel_label"])
        a = aggregate_sensitivity(labs, tie_rule)
        sens.append(a["panel_label"])
        if a["tie"]:
            ties[a["tie"]] += 1
    cats3 = [panel_category(p) for p in off3]
    cats4 = [panel_category(p) for p in sens]
    n_claude = sum(1 for x in judge_cats[CLAUDE] if x is not None)
    both = [(a, b) for a, b in zip(cats3, cats4) if a is not None and b is not None]
    return {
        "n": len(qids), "n_claude_records": len(recs[CLAUDE]), "n_claude_usable": n_claude,
        "judges": {j: three_numbers(judge_cats[j]) for j in ALL_JUDGES},
        "severity": {j: dict(Counter(judge_sev[j])) for j in ALL_JUDGES},
        "panel_official_3": three_numbers(cats3),
        "panel_sensitivity_4": three_numbers(cats4),
        "panel_label_distribution": {"official_3": dict(Counter(off3)), "sensitivity_4": dict(Counter(sens))},
        "unresolved": {"official_3": sum(1 for p in off3 if p == UNRESOLVED), "sensitivity_4": sum(1 for p in sens if p == UNRESOLVED)},
        "ties": dict(ties),
        "changed_vs_official_3cat": sum(1 for a, b in both if a != b), "n_compared": len(both),
        "claude_vs_official_panel_3cat_agreement": (round(sum(1 for q, p in zip(judge_cats[CLAUDE], cats3)
                                                              if q is not None and p is not None and q == p)
                                                          / max(1, sum(1 for q, p in zip(judge_cats[CLAUDE], cats3)
                                                                       if q is not None and p is not None)), 4)),
        "fleiss_kappa_3cat_4judges": fleiss_kappa([list(r) for r in zip(*(judge_cats[j] for j in ALL_JUDGES))
                                                   if None not in r]),
        "fleiss_kappa_3cat_official_3": fleiss_kappa([list(r) for r in zip(*(judge_cats[j] for j in OFFICIAL_JUDGES))
                                                      if None not in r]),
        "n_complete_4": sum(1 for r in zip(*(judge_cats[j] for j in ALL_JUDGES)) if None not in r),
    }


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def markdown(rep: dict) -> str:
    L = [f"# Claude sensitivity panel (judge-v2.3), tie rule `{rep['tie_rule']}`", "",
         "Sensitivity analysis: NOT the official panel. Rates are on the 3-category scale of the official report "
         "(rate = hallucinated / (hallucinated + not hallucinated), ABSTAIN excluded; abstain-as-failed = "
         "(hallucinated + ABSTAIN) / usable).", "",
         "Cell ids: " + ", ".join(f"{c} = {rep['cells'][c]}" for c in bci.RUN_LABELS), ""]
    L += ["## Panels", "", "| Run | Official 3-judge rate | Sensitivity 4-judge rate | Δ (pp) | abstain-as-failed 3→4 | "
          "ABSTAIN 3→4 | UNRESOLVED 3→4 | ties (abstain/label) | items changed (3cat) |", "|---|---|---|---|---|---|---|---|---|"]
    for c in bci.RUN_LABELS:
        s = rep["conditions"][c]
        p3, p4 = s["panel_official_3"], s["panel_sensitivity_4"]
        d = None if p3["rate_excl_abstain"] is None or p4["rate_excl_abstain"] is None else 100 * (p4["rate_excl_abstain"] - p3["rate_excl_abstain"])
        L.append(f"| {c} | {pct(p3['rate_excl_abstain'])} | {pct(p4['rate_excl_abstain'])} | "
                 f"{'n/a' if d is None else f'{d:+.1f}'} | {pct(p3['rate_abstain_as_failed'])}→{pct(p4['rate_abstain_as_failed'])} | "
                 f"{p3['abstain']}→{p4['abstain']} | {s['unresolved']['official_3']}→{s['unresolved']['sensitivity_4']} | "
                 f"{s['ties'].get('abstain', 0)}/{s['ties'].get('label', 0)} | {s['changed_vs_official_3cat']}/{s['n_compared']} |")
    L += ["", "## Per judge", "", "| Run | Judge | n usable | hallucinated | not halluc. | ABSTAIN | rate (excl. ABSTAIN) | "
          "abstain-as-failed | abstain rate | missing |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in bci.RUN_LABELS:
        for j in ALL_JUDGES:
            t = rep["conditions"][c]["judges"][j]
            L.append(f"| {c} | {j} | {t['n_usable']} | {t['hallucinated']} | {t['not_hallucinated']} | {t['abstain']} | "
                     f"{pct(t['rate_excl_abstain'])} | {pct(t['rate_abstain_as_failed'])} | {pct(t['abstain_rate'])} | {t['n_missing']} |")
    L += ["", "## Agreement", "", "| Run | Fleiss κ (3 official judges) | Fleiss κ (4 judges) | n complete (4) | "
          "claude = official panel (3-cat) | claude records / usable |", "|---|---|---|---|---|---|"]
    for c in bci.RUN_LABELS:
        s = rep["conditions"][c]
        L.append(f"| {c} | {s['fleiss_kappa_3cat_official_3']} | {s['fleiss_kappa_3cat_4judges']} | {s['n_complete_4']} | "
                 f"{pct(s['claude_vs_official_panel_3cat_agreement'])} | {s['n_claude_records']} / {s['n_claude_usable']} |")
    L += ["", "Rows with missing Claude records are UNRESOLVED in the 4-judge panel unless the rule is `official3`, and "
          "are left out of Fleiss κ; complete the 1,536 items before reading the table."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tie-rule", required=True, help=f"{FIXED_RULES} or tiebreak:<judge_id>")
    ap.add_argument("--sens-dir", default=str(ROOT / "results" / "sensitivity_claude"))
    args = ap.parse_args(argv)
    tie_rule = parse_tie_rule(args.tie_rule)
    sens = Path(args.sens_dir)
    data, cells = load_records()
    n_claude = load_claude(sens, data)
    rep = {"tie_rule": tie_rule, "cells": cells, "n_claude_loaded": n_claude,
           "conditions": {c: condition_summary(c, data[c], tie_rule) for c in bci.RUN_LABELS}}
    sens.mkdir(parents=True, exist_ok=True)
    stem = "summary_" + tie_rule.replace(":", "_")
    (sens / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (sens / f"{stem}.md").write_text(markdown(rep), encoding="utf-8")
    print(markdown(rep))
    print(f"wrote {sens / (stem + '.md')} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
