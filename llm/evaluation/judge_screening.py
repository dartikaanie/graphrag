"""
judge_screening.py
==================
Amendment 14 (draft) Qwen judge screening: applies the pre-registered
eligibility rule (reports/amendment14_draft.md, "Qwen judge eligibility rule")
to one screened judge's records, and reports -- for information only --
agreement with the frozen v1 human labels and the panel-v2 statistics that
the screened judge would give together with the STORED Llama-3.3 and
DeepSeek-V3.2 judge-v2.3 records. Reads stored outputs only; never calls an API.

Criteria (all must hold):
  (1) HALLUCINATED >= 5% of the usable non-ABSTAIN items;
  (2) 0 claims whose [R]-quoted evidence is absent from the reference but
      present in the candidate answer;
  (3) 60/60 usable records, no reasoning content, no <think>; a served-model
      mismatch invalidates the screening.

CLI (from llm/evaluation; joins through the verified key outside the repo):
    python3 judge_screening.py --screen-out results/screening_a14_qwen3-235b_judge-v2.3.jsonl \
        --stored-panel llm/evaluation/results/panel_calibration_judge-v2.3.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judge_binary import (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED,  # noqa: E402
                          category_of)
from judge_panel_aggregate import aggregate_labels, panel_category  # noqa: E402
from llm import archive_paths  # noqa: E402

MIN_HALLUCINATED_SHARE = 0.05
N_CALIBRATION_ITEMS = 60
STORED_JUDGES = ("llama", "deepseek")  # panel-v1 members kept in panel-v2
PANEL_V2_ID = "panel-v2"

_QUOTE = re.compile(r"\[R\]\s*[\"“”]([^\"“”]+)[\"“”]")
_MARKER = re.compile(r"\[[^\[\]\n]{1,40}\]")  # citation markers such as [SO-123], [K1]


def normalise(text: str) -> str:
    text = _MARKER.sub(" ", text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n.,;:!?'\"“”‘’`()")


def reference_quotes(record: dict) -> list[str]:
    """Every [R]-prefixed quoted string in the claims' evidence."""
    out = []
    for c in record.get("claims") or []:
        out += _QUOTE.findall(str(c.get("evidence") or ""))
    return out


def quote_check(record: dict, reference_text: str, candidate_text: str) -> dict:
    """Criterion (2) for one record: violations = quotes not in the reference but in the candidate."""
    ref, cand = normalise(reference_text), normalise(candidate_text)
    violations, in_neither, n = [], [], 0
    for q in reference_quotes(record):
        nq = normalise(q)
        if not nq:
            continue
        n += 1
        if nq in ref:
            continue
        (violations if nq in cand else in_neither).append(q)
    return {"n_quotes": n, "violations": violations, "in_neither": in_neither}


def criteria(records: dict, texts: dict, n_expected: int = N_CALIBRATION_ITEMS) -> dict:
    """records: {item_id: record or None}; texts: {item_id: (reference_text, candidate_text)}."""
    usable = {i: r for i, r in records.items() if r is not None and category_of(r) is not None}
    cats = Counter(category_of(r) for r in usable.values())
    non_abstain = cats[CATEGORY_HALLUCINATED] + cats[CATEGORY_NOT_HALLUCINATED]
    share = cats[CATEGORY_HALLUCINATED] / non_abstain if non_abstain else None
    c1 = share is not None and share >= MIN_HALLUCINATED_SHARE

    quote = {i: quote_check(r, *texts[i]) for i, r in usable.items()}
    n_viol = sum(len(q["violations"]) for q in quote.values())
    c2 = n_viol == 0

    present = [r for r in records.values() if r is not None]
    reasoning = sum(1 for r in present if r.get("reasoning_present"))
    think = sum(1 for r in present if "<think>" in (r.get("raw_response") or ""))
    first_attempt_fail = sum(1 for r in present if (r.get("attempts") or [{}])[0].get("error"))
    mismatch = sum(1 for r in present if r.get("served_model_mismatch"))
    c3 = len(usable) == n_expected and reasoning == 0 and think == 0
    return {
        "c1_not_degenerate": {"pass": c1, "hallucinated": cats[CATEGORY_HALLUCINATED], "non_abstain": non_abstain,
                              "share": round(share, 4) if share is not None else None,
                              "threshold": MIN_HALLUCINATED_SHARE, "categories": dict(cats)},
        "c2_no_candidate_as_reference_quote": {
            "pass": c2, "violations": n_viol, "n_quotes": sum(q["n_quotes"] for q in quote.values()),
            "quotes_in_neither_text": sum(len(q["in_neither"]) for q in quote.values()),
            "violating_items": {i: q["violations"] for i, q in quote.items() if q["violations"]}},
        "c3_clean_parse_no_reasoning": {"pass": c3, "usable": len(usable), "expected": n_expected,
                                        "reasoning_present": reasoning, "think_blocks": think,
                                        "first_attempt_parse_or_call_failures": first_attempt_fail},
        "served_model_mismatch": mismatch,
        "valid": mismatch == 0,
        "eligible": c1 and c2 and c3 and mismatch == 0,
    }


def per_condition_recall(rater: dict, human: dict, conditions: dict) -> dict:
    out = {}
    for cond in sorted(set(conditions.values())):
        ids = [i for i in human if conditions[i] == cond and human[i] == CATEGORY_HALLUCINATED]
        hit = sum(1 for i in ids if rater.get(i) == CATEGORY_HALLUCINATED)
        out[cond] = f"{hit}/{len(ids)}"
    return out


def panel_v2(stored: dict, screened: dict, items: list[str], screened_id: str) -> dict:
    """{item_id: panel record} from the stored llama/deepseek records and the screened judge's records."""
    out = {}
    for i in items:
        labels = {j: (stored[j].get(i) or {}).get("label") if _usable(stored[j].get(i)) else None for j in STORED_JUDGES}
        labels[screened_id] = screened[i]["label"] if _usable(screened.get(i)) else None
        out[i] = aggregate_labels(labels, PANEL_V2_ID)
    return out


def _usable(r) -> bool:
    return bool(r) and r.get("parse_ok") and not r.get("call_failed")


def main(argv=None) -> int:
    from annotation_tool import AnnotationError, assert_annotation_frozen, key_dir_from, load_annotator, load_verified_key
    from _judge_common import html_to_text, load_calibration_registry
    from judge_interjudge_analysis import fleiss_kappa
    from judge_panel import load_panel_items, read_jsonl
    from judge_panel_agreement import agreement_report

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screen-out", default=None, help="judge_panel.py --screen-judge --out file")
    p.add_argument("--stored-panel", required=True, help="Stored panel-v1 judge-v2.3 per-judge records")
    p.add_argument("--key-dir", default=None)
    p.add_argument("--out", default=None, help="Report stem (default: <screen-out stem>_report)")
    p.add_argument("--fairness", action="store_true",
                   help="Apply criterion (2) to every judge in --stored-panel (same method); no screened judge")
    args = p.parse_args(argv)
    if not args.fairness and not args.screen_out:
        p.error("--screen-out is required unless --fairness")

    try:
        frozen = assert_annotation_frozen("calibration", "annotator1")
    except AnnotationError as e:
        print(f"[ERROR] {e}")
        return 1
    key = load_verified_key("calibration", key_dir_from(args.key_dir))
    researcher, problems = load_annotator("calibration", "annotator1", key)
    if problems:
        print(f"[STOP] {len(problems)} problem(s) in the frozen annotation sheet")
        return 1
    key_items = key["items"]
    by_rq = {(k["run_id"], int(k["question_id"])): k["item_id"] for k in key_items}
    human = {k["item_id"]: (researcher.get(k["item_id"]) or {}).get("category") for k in key_items}
    conditions = {k["item_id"]: k["run_label"] for k in key_items}

    run_files = [str(archive_paths.REPO_ROOT / r["output_path"]) for r in load_calibration_registry()["runs"]]
    texts = {}
    for it in load_panel_items(run_files):
        i = by_rq.get((it["run_id"], it["question_id"]))
        if i is not None:
            texts[i] = (html_to_text(it["ground_truth_answer_html"]), it["candidate"])

    def by_item(records, judge_id=None):
        out = {}
        for r in records:
            i = by_rq.get((r.get("run_id"), int(r.get("question_id", -1))))
            if i is not None and (judge_id is None or r.get("judge_id") == judge_id) and r.get("prompt_version") == "judge-v2.3":
                out[i] = r
        return out

    if args.fairness:
        stored_all = read_jsonl(archive_paths.resolve(args.stored_panel))
        rep = fairness_report({j: by_item(stored_all, j) for j in sorted({r.get("judge_id") for r in stored_all})},
                              texts, args.stored_panel)
        stem = Path(args.out) if args.out else Path("results") / "screening_a14_criterion2_fairness_check"
        stem.with_name(stem.name + ".json").write_text(json.dumps(rep, indent=2) + "\n")
        stem.with_name(stem.name + ".md").write_text(fairness_markdown(rep))
        print(fairness_markdown(rep))
        print(f"Report -> {stem.with_name(stem.name + '.md')}")
        return 0

    screen_path = archive_paths.resolve(args.screen_out)
    screen_records = read_jsonl(screen_path)
    judge_ids = {r.get("judge_id") for r in screen_records}
    if len(judge_ids) != 1:
        print(f"[ERROR] the screening file must hold one judge, found {judge_ids}")
        return 1
    sid = judge_ids.pop()
    screened = by_item(screen_records)
    records = {i: screened.get(i) for i in human}
    crit = criteria(records, texts)

    stored_all = read_jsonl(archive_paths.resolve(args.stored_panel))
    stored = {j: by_item(stored_all, j) for j in (*STORED_JUDGES, "qwen")}
    raters = {sid: {i: category_of(r) if r else None for i, r in records.items()}}
    for j in (*STORED_JUDGES, "qwen"):
        raters[f"{j} (stored)"] = {i: category_of(stored[j][i]) if i in stored[j] else None for i in human}
    pv2 = panel_v2(stored, screened, list(human), sid)
    raters["panel"] = {i: panel_category(pv2[i]["panel_label"]) for i in human}
    rep = agreement_report(human, raters, conditions)

    complete = [i for i in human if all(raters[r][i] is not None for r in (sid, "llama (stored)", "deepseek (stored)"))]
    rows3 = [[raters[r][i] for r in (sid, "llama (stored)", "deepseek (stored)")] for i in complete]
    binary_rows = [[x for x in row] for row in rows3 if CATEGORY_ABSTAIN not in row]
    summary = {
        "screened_judge": sid, "model": screen_records[0].get("judge_model"), "annotation_freeze": frozen["sha256"],
        "criteria": crit,
        "cost_usd": round(sum(r.get("estimated_cost_usd") or 0 for r in screen_records), 6),
        "latency_s": _pcts([r.get("latency_s") for r in screen_records if r.get("latency_s") is not None]),
        "agreement_vs_frozen_v1": {
            name: {"kappa_3cat": rep["three_category"][name]["kappa"],
                   "kappa_binary_excl": rep["vs_human"][name]["abstain_excluded"]["kappa"],
                   "recall_excl": rep["vs_human"][name]["abstain_excluded"].get("hallucination_recall"),
                   "false_alarm_excl": rep["vs_human"][name]["abstain_excluded"].get("false_alarm_rate"),
                   "per_condition_recall": per_condition_recall(raters[name], human, conditions)}
            for name in raters},
        "panel_v2": {
            "members": ["llama (stored)", "deepseek (stored)", sid],
            "labels": dict(Counter(pv2[i]["panel_label"] for i in human)),
            "unresolved": sum(1 for i in human if pv2[i]["panel_label"] == "UNRESOLVED"),
            "incomplete": sum(1 for i in human if pv2[i]["panel_incomplete"]),
            "fleiss_3cat_complete": {"kappa": fleiss_kappa(rows3, (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED,
                                                                   CATEGORY_NOT_HALLUCINATED)), "n": len(rows3)},
            "fleiss_binary_excl": {"kappa": fleiss_kappa(binary_rows, (CATEGORY_HALLUCINATED, CATEGORY_NOT_HALLUCINATED)),
                                   "n": len(binary_rows)},
        },
        "agreement_report": rep,
    }
    stem = Path(args.out) if args.out else screen_path.with_name(screen_path.stem + "_report")
    md_path, json_path = stem.with_name(stem.name + ".md"), stem.with_name(stem.name + ".json")
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    md_path.write_text(markdown(summary))
    print(markdown(summary))
    print(f"Report -> {md_path}")
    return 0


def fairness_report(records_by_judge: dict, texts: dict, source: str) -> dict:
    """Criterion (2), exactly as for the screened judges, on stored records of other judges."""
    out = {"source": source, "method": "judge_screening.quote_check (criterion (2) of the eligibility rule)",
           "judges": {}}
    for j, recs in records_by_judge.items():
        checks = {i: quote_check(r, *texts[i]) for i, r in recs.items() if i in texts and _usable(r)}
        affected = {i: c["violations"] for i, c in checks.items() if c["violations"]}
        out["judges"][j] = {"model": next(iter(recs.values())).get("judge_model") if recs else None,
                            "usable_records": len(checks), "n_quotes": sum(c["n_quotes"] for c in checks.values()),
                            "violations": sum(len(v) for v in affected.values()),
                            "affected_items": len(affected), "violating_items": dict(sorted(affected.items())),
                            "quotes_in_neither_text": sum(len(c["in_neither"]) for c in checks.values())}
    return out


def fairness_markdown(rep: dict) -> str:
    lines = ["# Criterion (2) fairness check: stored judge-v2.3 calibration outputs", "",
             f"Source: `{rep['source']}`. Method: {rep['method']}; stored outputs only, no API call.", "",
             "| Judge | Model | Usable records | [R] quotes | Violations | Affected items | Items | Quotes in neither text |",
             "|---|---|---|---|---|---|---|---|"]
    for j, r in rep["judges"].items():
        lines.append(f"| {j} | {r['model']} | {r['usable_records']} | {r['n_quotes']} | {r['violations']} | "
                     f"{r['affected_items']} | {', '.join(r['violating_items']) or '--'} | {r['quotes_in_neither_text']} |")
    return "\n".join(lines) + "\n"


def _pcts(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {"p50": s[len(s) // 2], "p95": s[max(0, int(round(0.95 * len(s))) - 1)], "n": len(s)}


def _k(x):
    return "--" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _frac(f):
    if isinstance(f, dict):
        return f"{f.get('x')}/{f.get('n')}" if "x" in f else json.dumps(f)
    return _k(f)


def markdown(s: dict) -> str:
    c = s["criteria"]
    c1, c2, c3 = (c["c1_not_degenerate"], c["c2_no_candidate_as_reference_quote"], c["c3_clean_parse_no_reasoning"])
    lines = [
        f"# Screening: {s['model']} (judge-v2.3, 60 calibration items)", "",
        f"**Eligible: {'YES' if c['eligible'] else 'NO'}** (rule: reports/amendment14_draft.md)", "",
        "| Criterion | Result | Pass |", "|---|---|---|",
        f"| (1) HALLUCINATED >= 5% of non-ABSTAIN | {c1['hallucinated']}/{c1['non_abstain']} = {_k(c1['share'])} | {c1['pass']} |",
        f"| (2) candidate text as [R] quote | {c2['violations']} violations in {c2['n_quotes']} quotes "
        f"({c2['quotes_in_neither_text']} in neither text) | {c2['pass']} |",
        f"| (3) usable, no reasoning | {c3['usable']}/{c3['expected']} usable; reasoning {c3['reasoning_present']}; "
        f"<think> {c3['think_blocks']}; first-attempt failures {c3['first_attempt_parse_or_call_failures']} | {c3['pass']} |",
        f"| validity: served-model mismatch | {c['served_model_mismatch']} | {c['valid']} |", "",
        f"Cost ${s['cost_usd']:.4f}; latency {s['latency_s']}", "",
        "## Agreement with the frozen v1 labels (information only)", "",
        "| Rater | 3-cat kappa | binary kappa (excl.) | recall (excl.) | false alarm (excl.) | recall per condition |",
        "|---|---|---|---|---|---|",
    ]
    for name, a in s["agreement_vs_frozen_v1"].items():
        lines.append(f"| {name} | {_k(a['kappa_3cat'])} | {_k(a['kappa_binary_excl'])} | {_frac(a['recall_excl'])} | "
                     f"{_frac(a['false_alarm_excl'])} | {', '.join(f'{k} {v}' for k, v in a['per_condition_recall'].items())} |")
    pv = s["panel_v2"]
    lines += ["", "`panel` = panel-v2 (stored Llama + stored DeepSeek + the screened judge, 2-of-3).", "",
              "## panel-v2 from stored outputs (no extra calls)", "",
              f"- labels: {pv['labels']}", f"- UNRESOLVED: {pv['unresolved']}; incomplete panels: {pv['incomplete']}",
              f"- Fleiss' kappa, 3 categories (complete items): {_k(pv['fleiss_3cat_complete']['kappa'])} "
              f"(n = {pv['fleiss_3cat_complete']['n']})",
              f"- Fleiss' kappa, binary, ABSTAIN excluded: {_k(pv['fleiss_binary_excl']['kappa'])} "
              f"(n = {pv['fleiss_binary_excl']['n']})", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
