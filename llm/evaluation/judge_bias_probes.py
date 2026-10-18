#!/usr/bin/env python3
"""
judge_bias_probes.py
=====================================
Judge bias probes on the DEV set (docs/DECISION_C_SCORING.md, Amendment 11,
change 6). Descriptive only. Every judge call goes through the judge-v2.2
runner (llm_judge_hallucination_v2_2.judge_one), so the probes measure the
instrument that is calibrated.

  1. Authority: N_PROBE B/C/D answers (seed fixed, spread evenly over the
     conditions) judged twice -- blind-v3 (citations removed) and with the
     citations kept as [Kn] labels -> flip rate of the 3-category label
     (ABSTAIN / not hallucinated / hallucinated) and of HALU on items both
     calls judge non-ABSTAIN.
  2. Length: point-biserial correlation of the answer length (cl100k tokens
     of the blind-v3 candidate) with HALU (1/0, non-ABSTAIN), per condition,
     on the blind-v3 judgments of step 1 (no extra calls).
  3. Order: the same items with the [K1]..[K5] context order shuffled (fixed
     seed per item) -> flip rate against the blind-v3 judgment.

Default is a PLAN (no API call): it prints the sampled items and the number
of judge calls. Judge calls happen only with --run.

    python3 judge_bias_probes.py --run-files <dev B/C/D runs>              # plan, 0 API calls
    python3 judge_bias_probes.py --run-files <dev B/C/D runs> --run        # 3 x N_PROBE calls per judge
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import llm_judge_hallucination_v2_2 as runner  # noqa: E402
from judge_binary import CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, category_of  # noqa: E402

N_PROBE = 40
SEED = 20261019
RETRIEVAL_CONDITIONS = ("B", "C", "D")
OUT_DIR = Path(__file__).resolve().parent / "results" / "bias_probes"


class ProbeError(ValueError):
    pass


def run_sample_split(path) -> str | None:
    with open(path) as f:
        for line in f:
            if line.strip():
                return (json.loads(line).get("config") or {}).get("sample_split")
    return None


def sample_probe_items(items: list[dict], n: int = N_PROBE, seed: int = SEED) -> list[dict]:
    """n items from the B/C/D items, spread as evenly as possible over the conditions present (sorted
    by run and question within a condition, drawn with random.Random(seed, condition))."""
    by_cond: dict[str, list] = {}
    for it in items:
        if str(it.get("condition")).upper() in RETRIEVAL_CONDITIONS:
            by_cond.setdefault(str(it["condition"]).upper(), []).append(it)
    if not by_cond:
        raise ProbeError("no B/C/D items among the run files")
    conds = sorted(by_cond)
    base, rem = divmod(n, len(conds))
    out = []
    for i, c in enumerate(conds):
        pool = sorted(by_cond[c], key=lambda it: (str(it["run_id"]), it["question_id"]))
        k = base + (1 if i < rem else 0)
        if len(pool) < k:
            raise ProbeError(f"condition {c} has {len(pool)} items, needs {k}")
        out += random.Random(f"{seed}|{c}").sample(pool, k)
    return out


def shuffled_context(item: dict, seed: int = SEED) -> list:
    ctx = list(item.get("retrieved_context") or [])
    random.Random(f"{seed}|order|{item['run_id']}|{item['question_id']}").shuffle(ctx)
    return ctx


def flip_stats(base: list[dict], variant: list[dict]) -> dict:
    """Flip rate between paired judgments (same items, same order): 3-category label, and HALU on items
    both judge non-ABSTAIN. Unusable records are excluded and counted."""
    pairs = [(category_of(a), category_of(b)) for a, b in zip(base, variant)]
    usable = [(a, b) for a, b in pairs if a is not None and b is not None]
    both = [(a, b) for a, b in usable if CATEGORY_ABSTAIN not in (a, b)]
    flips3 = sum(1 for a, b in usable if a != b)
    flipsb = sum(1 for a, b in both if a != b)
    return {
        "n_pairs": len(pairs), "n_unusable": len(pairs) - len(usable),
        "n_usable": len(usable), "flips_3cat": flips3,
        "flip_rate_3cat": round(flips3 / len(usable), 4) if usable else None,
        "n_both_non_abstain": len(both), "flips_halu": flipsb,
        "flip_rate_halu": round(flipsb / len(both), 4) if both else None,
        "to_hallucinated": sum(1 for a, b in both if a != CATEGORY_HALLUCINATED and b == CATEGORY_HALLUCINATED),
        "to_not_hallucinated": sum(1 for a, b in both if a == CATEGORY_HALLUCINATED and b != CATEGORY_HALLUCINATED),
    }


def point_biserial(lengths: list[float], halu: list[int]) -> float | None:
    """Pearson correlation of a continuous variable with a 0/1 variable; None if undefined."""
    n = len(lengths)
    if n < 3 or len(set(halu)) < 2 or len(set(lengths)) < 2:
        return None
    ml, mh = sum(lengths) / n, sum(halu) / n
    cov = sum((x - ml) * (y - mh) for x, y in zip(lengths, halu))
    sx = math.sqrt(sum((x - ml) ** 2 for x in lengths))
    sy = math.sqrt(sum((y - mh) ** 2 for y in halu))
    return round(cov / (sx * sy), 4)


def token_length(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))


def length_probe(items: list[dict], records: list[dict]) -> dict:
    from judge_blinding import blind_candidate_v3

    out = {}
    for c in sorted({str(it["condition"]).upper() for it in items}):
        pairs = [(token_length(blind_candidate_v3(it["candidate"])), 1 if category_of(r) == CATEGORY_HALLUCINATED else 0)
                 for it, r in zip(items, records)
                 if str(it["condition"]).upper() == c and category_of(r) not in (None, CATEGORY_ABSTAIN)]
        out[c] = {"n": len(pairs), "n_hallucinated": sum(h for _, h in pairs),
                  "r_point_biserial": point_biserial([x for x, _ in pairs], [h for _, h in pairs])}
    return out


def run_probes(items: list[dict], judge_id: str, client, config, bodies: dict, references: dict) -> dict:
    """bodies/references: {question_id: text}. Three calls per item through runner.judge_one."""
    def judge(it, **kw):
        return runner.judge_one(it, judge_id, client, config, bodies.get(it["question_id"], ""),
                                references.get(it["question_id"], ""), **kw)

    blind = [judge(it) for it in items]
    labels = [judge(it, keep_citation_labels=True) for it in items]
    order = [judge({**it, "retrieved_context": shuffled_context(it)}) for it in items]
    return {
        "judge_id": judge_id, "judge_model": config.model, "prompt_version": runner.PROMPT_VERSION,
        "n_items": len(items), "items": [{"run_id": it["run_id"], "question_id": it["question_id"],
                                          "condition": it["condition"]} for it in items],
        "authority": flip_stats(blind, labels),
        "order": flip_stats(blind, order),
        "length": length_probe(items, blind),
        "records": {"blind": blind, "citation_labels": labels, "context_shuffled": order},
    }


def probe_markdown(rep: dict) -> str:
    a, o = rep["authority"], rep["order"]
    lines = [f"# Judge bias probes ({rep['prompt_version']}, {rep['judge_id']} = {rep['judge_model']}; dev set)", "",
             f"Items: {rep['n_items']} B/C/D answers. Descriptive only (Amendment 11).", "",
             "| probe | pairs | usable | flips (3 categories) | flip rate | both non-ABSTAIN | HALU flips | HALU flip rate |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in (("authority: [Kn] kept vs blind-v3", a), ("order: context shuffled vs original", o)):
        lines.append(f"| {name} | {s['n_pairs']} | {s['n_usable']} | {s['flips_3cat']} | {s['flip_rate_3cat']} | "
                     f"{s['n_both_non_abstain']} | {s['flips_halu']} | {s['flip_rate_halu']} |")
    lines += ["", "## Length vs HALU (point-biserial, blind-v3 judgments, non-ABSTAIN)", "",
              "| condition | n | hallucinated | r |", "|---|---|---|---|"]
    for c, s in rep["length"].items():
        lines.append(f"| {c} | {s['n']} | {s['n_hallucinated']} | {s['r_point_biserial']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from _judge_common import html_to_text, load_question_bodies
    from judge_clients import get_judge_client

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-files", nargs="+", required=True, help="dev runs of B, C and/or D")
    p.add_argument("--judge", choices=["primary", "secondary"], default="primary")
    p.add_argument("--n", type=int, default=N_PROBE)
    p.add_argument("--run", action="store_true", help="make the judge calls (paid); default: plan only")
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    args = p.parse_args(argv)

    wrong = [Path(f).name for f in args.run_files if run_sample_split(f) != "dev"]
    if wrong:
        print(f"[STOP] bias probes run on the dev set only; not dev: {wrong}")
        return 1
    items = runner.load_items(args.run_files, None)
    sample = sample_probe_items(items, args.n)
    per_cond = ", ".join(f"{c}: {sum(1 for i in sample if i['condition'] == c)}" for c in RETRIEVAL_CONDITIONS)
    print(f"[plan] {len(sample)} items ({per_cond}); {3 * len(sample)} judge calls with {args.judge} "
          f"(blind-v3, citation labels kept, context shuffled)")
    if not args.run:
        print("[plan] no API call made; re-run with --run")
        return 0
    qids = sorted({it["question_id"] for it in sample})
    bodies = {q: html_to_text(b) for q, b in load_question_bodies(args.questions_parquet, qids).items()}
    refs = {it["question_id"]: html_to_text(it["ground_truth_answer_html"]) for it in sample}
    client, config = get_judge_client(args.judge)
    rep = run_probes(sample, args.judge, client, config, bodies, refs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT_DIR / f"bias_probes_{args.judge}_{ts}.json").write_text(json.dumps(rep, indent=2, default=str))
    (OUT_DIR / f"bias_probes_{args.judge}_{ts}.md").write_text(probe_markdown(rep))
    print(probe_markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
