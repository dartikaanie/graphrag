#!/usr/bin/env python3
"""
trust_ablation_report.py
=====================================
C-primary vs C-ablation on the official evaluation (docs/DECISION_C_SCORING.md,
Amendment 10). No LLM calls: reads the 4 test-sample C runs and their judge
outputs, and reports descriptively (no inferential test):

  - binary hallucination rate per run, per grounding (on, off) and pooled,
    with ABSTAIN excluded and with ABSTAIN counted as non-factual (Amendment 4);
  - paired counts per question (same question, same grounding):
    hallucinated in C-ablation only / in C-primary only / in both / in neither,
    once with ABSTAIN counted as non-factual and once on the pairs where
    neither answer is ABSTAIN (pairs left out are counted);
  - cosine similarity to the accepted answer and NF2 (share of answers with a
    valid citation) per run.

Judge records are read exactly as stage 2 reads them (stage2_report.py:
primary judge only, the frozen judge_version, no unusable record).

CHECKS (stops on any failure)
------------------------------------------------------------
  - official_params.json is "selected"; each run's alpha is alpha_primary or
    alpha_ablation, which gives its role (c_primary / c_ablation). A role in
    the run manifest, if present, must agree.
  - Each run: condition C, v3, sample_split test, expansion_require_answer
    true, the official leakage mode; one run per (role, grounding).
  - For each grounding, the C-primary and C-ablation configs differ ONLY in
    alpha (Amendment 10: identical procedure).
  - Every run covers exactly the frozen test sample (llm/sample_ids/test_ids_v1.json).

    python3 llm/evaluation/trust_ablation_report.py \\
        --run-files <C-primary on> <C-primary off> <C-ablation on> <C-ablation off> \\
        --judge-files <judge-v2.x output>.jsonl --judge-version judge-v2.1
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "llm" / "c_graphrag"))
sys.path.insert(0, str(REPO_ROOT))

import stage2_report as s2  # noqa: E402
from judge_binary import CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED, category_of  # noqa: E402

from llm.c_graphrag.official_params import (ROLE_ABLATION, ROLE_PRIMARY, load_official_params,  # noqa: E402
                                            official_alphas, role_for_alpha)
from llm.official_run_matrix import LEAKAGE_MODE  # noqa: E402
from llm.sample_split import load_frozen_ids  # noqa: E402

ROLES = (ROLE_PRIMARY, ROLE_ABLATION)
GROUNDINGS = ("on", "off")
OUT_DIR = REPO_ROOT / "llm" / "evaluation" / "results" / "trust_ablation"
# keys that may differ between C-primary and C-ablation of the same grounding
ALLOWED_DIFF = {"alpha"}


class AblationError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def frozen_test_ids() -> list[int]:
    return [int(q) for q in load_frozen_ids("test")["ids"]]


def config_differences(cfg_a: dict, cfg_b: dict, allowed=ALLOWED_DIFF) -> list[str]:
    """Config keys (other than `allowed`) whose values differ between two runs."""
    return sorted(k for k in set(cfg_a) | set(cfg_b) if k not in allowed and cfg_a.get(k) != cfg_b.get(k))


def load_runs(run_files: list[Path], official: dict | None) -> dict:
    """{(role, grounding): {"path", "question_ids", "config", "records"}} after all checks."""
    if not official_alphas(official):
        raise AblationError("official_params.json is not 'selected': alpha_primary/alpha_ablation are not set "
                            "(fill them in from the stage-2 report first).")
    sample = set(frozen_test_ids())
    runs = {}
    for path in map(Path, run_files):
        recs = _read_jsonl(path)
        if not recs:
            raise AblationError(f"{path.name}: empty run file")
        if len({json.dumps(r.get("config"), sort_keys=True) for r in recs}) != 1:
            raise AblationError(f"{path.name}: records carry more than one config")
        cfg = recs[0]["config"]
        wrong = [k for k, want in (("condition", "C"), ("c_retrieval_version", "v3"), ("sample_split", "test"),
                                   ("expansion_require_answer", True), ("leakage", LEAKAGE_MODE))
                 if cfg.get(k) != want]
        if wrong:
            raise AblationError(f"{path.name}: wrong {wrong} (config {cfg})")
        role = role_for_alpha(official, cfg.get("alpha"))
        if role is None:
            raise AblationError(f"{path.name}: alpha {cfg.get('alpha')!r} is neither alpha_primary nor alpha_ablation")
        manifest = path.with_name(path.name + ".manifest.json")
        if manifest.exists():
            recorded = json.loads(manifest.read_text()).get("official_role")
            if recorded is not None and recorded != role:
                raise AblationError(f"{path.name}: manifest role {recorded!r} != {role!r} from alpha")
        key = (role, "on" if cfg.get("require_grounding") else "off")
        if key in runs:
            raise AblationError(f"two runs for {key}: {runs[key]['path'].name}, {path.name}")
        qids = [int(r["question_id"]) for r in recs]
        if len(set(qids)) != len(qids) or set(qids) != sample:
            raise AblationError(f"{path.name}: questions are not exactly the frozen test sample "
                                f"({len(set(qids) - sample)} extra, {len(sample - set(qids))} missing)")
        runs[key] = {"path": path, "question_ids": set(qids), "config": cfg,
                     "records": {int(r["question_id"]): r for r in recs}}
    expected = {(r, g) for r in ROLES for g in GROUNDINGS}
    if set(runs) != expected:
        raise AblationError(f"missing runs for {sorted(expected - set(runs))}")
    for g in GROUNDINGS:
        diff = config_differences(runs[(ROLE_PRIMARY, g)]["config"], runs[(ROLE_ABLATION, g)]["config"])
        if diff:
            raise AblationError(f"grounding {g}: C-primary and C-ablation configs differ in {diff}, not only alpha")
    return runs


def _is_hallucinated(record: dict, abstain_non_factual: bool) -> bool:
    cat = category_of(record)
    return cat == CATEGORY_HALLUCINATED or (abstain_non_factual and cat == CATEGORY_ABSTAIN)


def paired_table(primary: dict, ablation: dict, abstain: str = "non_factual") -> dict:
    """Paired counts over the questions judged in both runs.
    abstain="non_factual": ABSTAIN counts as hallucinated. abstain="exclude":
    pairs where either answer is ABSTAIN are left out (and counted)."""
    if abstain not in ("non_factual", "exclude"):
        raise ValueError(abstain)
    out = {"ablation_only": 0, "primary_only": 0, "both": 0, "neither": 0, "n_pairs": 0, "n_excluded_abstain": 0}
    for qid in sorted(set(primary) & set(ablation)):
        p, a = primary[qid], ablation[qid]
        if abstain == "exclude" and CATEGORY_ABSTAIN in (category_of(p), category_of(a)):
            out["n_excluded_abstain"] += 1
            continue
        hp, ha = _is_hallucinated(p, abstain == "non_factual"), _is_hallucinated(a, abstain == "non_factual")
        out["both" if hp and ha else "ablation_only" if ha else "primary_only" if hp else "neither"] += 1
        out["n_pairs"] += 1
    return out


def _add_tables(x: dict, y: dict) -> dict:
    return {k: x[k] + y[k] for k in x}


def run_metrics(records: dict) -> dict:
    cos = [r["cosine_similarity"] for r in records.values() if r.get("cosine_similarity") is not None]
    cit = [bool(r.get("has_valid_citation")) for r in records.values()]
    return {"n": len(records),
            "cosine_mean": round(statistics.mean(cos), 4) if cos else None,
            "cosine_median": round(statistics.median(cos), 4) if cos else None,
            "nf2_pct_valid_citation": round(100 * sum(cit) / len(cit), 2) if cit else None}


def build_report(runs: dict, judged: dict) -> dict:
    hall = {f"{role}|{g}": s2.counts(list(judged[(role, g)].values())) for role in ROLES for g in GROUNDINGS}
    hall.update({f"{role}|pooled": s2.counts([r for g in GROUNDINGS for r in judged[(role, g)].values()])
                 for role in ROLES})
    paired = {}
    for mode in ("non_factual", "exclude"):
        per_g = {g: paired_table(judged[(ROLE_PRIMARY, g)], judged[(ROLE_ABLATION, g)], mode) for g in GROUNDINGS}
        paired[mode] = {**per_g, "pooled": _add_tables(per_g["on"], per_g["off"])}
    other = {f"{role}|{g}": run_metrics(runs[(role, g)]["records"]) for role in ROLES for g in GROUNDINGS}
    alphas = {role: runs[(role, "on")]["config"]["alpha"] for role in ROLES}
    return {"alphas": alphas, "hallucination": hall, "paired": paired, "cosine_nf2": other,
            "runs": {f"{r}|{g}": runs[(r, g)]["path"].name for r, g in runs}}


def build_markdown(rep: dict, judge_version: str) -> str:
    a = rep["alphas"]
    lines = ["# C-primary vs C-ablation (Amendment 10; test sample, descriptive)", "",
             f"- C-primary: alpha = {a[ROLE_PRIMARY]}  |  C-ablation: alpha = {a[ROLE_ABLATION]}  |  "
             f"judge {judge_version} (primary)", "",
             "## Binary hallucination rate", "",
             "| run | grounding | n | hallucinated | ABSTAIN | rate excl. ABSTAIN | rate incl. ABSTAIN |",
             "|---|---|---|---|---|---|---|"]
    for role in ROLES:
        for g in (*GROUNDINGS, "pooled"):
            m = rep["hallucination"][f"{role}|{g}"]
            lines.append(f"| {role} | {g} | {m['n']} | {m['n_hallucinated']} | {m['n_abstain']} | "
                         f"{m['rate_excl_abstain']} | {m['rate_incl_abstain']} |")
    for mode, title in (("non_factual", "ABSTAIN counted as non-factual"),
                        ("exclude", "pairs with an ABSTAIN answer left out")):
        lines += ["", f"## Paired counts per question ({title})", "",
                  "| grounding | C-ablation only | C-primary only | both | neither | pairs | left out (ABSTAIN) |",
                  "|---|---|---|---|---|---|---|"]
        for g in (*GROUNDINGS, "pooled"):
            t = rep["paired"][mode][g]
            lines.append(f"| {g} | {t['ablation_only']} | {t['primary_only']} | {t['both']} | {t['neither']} | "
                         f"{t['n_pairs']} | {t['n_excluded_abstain']} |")
    lines += ["", "## Cosine similarity and NF2", "",
              "| run | grounding | n | cosine mean | cosine median | NF2 % valid citation |", "|---|---|---|---|---|---|"]
    for role in ROLES:
        for g in GROUNDINGS:
            m = rep["cosine_nf2"][f"{role}|{g}"]
            lines.append(f"| {role} | {g} | {m['n']} | {m['cosine_mean']} | {m['cosine_median']} | "
                         f"{m['nf2_pct_valid_citation']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="C-primary vs C-ablation report (Amendment 10, no API calls).")
    p.add_argument("--run-files", nargs=4, required=True)
    p.add_argument("--judge-files", nargs="+", required=True)
    p.add_argument("--judge-version", required=True, help="the frozen judge version (Amendment 4 Record)")
    args = p.parse_args()
    try:
        runs = load_runs(args.run_files, load_official_params())
        judged = s2.load_judgments(list(map(Path, args.judge_files)), runs, args.judge_version)
    except (AblationError, s2.Stage2Error) as e:
        print(f"[STOP] {e}")
        return 1
    rep = build_report(runs, judged)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = build_markdown(rep, args.judge_version)
    (OUT_DIR / f"trust_ablation_report_{ts}.md").write_text(md)
    (OUT_DIR / f"trust_ablation_report_{ts}.json").write_text(json.dumps(
        {**rep, "judge_version": args.judge_version, "judge_files": args.judge_files}, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
