"""
stage2_report.py
=====================================
Stage 2 of the C retrieval v3 α selection (docs/DECISION_C_SCORING.md,
"Selection procedure" + Amendments 4, 5 and 6). No LLM calls: this
script only READS already-generated C runs and their judge outputs, checks
that they are exactly the runs the procedure requires, computes the pooled
binary hallucination counts per α, and applies
ctxrel_sweep.select_stage2_winner() -- the same helper the procedure names.

INPUT (all required)
------------------------------------------------------------
--run-files    The 4 C generation runs on the dev split: for each stage-2
               candidate α (STAGE2_ALPHAS: 0.5 and 0.25, the Amendment 5
               candidates; replaced by the top_two of the Amendment 6
               stage-1 recheck if that differs), one run with grounding
               on and one with grounding off.
--judge-files  judge-v2.x output file(s) (llm_judge_hallucination_v2_1.py
               --out ...) containing the PRIMARY judge's records for those
               4 runs. Other judge_ids in the same files are ignored.
--judge-version The FROZEN judge version from Amendment 4's Record
               (e.g. "judge-v2.1"). Every primary record must carry it.

CHECKS (the script stops on any failure -- a stage-2 result is only valid
if every check passes)
------------------------------------------------------------
  - Each run: condition C, c_retrieval_version v3, sample_split dev,
    alpha in the candidates, exactly one run per (alpha, grounding), and
    the same 50 dev question_ids in every run.
  - Each run's manifest started after the Amendment 6 graph update
    (TRUST_V3_APPLIED_AT), so every run used trustScore "v3-logscore".
    Timestamps are compared timezone-aware. While TRUST_V3_APPLIED_AT is
    not filled in, the script refuses to produce a report.
  - Each run's manifest records the graph's trustScoreVersion (read from
    Neo4j at run start) and it is "v3-logscore". A manifest without it is
    rejected.
  - Exactly one primary-judge record per (run, question), all with the
    frozen judge_version, none unusable (failed call / parse error).

METRICS (Amendment 4: binary primary metric; pooled over grounding on+off)
------------------------------------------------------------
  category per answer: ABSTAIN / NOT_HALLUCINATED / HALLUCINATED
  (judge_binary.category_of).
  - excl. ABSTAIN : count = HALLUCINATED,           n = HALLUCINATED + NOT_HALLUCINATED
  - incl. ABSTAIN : count = HALLUCINATED + ABSTAIN, n = all answers
    ("ABSTAIN counted as non-factual", as the Selection procedure says)
  Descriptive only: per-grounding breakdown, severity distribution,
  unverifiable-claim rate.

OUTPUT (llm/c_graphrag/results/stage2/)
------------------------------------------------------------
  stage2_report_<ts>.md / .json  -- tables, winner, reasoning, and two
  DRAFTS for the researcher to review and commit by hand:
    * the Outcome section text for docs/DECISION_C_SCORING.md
    * the new official_params block (status "locked")
  The script never edits the decision record or official_params.json.

CARA PAKAI (dari root repo)
------------------------------------------------------------
    python3 llm/c_graphrag/stage2_report.py \\
        --run-files llm/c_graphrag/results/<a0.5 grounded>.jsonl ... (4 file) \\
        --judge-files llm/evaluation/results/<jv2_1 output>.jsonl \\
        --judge-version judge-v2.1
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01_data_cleaning"))

from ctxrel_sweep import select_stage2_winner  # noqa: E402
from judge_binary import (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED,  # noqa: E402
                          CATEGORY_NOT_HALLUCINATED, category_of, is_unusable, severity_of)
from trust_formulas import TRUST_SCORE_VERSION  # noqa: E402

# Stage-2 candidates. Amendment 6: the v3 stage-1 recheck is binding; if its
# top_two differ from these, this constant is updated to the v3 top_two
# before any stage-2 run.
STAGE2_ALPHAS = (0.5, 0.25)
N_DEV = 50
JUDGE_ID = "primary"
# The APPLIED_AT line printed by `01_data_cleaning/14_apply_logscore_trust.py
# --apply` after its verification passed (Amendment 6, Record). ISO 8601 WITH
# a UTC offset, e.g. "2026-10-14T09:30:00+07:00". Runs must start after this.
# None = Amendment 6 not applied yet: no report is produced.
TRUST_V3_APPLIED_AT = None
OUT_DIR = Path(__file__).resolve().parent / "results" / "stage2"
OFFICIAL_DEFAULTS = {
    "sample_split": "test", "max_hops": 2,
    "edge_types": ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"],
    "use_author_trust": False, "accepted_only": False,
}


class Stage2Error(RuntimeError):
    """A required check failed; the stage-2 result would not be valid."""


# ---------------------------------------------------------------------
# Loading + checks
# ---------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _grounding(config: dict) -> str:
    g = config.get("require_grounding")
    if not isinstance(g, bool):
        raise Stage2Error(f"config has no boolean require_grounding: {config}")
    return "on" if g else "off"


def _parse_iso(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def trust_cutoff(applied_at: str | None = None) -> datetime:
    """Timezone-aware moment of the Amendment 6 graph update. Fails closed
    when it is not set, not ISO 8601, or has no UTC offset."""
    value = TRUST_V3_APPLIED_AT if applied_at is None else applied_at
    if not value:
        raise Stage2Error("TRUST_V3_APPLIED_AT is not set: fill it in with the APPLIED_AT line printed by "
                          "14_apply_logscore_trust.py --apply (Amendment 6) before running stage 2.")
    try:
        cutoff = _parse_iso(value)
    except ValueError:
        raise Stage2Error(f"TRUST_V3_APPLIED_AT is not an ISO 8601 timestamp: {value!r}") from None
    if cutoff.tzinfo is None:
        raise Stage2Error(f"TRUST_V3_APPLIED_AT has no UTC offset: {value!r} (write e.g. ...+07:00)")
    return cutoff


def _started_at(manifest: dict, name: str) -> datetime:
    """Timezone-aware run start; `started_at_utc` without an offset is UTC."""
    started = manifest.get("started_at_utc")
    try:
        dt = _parse_iso(started) if started else None
    except ValueError:
        dt = None
    if dt is None:
        raise Stage2Error(f"{name}: manifest has no valid started_at_utc ({started!r})")
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _recorded_trust_version(manifest: dict) -> str | None:
    for holder in (manifest, manifest.get("config") or {}, manifest.get("graph") or {}):
        for key in ("trustScoreVersion", "trust_score_version"):
            if holder.get(key) is not None:
                return holder[key]
    return None


def load_runs(run_files: list[Path], alphas=STAGE2_ALPHAS, applied_at: str | None = None) -> dict:
    """{(alpha, grounding): {"path", "question_ids", "started_at"}} after all run checks."""
    runs, qid_sets = {}, []
    cutoff = trust_cutoff(applied_at)
    for path in run_files:
        path = Path(path)
        recs = _read_jsonl(path)
        if not recs:
            raise Stage2Error(f"{path.name}: empty run file")
        configs = {json.dumps(r.get("config"), sort_keys=True) for r in recs}
        if len(configs) != 1:
            raise Stage2Error(f"{path.name}: records carry {len(configs)} different configs")
        cfg = recs[0]["config"]
        problems = [k for k, want in (("condition", "C"), ("c_retrieval_version", "v3"), ("sample_split", "dev"))
                    if cfg.get(k) != want]
        if problems:
            raise Stage2Error(f"{path.name}: wrong {problems} (config {cfg})")
        alpha = cfg.get("alpha")
        if alpha not in alphas:
            raise Stage2Error(f"{path.name}: alpha {alpha} is not a stage-2 candidate {list(alphas)}")
        key = (alpha, _grounding(cfg))
        if key in runs:
            raise Stage2Error(f"two runs for alpha={key[0]}, grounding={key[1]}: {runs[key]['path'].name}, {path.name}")
        qids = [int(r["question_id"]) for r in recs]
        if len(qids) != N_DEV or len(set(qids)) != N_DEV:
            raise Stage2Error(f"{path.name}: expected {N_DEV} distinct questions, got {len(set(qids))} of {len(qids)}")
        manifest = path.with_name(path.name + ".manifest.json")
        if not manifest.exists():
            raise Stage2Error(f"{path.name}: manifest missing ({manifest.name})")
        manifest_data = json.loads(manifest.read_text())
        started = manifest_data.get("started_at_utc")
        if _started_at(manifest_data, path.name) <= cutoff:
            raise Stage2Error(f"{path.name}: started {started}, not after the Amendment 6 graph update "
                              f"({cutoff.isoformat()})")
        version = _recorded_trust_version(manifest_data)
        if version != TRUST_SCORE_VERSION:
            raise Stage2Error(f"{path.name}: manifest records trustScoreVersion {version!r}, "
                              f"expected {TRUST_SCORE_VERSION!r}")
        runs[key] = {"path": path, "question_ids": set(qids), "started_at": started}
        qid_sets.append(set(qids))
    expected = {(a, g) for a in alphas for g in ("on", "off")}
    if set(runs) != expected:
        raise Stage2Error(f"missing runs for {sorted(expected - set(runs))}")
    if any(s != qid_sets[0] for s in qid_sets):
        raise Stage2Error("runs do not cover the same dev questions")
    return runs


def load_judgments(judge_files: list[Path], runs: dict, judge_version: str) -> dict:
    """{(alpha, grounding): {question_id: record}} -- primary judge only."""
    by_name = {info["path"].name: key for key, info in runs.items()}
    out = {key: {} for key in runs}
    for jf in judge_files:
        for r in _read_jsonl(Path(jf)):
            if r.get("judge_id") != JUDGE_ID:
                continue
            key = by_name.get(Path(str(r.get("source_file", ""))).name)
            if key is None:
                continue
            if r.get("judge_version") != judge_version:
                raise Stage2Error(f"record for q{r.get('question_id')} in {jf}: judge_version "
                                  f"{r.get('judge_version')!r} != frozen {judge_version!r}")
            qid = int(r["question_id"])
            if qid in out[key]:
                raise Stage2Error(f"duplicate primary record for alpha={key[0]}, grounding={key[1]}, q{qid}")
            out[key][qid] = r
    for key, recs in out.items():
        missing = runs[key]["question_ids"] - set(recs)
        if missing:
            raise Stage2Error(f"alpha={key[0]}, grounding={key[1]}: {len(missing)} answers not judged, "
                              f"e.g. {sorted(missing)[:5]}")
        bad = [q for q, r in recs.items() if is_unusable(r)]
        if bad:
            raise Stage2Error(f"alpha={key[0]}, grounding={key[1]}: unusable judge records "
                              f"(failed call / parse error) for {sorted(bad)[:5]} -- re-judge before stage 2")
    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def counts(records: list[dict]) -> dict:
    cats = Counter(category_of(r) for r in records)
    h, n_ok, ab = cats[CATEGORY_HALLUCINATED], cats[CATEGORY_NOT_HALLUCINATED], cats[CATEGORY_ABSTAIN]
    claims = [c for r in records for c in (r.get("claims") or [])]
    n_unver = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "UNVERIFIABLE")
    sev = Counter(severity_of(r) for r in records if category_of(r) == CATEGORY_HALLUCINATED)
    return {
        "n": len(records), "n_abstain": ab, "n_not_hallucinated": n_ok, "n_hallucinated": h,
        "hallucination_count_excl_abstain": h, "n_excl_abstain": h + n_ok,
        "hallucination_count_incl_abstain": h + ab, "n_incl_abstain": h + n_ok + ab,
        "rate_excl_abstain": round(h / (h + n_ok), 4) if h + n_ok else None,
        "rate_incl_abstain": round((h + ab) / (h + n_ok + ab), 4) if records else None,
        "abstain_rate": round(ab / len(records), 4) if records else None,
        "severity": dict(sev), "unverifiable_claim_rate": round(n_unver / len(claims), 4) if claims else None,
    }


def stage2_metrics(judged: dict, alphas=STAGE2_ALPHAS) -> tuple[dict, dict]:
    """(pooled metrics per alpha, per-(alpha, grounding) metrics)."""
    pooled = {a: counts([r for g in ("on", "off") for r in judged[(a, g)].values()]) for a in alphas}
    per_run = {f"{a}|{g}": counts(list(judged[(a, g)].values())) for a in alphas for g in ("on", "off")}
    return pooled, per_run


def draft_official_params(selection: dict) -> dict:
    if selection["keep_current"]:
        return {"status": "locked", "c_retrieval_version": "v2", "alpha": None, **OFFICIAL_DEFAULTS}
    return {"status": "locked", "c_retrieval_version": "v3", "alpha": selection["winner"], **OFFICIAL_DEFAULTS}


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def build_markdown(pooled, per_run, selection, runs, judge_version, judge_files, official) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Stage 2 Report -- C retrieval v3 alpha selection", "",
        f"Generated: {stamp}  |  judge: {judge_version} (primary)  |  pooled over grounding on + off", "",
        "Runs:", *[f"- alpha={a}, grounding={g}: `{info['path'].name}` (started {info['started_at']})"
                   for (a, g), info in sorted(runs.items())],
        "Judge files:", *[f"- `{Path(j).name}`" for j in judge_files], "",
        "## Pooled (selection metrics)", "",
        "| alpha | n | hallucinated | not hallucinated | ABSTAIN | rate excl. ABSTAIN | rate incl. ABSTAIN |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, m in pooled.items():
        lines.append(f"| {a} | {m['n']} | {m['n_hallucinated']} | {m['n_not_hallucinated']} | {m['n_abstain']} | "
                     f"{m['rate_excl_abstain']} ({m['hallucination_count_excl_abstain']}/{m['n_excl_abstain']}) | "
                     f"{m['rate_incl_abstain']} ({m['hallucination_count_incl_abstain']}/{m['n_incl_abstain']}) |")
    lines += ["", "## Per grounding (descriptive)", "",
              "| alpha | grounding | hallucinated | ABSTAIN | rate excl. ABSTAIN | severity | unverifiable-claim rate |",
              "|---|---|---|---|---|---|---|"]
    for k, m in per_run.items():
        a, g = k.split("|")
        lines.append(f"| {a} | {g} | {m['n_hallucinated']} | {m['n_abstain']} | {m['rate_excl_abstain']} | "
                     f"{m['severity']} | {m['unverifiable_claim_rate']} |")
    lines += ["", "## Selection (select_stage2_winner)", "",
              f"- winner: **alpha = {selection['winner']}**  |  keep_current: {selection['keep_current']}",
              f"- reasoning: {selection['reasoning']}", "",
              "## DRAFT -- Outcome section for docs/DECISION_C_SCORING.md (review, then paste)", "",
              "```markdown",
              f"- Stage 1 of record: Amendment 6 recheck (trustScore {TRUST_SCORE_VERSION}), "
              f"top_two = {list(STAGE2_ALPHAS)}.",
              f"- Stage 2: {judge_version} (primary judge), pooled grounding on + off, n = "
              f"{next(iter(pooled.values()))['n']} answers per alpha.",
              *[f"  - alpha = {a}: hallucination rate excl. ABSTAIN {m['rate_excl_abstain']} "
                f"({m['hallucination_count_excl_abstain']}/{m['n_excl_abstain']}); incl. ABSTAIN "
                f"{m['rate_incl_abstain']} ({m['hallucination_count_incl_abstain']}/{m['n_incl_abstain']})"
                for a, m in pooled.items()],
              f"- Chosen: {'C retrieval v2 (keep-current rule)' if selection['keep_current'] else 'alpha = ' + str(selection['winner'])}. "
              f"{selection['reasoning']}",
              f"- Date: [YYYY-MM-DD]  |  report: [path of this report]",
              "```", "",
              "## DRAFT -- official_params block (same JSON in the decision record AND official_params.json)", "",
              "```json", json.dumps(official, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage-2 report for the C retrieval v3 alpha selection (no API calls).")
    p.add_argument("--run-files", nargs="+", required=True)
    p.add_argument("--judge-files", nargs="+", required=True)
    p.add_argument("--judge-version", required=True, help="Frozen judge version from Amendment 4 (e.g. judge-v2.1)")
    args = p.parse_args()
    try:
        runs = load_runs([Path(f) for f in args.run_files])
        judged = load_judgments([Path(f) for f in args.judge_files], runs, args.judge_version)
    except Stage2Error as e:
        print(f"[STOP] {e}")
        return 1
    pooled, per_run = stage2_metrics(judged)
    selection = select_stage2_winner(pooled)
    official = draft_official_params(selection)
    md = build_markdown(pooled, per_run, selection, runs, args.judge_version, args.judge_files, official)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"stage2_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    Path(f"{stem}.md").write_text(md)
    Path(f"{stem}.json").write_text(json.dumps({
        "judge_version": args.judge_version, "pooled": {str(a): m for a, m in pooled.items()},
        "per_run": per_run, "selection": selection, "draft_official_params": official,
        "runs": {f"{a}|{g}": str(i["path"]) for (a, g), i in runs.items()},
        "judge_files": [str(j) for j in args.judge_files],
    }, indent=2, default=str))
    print(md)
    print(f"Report: {stem}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
