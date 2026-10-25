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
               candidate α (STAGE2_ALPHAS: the top_two of the stage-1
               report of record, see stage2_candidates.py), one run with grounding on
               and one with grounding off.
--judge-files  judge-v2.x output file(s) (llm_judge_hallucination_v2_1.py
               --out ...) containing the PRIMARY judge's records for those
               4 runs. Other judge_ids in the same files are ignored.
--judge-version The FROZEN judge version from Amendment 4's Record
               (e.g. "judge-v2.1"). Every primary record must carry it.

CHECKS (the script stops on any failure -- a stage-2 result is only valid
if every check passes)
------------------------------------------------------------
  - Each run: condition C, c_retrieval_version v3, sample_split dev,
    alpha in the candidates, exactly one run per (alpha, grounding).
  - Each run's config pins the answerable-only expansion
    (expansion_require_answer = true, Amendment 9) and the official leakage
    mode (llm/official_run_matrix.LEAKAGE_MODE: "thread", Amendment 15).
  - Each run covers exactly the stage-1 dev questions: the frozen dev list
    (llm/sample_ids/dev_ids_v1.json, positions 385-434), not merely the
    same questions across the 4 runs.
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
  The script never edits the decision record; it writes official_params.json
  only with the explicit --apply flag (Amendment 14 draft, plan step 6).

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxrel_sweep import select_stage2_winner  # noqa: E402
from judge_binary import (CATEGORY_ABSTAIN, CATEGORY_HALLUCINATED,  # noqa: E402
                          CATEGORY_NOT_HALLUCINATED, category_of, is_unusable, severity_of)
from trust_formulas import TRUST_SCORE_VERSION  # noqa: E402

from llm.official_run_matrix import LEAKAGE_MODE as OFFICIAL_LEAKAGE_MODE  # noqa: E402
from llm.c_graphrag.official_params import OFFICIAL_C_RETRIEVAL_VERSION  # noqa: E402
from llm.sample_split import load_frozen_ids  # noqa: E402

# Stage-2 candidates: DERIVED from the stage-1 report of record (llm/c_graphrag/stage2_candidates.py; Amendment 17:
# C v4 with answerable-only anchors, top_two = [0.5, 0.25]), never written by hand. EXPLORATORY_ALPHAS (alpha 0) are
# run the same way as a pre-declared reference and are reported separately; the selection rule never sees them.
from llm.c_graphrag.stage2_candidates import EXPLORATORY_ALPHAS, STAGE2_ALPHAS  # noqa: E402
N_DEV = 50
JUDGE_ID = "primary"
# The APPLIED_AT line printed by `01_data_cleaning/14_apply_logscore_trust.py
# --apply` after its verification passed (Amendment 6, Record). ISO 8601 WITH
# a UTC offset, e.g. "2026-10-14T09:30:00+07:00". Runs must start after this.
# None = Amendment 6 not applied yet: no report is produced.
TRUST_V3_APPLIED_AT = "2026-10-14T15:56:53+07:00"  # log 01_data_cleaning/logs/20261014T085255Z_apply_logscore_trust.log
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


def stage1_dev_ids() -> list[int]:
    """The stage-1 dev questions (positions 385-434), from the frozen list."""
    return [int(q) for q in load_frozen_ids("dev")["ids"]]


def load_runs(run_files: list[Path], alphas=STAGE2_ALPHAS, applied_at: str | None = None) -> dict:
    """{(alpha, grounding): {"path", "question_ids", "started_at"}} after all run checks."""
    runs, qid_sets = {}, []
    cutoff = trust_cutoff(applied_at)
    dev_ids = set(stage1_dev_ids())
    for path in run_files:
        path = Path(path)
        recs = _read_jsonl(path)
        if not recs:
            raise Stage2Error(f"{path.name}: empty run file")
        configs = {json.dumps(r.get("config"), sort_keys=True) for r in recs}
        if len(configs) != 1:
            raise Stage2Error(f"{path.name}: records carry {len(configs)} different configs")
        cfg = recs[0]["config"]
        problems = [k for k, want in (("condition", "C"), ("c_retrieval_version", OFFICIAL_C_RETRIEVAL_VERSION), ("sample_split", "dev"))
                    if cfg.get(k) != want]
        if problems:
            raise Stage2Error(f"{path.name}: wrong {problems} (config {cfg})")
        if cfg.get("expansion_require_answer") is not True:
            raise Stage2Error(f"{path.name}: expansion_require_answer is {cfg.get('expansion_require_answer')!r}, "
                              "not true (Amendment 9)")
        if cfg.get("leakage") != OFFICIAL_LEAKAGE_MODE:
            raise Stage2Error(f"{path.name}: leakage mode {cfg.get('leakage')!r}, not the official "
                              f"{OFFICIAL_LEAKAGE_MODE!r} (Amendment 15)")
        alpha = cfg.get("alpha")
        if alpha not in (*alphas, *EXPLORATORY_ALPHAS):
            raise Stage2Error(f"{path.name}: alpha {alpha} is not a stage-2 candidate {list(alphas)} "
                              f"(or an exploratory reference {list(EXPLORATORY_ALPHAS)})")
        key = (alpha, _grounding(cfg))
        if key in runs:
            raise Stage2Error(f"two runs for alpha={key[0]}, grounding={key[1]}: {runs[key]['path'].name}, {path.name}")
        qids = [int(r["question_id"]) for r in recs]
        if len(qids) != N_DEV or len(set(qids)) != N_DEV:
            raise Stage2Error(f"{path.name}: expected {N_DEV} distinct questions, got {len(set(qids))} of {len(qids)}")
        if set(qids) != dev_ids:
            raise Stage2Error(f"{path.name}: questions differ from the stage-1 dev list (positions 385-434): "
                              f"{len(set(qids) - dev_ids)} extra, {len(dev_ids - set(qids))} missing")
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
    extra = {k for k in runs if k[0] in EXPLORATORY_ALPHAS}          # exploratory reference runs: all-or-nothing per alpha
    for a in {k[0] for k in extra}:
        expected |= {(a, g) for g in ("on", "off")}
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

def counts(records: list[dict], category=None, severity=None) -> dict:
    category, severity = category or category_of, severity or severity_of
    cats = Counter(category(r) for r in records)
    h, n_ok, ab = cats[CATEGORY_HALLUCINATED], cats[CATEGORY_NOT_HALLUCINATED], cats[CATEGORY_ABSTAIN]
    claims = [c for r in records for c in (r.get("claims") or [])]
    n_unver = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "UNVERIFIABLE")
    sev = Counter(severity(r) for r in records if category(r) == CATEGORY_HALLUCINATED)
    return {
        "n": len(records), "n_abstain": ab, "n_not_hallucinated": n_ok, "n_hallucinated": h,
        "hallucination_count_excl_abstain": h, "n_excl_abstain": h + n_ok,
        "hallucination_count_incl_abstain": h + ab, "n_incl_abstain": h + n_ok + ab,
        "rate_excl_abstain": round(h / (h + n_ok), 4) if h + n_ok else None,
        "rate_incl_abstain": round((h + ab) / (h + n_ok + ab), 4) if records else None,
        "abstain_rate": round(ab / len(records), 4) if records else None,
        "severity": dict(sev), "unverifiable_claim_rate": round(n_unver / len(claims), 4) if claims else None,
    }


def stage2_metrics(judged: dict, alphas=STAGE2_ALPHAS, category=None, severity=None) -> tuple[dict, dict]:
    """(pooled metrics per alpha, per-(alpha, grounding) metrics)."""
    pooled = {a: counts([r for g in ("on", "off") for r in judged[(a, g)].values()], category, severity) for a in alphas}
    per_run = {f"{a}|{g}": counts(list(judged[(a, g)].values()), category, severity)
               for a in alphas for g in ("on", "off")}
    return pooled, per_run


# ---------------------------------------------------------------------
# Panel label (Amendment 14 draft (g)): the stage-2 selection uses the panel-v2 label, recomputed with the frozen
# 2-of-3 aggregation from the three per-judge records; every judge must have a usable record (else re-judge).
# ---------------------------------------------------------------------

PANEL_ID = "panel-v2"
PANEL_PROMPT_VERSION = "judge-v2.3"


def load_panel_judgments(judge_files: list[Path], runs: dict, prompt_version: str = PANEL_PROMPT_VERSION,
                         panel_id: str = PANEL_ID) -> tuple[dict, dict]:
    """({(alpha, grounding): {qid: panel record}}, {judge_id: {(alpha, grounding): {qid: judge record}}})."""
    from judge_panel_aggregate import aggregate_labels
    from judge_panel_config import PANELS
    from llm import archive_paths

    judge_ids = [j.judge_id for j in PANELS[panel_id]]
    by_path = {str(archive_paths.canonical(info["path"])): key for key, info in runs.items()}
    per_judge = {j: {key: {} for key in runs} for j in judge_ids}
    for jf in judge_files:
        for r in _read_jsonl(archive_paths.resolve(jf)):
            if r.get("panel_id") != panel_id or r.get("judge_id") not in per_judge:
                continue
            key = by_path.get(str(archive_paths.canonical(r.get("source_file", ""))))
            if key is None:
                continue
            if r.get("prompt_version") != prompt_version:
                raise Stage2Error(f"{jf}: record with prompt_version {r.get('prompt_version')!r} != {prompt_version!r}")
            qid = int(r["question_id"])
            if qid in per_judge[r["judge_id"]][key]:
                raise Stage2Error(f"duplicate {r['judge_id']} record for alpha={key[0]}, grounding={key[1]}, q{qid}")
            per_judge[r["judge_id"]][key][qid] = r
    panel = {key: {} for key in runs}
    for key, info in runs.items():
        for j in judge_ids:
            recs = per_judge[j][key]
            missing = info["question_ids"] - set(recs)
            bad = [q for q, r in recs.items() if is_unusable(r)]
            if missing or bad:
                raise Stage2Error(f"alpha={key[0]}, grounding={key[1]}: judge {j} has {len(missing)} missing and "
                                  f"{len(bad)} unusable records -- re-judge (the runner retries failures) before stage 2")
        for qid in sorted(info["question_ids"]):
            agg = aggregate_labels({j: per_judge[j][key][qid]["label"] for j in judge_ids}, panel_id)
            panel[key][qid] = {"question_id": qid, **agg}
    return panel, per_judge


def _panel_category(r: dict):
    from judge_panel_aggregate import panel_category
    return panel_category(r["panel_label"])


def _panel_severity(r: dict):
    return {"HALUSINASI_PENUH": "PENUH", "HALUSINASI_SEBAGIAN": "SEBAGIAN"}.get(r["panel_label"], "UNSPECIFIED")


SELECTION_RECORD_PATH = "docs/DECISION_C_SCORING.md#amendment-10-2026-10-18-trust-ablation-on-the-official-evaluation"


def apply_official_params(official: dict, commit: str, path: Path | None = None) -> Path:
    """--apply: writes the selected official parameters (alpha* / alpha_abl) to official_params.json, which
    unlocks Condition C on the test sample (official_params.check_c_test_run)."""
    from official_params import OFFICIAL_PARAMS_PATH, STATUS_PENDING
    path = Path(path or OFFICIAL_PARAMS_PATH)
    current = json.loads(path.read_text())
    if current.get("status") != STATUS_PENDING:
        raise Stage2Error(f"{path.name} status is {current.get('status')!r}, not {STATUS_PENDING!r}: already applied")
    out = {**official, "selection_record": {**official["selection_record"], "commit": commit}}
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def draft_official_params(selection: dict, alphas=STAGE2_ALPHAS) -> dict:
    """Amendment 10 schema: C-primary = the stage-2 winner, C-ablation =
    the other stage-2 candidate. selection_record.commit is left null in
    the draft: fill in the commit of the stage-2 report/runs when pasting."""
    if selection["keep_current"]:
        raise Stage2Error("keep-current rule (alpha = 1) cannot apply to the Amendment 10 candidates "
                          f"{list(alphas)}; check the selection before drafting official_params.")
    winner = selection["winner"]
    others = [a for a in alphas if a != winner]
    if winner not in alphas or len(others) != 1:
        raise Stage2Error(f"winner {winner!r} must be one of the two stage-2 candidates {list(alphas)}")
    return {"status": "selected", "c_retrieval_version": OFFICIAL_C_RETRIEVAL_VERSION, "alpha_primary": winner, "alpha_ablation": others[0],
            "selection_record": {"path": SELECTION_RECORD_PATH, "commit": None}, **OFFICIAL_DEFAULTS}


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def build_markdown(pooled, per_run, selection, runs, judge_version, judge_files, official) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# Stage 2 Report -- C retrieval v4 alpha selection", "",
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
              f"- Stage 1 of record: Amendment 17 re-run (C v4, answerable-only anchors; trustScore {TRUST_SCORE_VERSION}), "
              f"top_two = {list(STAGE2_ALPHAS)}.",
              f"- Stage 2: {judge_version}, pooled grounding on + off, n = "
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
    p.add_argument("--judge-version", required=True, help="Frozen judge version (Amendment 14 draft: judge-v2.3)")
    p.add_argument("--panel", choices=[PANEL_ID], default=None,
                   help="Select with the panel label (Amendment 14 (g)); --judge-files = judge_panel.py --out file(s)")
    p.add_argument("--apply", action="store_true",
                   help="Also write alpha*/alpha_abl to official_params.json (unlocks C on the test sample)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from llm import run_registry
    run_registry.add_cli_flag(p)
    args = p.parse_args()
    try:
        runs = load_runs([Path(f) for f in args.run_files])
        per_judge = None
        if args.panel:
            judged, per_judge = load_panel_judgments([Path(f) for f in args.judge_files], runs, args.judge_version,
                                                     args.panel)
        else:
            judged = load_judgments([Path(f) for f in args.judge_files], runs, args.judge_version)
        # Stage-2 inputs must also be registered as "dev-stage2" (results/run_registry.json; Amendment 14)
        run_registry.assert_reportable([*args.run_files, *args.judge_files], (run_registry.STATUS_DEV_STAGE2,),
                                       args.include_archived)
    except (Stage2Error, run_registry.RegistryError) as e:
        print(f"[STOP] {e}")
        return 1
    expl = sorted({a for a, _ in runs if a in EXPLORATORY_ALPHAS})
    every = (*STAGE2_ALPHAS, *expl)
    if args.panel:
        pooled, per_run = stage2_metrics(judged, alphas=STAGE2_ALPHAS, category=_panel_category, severity=_panel_severity)
        expl_pooled, expl_per_run = stage2_metrics(judged, alphas=expl, category=_panel_category, severity=_panel_severity)
        per_judge_metrics = {j: stage2_metrics(recs, alphas=every)[0] for j, recs in per_judge.items()}
        per_judge_per_run = {j: stage2_metrics(recs, alphas=every)[1] for j, recs in per_judge.items()}
    else:
        pooled, per_run = stage2_metrics(judged, alphas=STAGE2_ALPHAS)
        expl_pooled, expl_per_run = stage2_metrics(judged, alphas=expl)
        per_judge_metrics, per_judge_per_run = {}, {}
    context_items = {f"{a}|{g}": round(sum(len(r.get("retrieved_context") or []) for r in _read_jsonl(info["path"]))
                                       / N_DEV, 3) for (a, g), info in runs.items()}
    selection = select_stage2_winner(pooled)
    official = draft_official_params(selection)
    label = f"{args.judge_version}, {args.panel} PANEL label" if args.panel else f"{args.judge_version} (primary)"
    md = build_markdown(pooled, per_run, selection, runs, label, args.judge_files, official)
    if per_judge_metrics:
        md += "\n## Per judge (reported alongside; not the selection label)\n\n| judge | alpha | rate excl. ABSTAIN | ABSTAIN rate | rate incl. ABSTAIN |\n|---|---|---|---|---|\n"
        md += "\n".join(f"| {j} | {a} | {m['rate_excl_abstain']} ({m['hallucination_count_excl_abstain']}/{m['n_excl_abstain']}) | "
                        f"{m['abstain_rate']} | {m['rate_incl_abstain']} |" for j, pm in per_judge_metrics.items()
                        for a, m in pm.items()) + "\n"
    if expl:
        md += ("\n## EXPLORATORY reference (alpha " + ", ".join(map(str, expl)) + ") -- pre-declared, NOT eligible for "
               "selection; the rule above never saw it\n\n| alpha | n | hallucinated | not hallucinated | ABSTAIN | "
               "rate excl. ABSTAIN | rate incl. ABSTAIN |\n|---|---|---|---|---|---|---|\n")
        md += "\n".join(f"| {a} | {m['n']} | {m['n_hallucinated']} | {m['n_not_hallucinated']} | {m['n_abstain']} | "
                        f"{m['rate_excl_abstain']} ({m['hallucination_count_excl_abstain']}/{m['n_excl_abstain']}) | "
                        f"{m['rate_incl_abstain']} ({m['hallucination_count_incl_abstain']}/{m['n_incl_abstain']}) |"
                        for a, m in expl_pooled.items()) + "\n"
        md += ("\n| alpha | grounding | hallucinated | ABSTAIN | rate excl. ABSTAIN | severity |\n|---|---|---|---|---|---|\n"
               + "\n".join(f"| {k.split('|')[0]} | {k.split('|')[1]} | {m['n_hallucinated']} | {m['n_abstain']} | "
                           f"{m['rate_excl_abstain']} | {m['severity']} |" for k, m in expl_per_run.items()) + "\n")
    md += ("\n## Mean context items per answer (retrieval, per run)\n\n| alpha | grounding | items |\n|---|---|---|\n"
           + "\n".join(f"| {k.split('|')[0]} | {k.split('|')[1]} | {v} |" for k, v in sorted(context_items.items())) + "\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"stage2_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    Path(f"{stem}.md").write_text(md)
    Path(f"{stem}.json").write_text(json.dumps({
        "judge_version": args.judge_version, "pooled": {str(a): m for a, m in pooled.items()},
        "per_run": per_run, "selection": selection, "draft_official_params": official,
        "runs": {f"{a}|{g}": str(i["path"]) for (a, g), i in runs.items()},
        "judge_files": [str(j) for j in args.judge_files], "panel": args.panel,
        "per_judge_pooled": {j: {str(a): m for a, m in pm.items()} for j, pm in per_judge_metrics.items()},
        "tie_rule_used": selection.get("reasoning"),
        "exploratory": {"alphas": expl, "note": "pre-declared reference, not eligible for selection",
                        "pooled": {str(a): m for a, m in expl_pooled.items()}, "per_run": expl_per_run},
        "per_judge_per_run": per_judge_per_run, "context_items_per_answer": context_items,
    }, indent=2, default=str))
    if args.apply:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True).stdout.strip()
        try:
            print(f"[apply] {apply_official_params(official, commit)} <- alpha*={official['alpha_primary']}, "
                  f"alpha_abl={official['alpha_ablation']} (commit {commit[:7]})")
        except Stage2Error as e:
            print(f"[STOP] {e}")
            return 1
    print(md)
    print(f"Report: {stem}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
