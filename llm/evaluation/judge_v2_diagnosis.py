"""
judge_v2_diagnosis.py
=====================================
2026-10-09 follow-up diagnosis (pilot diagnostic, n=90 per judge, test
sample -- NOT used to tune the judge). Diagnosis only: no API calls, no
prompt changes, no edits to judge code. Reads ONLY the reparsed
judge-v2 files (see reparse_judge_v2.py) plus the existing judge-v1
pilot output and the underlying generation files (for the two
manual-review items' candidate/reference text).

Two questions this answers:
  1. Why does judge-v2 (both judges) never produce HALUSINASI_SEBAGIAN
     on this pilot batch -- a code bug in derive_label_v2(), or judge
     behavior (the model essentially never emits a MINOR-severity
     CONTRADICTED/FABRICATED claim without an accompanying CORE one)?
  2. Quantify four specific leniency patterns per judge: severity on
     SUPPORTED claims, meta-statements extracted as claims, UNVERIFIABLE
     claims (and FAKTUAL items dominated by them), and claims asserting
     code validity marked SUPPORTED.

Writes: docs/JUDGE_V2_DIAGNOSIS.md

CARA PAKAI
----------------------------------------------------------------------------
    cd llm/evaluation
    python3 judge_v2_diagnosis.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = REPO_ROOT / "docs" / "JUDGE_V2_DIAGNOSIS.md"

V2_REPARSED_PATHS = {jid: RESULTS_DIR / f"jv2_dashboard_{jid}_judge-v2_reparsed.jsonl" for jid in ("primary", "secondary")}
V1_PATHS = {jid: RESULTS_DIR / f"jv1_dashboard_{jid}_judge-v1.jsonl" for jid in ("primary", "secondary")}

# Phrases matched for "meta-statement extracted as a claim" (section 3b).
# Kept as an explicit, visible list so the heuristic itself is auditable --
# NOT a general NLP classifier, just these literal (case-insensitive)
# substrings/patterns.
META_STATEMENT_PATTERNS = [
    r"retrieved sources? (?:do|does) not",
    r"sources? do(?:es)? not (?:cover|address|mention|discuss|provide)",
    r"not (?:specifically )?(?:covered|addressed) (?:by|in) the (?:sources|context|provided)",
    r"not addressed in the provided context",
    r"not covered in (?:detail )?(?:the|in) (?:retrieved )?(?:sources|context)",
    r"context does not (?:cover|address|mention)",
]
META_STATEMENT_RE = re.compile("|".join(META_STATEMENT_PATTERNS), re.IGNORECASE)

# Phrases matched for "claim asserting code validity" (section 3d).
CODE_VALIDITY_PATTERNS = [
    r"valid (?:rust )?(?:code|syntax)",
    r"(?:code|snippet) (?:is valid|works|compiles)",
]
CODE_VALIDITY_RE = re.compile("|".join(CODE_VALIDITY_PATTERNS), re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def non_error_records(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("parse_error")]


# ---------------------------------------------------------------------
# Section 2 -- SEBAGIAN reachability
# ---------------------------------------------------------------------

def verdict_severity_counts(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for r in non_error_records(records):
        for c in r.get("claims", []):
            counts[(c.get("verdict"), c.get("severity"))] += 1
    return counts


def items_with_minor_error_no_core(records: list[dict]) -> list[dict]:
    """Items that have >=1 CONTRADICTED/FABRICATED claim with severity
    MINOR and NO CONTRADICTED/FABRICATED claim with severity CORE --
    exactly the condition under which derive_label_v2() would return
    HALUSINASI_SEBAGIAN."""
    out = []
    for r in non_error_records(records):
        errors = [c for c in r.get("claims", []) if c.get("verdict") in ("CONTRADICTED", "FABRICATED")]
        has_core = any(c.get("severity") == "CORE" for c in errors)
        has_minor = any(c.get("severity") == "MINOR" for c in errors)
        if has_minor and not has_core:
            out.append(r)
    return out


# ---------------------------------------------------------------------
# Section 3 -- leniency patterns
# ---------------------------------------------------------------------

def supported_claims_with_severity(records: list[dict]) -> list[dict]:
    out = []
    for r in non_error_records(records):
        for c in r.get("claims", []):
            if c.get("verdict") == "SUPPORTED" and c.get("severity") is not None:
                out.append({"question_id": r["question_id"], "run_label": r.get("run_label"),
                             "severity": c.get("severity"), "claim": c.get("claim")})
    return out


def meta_statement_claims(records: list[dict]) -> list[dict]:
    out = []
    for r in non_error_records(records):
        for c in r.get("claims", []):
            text = c.get("claim") or ""
            if META_STATEMENT_RE.search(text):
                out.append({"question_id": r["question_id"], "run_label": r.get("run_label"),
                             "verdict": c.get("verdict"), "severity": c.get("severity"), "claim": text,
                             "label": r.get("label")})
    return out


def unverifiable_stats(records: list[dict]) -> dict:
    total = 0
    core = 0
    for r in non_error_records(records):
        for c in r.get("claims", []):
            if c.get("verdict") == "UNVERIFIABLE":
                total += 1
                if c.get("severity") == "CORE":
                    core += 1
    return {"total": total, "core": core}


def faktual_majority_unverifiable(records: list[dict], threshold: float = 0.5) -> list[dict]:
    out = []
    for r in non_error_records(records):
        if r.get("label") != "FAKTUAL":
            continue
        claims = r.get("claims", [])
        if not claims:
            continue
        n_unverif = sum(1 for c in claims if c.get("verdict") == "UNVERIFIABLE")
        if n_unverif / len(claims) >= threshold:
            out.append({"question_id": r["question_id"], "run_label": r.get("run_label"),
                         "n_unverifiable": n_unverif, "n_claims": len(claims)})
    return out


def code_validity_claims(records: list[dict]) -> list[dict]:
    out = []
    for r in non_error_records(records):
        for c in r.get("claims", []):
            text = c.get("claim") or ""
            if CODE_VALIDITY_RE.search(text) and c.get("verdict") == "SUPPORTED":
                out.append({"question_id": r["question_id"], "run_label": r.get("run_label"), "claim": text})
    return out


# ---------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------

def transition_totals(v1_records: list[dict], v2_records: list[dict], judge_id: str) -> dict:
    """Aggregate v1-HALUSINASI-* -> v2-{FAKTUAL,HALUSINASI_PENUH,ABSTAIN}
    counts for `judge_id`, used to state the corrected section-1 numbers."""
    v1_by_key = {(r["run_id"], r["question_id"]): r for r in v1_records if r.get("judge_id") == judge_id}
    v2_by_key = {(r["run_id"], r["question_id"]): r for r in v2_records if r.get("judge_id") == judge_id}
    common = set(v1_by_key) & set(v2_by_key)
    totals = Counter()
    n_v1_halusinasi = 0
    for key in common:
        v1_label = v1_by_key[key].get("label")
        if v1_label not in ("HALUSINASI_PENUH", "HALUSINASI_SEBAGIAN"):
            continue
        n_v1_halusinasi += 1
        v2_label = v2_by_key[key].get("label") if not v2_by_key[key].get("parse_error") else "PARSE_ERROR"
        if v2_label == "FAKTUAL":
            totals["to_faktual"] += 1
        elif v2_label == "HALUSINASI_PENUH":
            totals["to_penuh"] += 1
        elif v2_label == "ABSTAIN":
            totals["to_abstain"] += 1
        else:
            totals["to_other"] += 1
    return {"n_v1_halusinasi": n_v1_halusinasi, **totals}


def get_question_text(source_file: str, question_id: int) -> dict:
    with open(source_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("question_id") == question_id:
                return {"title": r.get("title"), "llm_answer": r.get("llm_answer"),
                         "ground_truth_answer": r.get("ground_truth_answer")}
    return {}


def main():
    v2 = {jid: load_jsonl(V2_REPARSED_PATHS[jid]) for jid in ("primary", "secondary")}
    v1 = {jid: load_jsonl(V1_PATHS[jid]) for jid in ("primary", "secondary")}
    v1_all = v1["primary"] + v1["secondary"]
    v2_all = v2["primary"] + v2["secondary"]

    lines = ["# Judge-v2 diagnosis: missing HALUSINASI_SEBAGIAN and leniency patterns", "",
             "**Diagnosis only -- no API calls, no prompt changes, no edits to judge code.** "
             "Computed on the reparsed judge-v2 files. Pilot batch (n=90 per judge) is part of "
             "the TEST sample -- nothing here is used to tune the judge; findings motivate a "
             "revision to be developed and validated on the dev set.", ""]

    lines.append("## 1. Corrected summary counts")
    lines.append("")
    lines.append("| judge | v1-HALUSINASI items | -> FAKTUAL | -> HALUSINASI_PENUH | -> ABSTAIN |")
    lines.append("|---|---|---|---|---|")
    for jid in ("primary", "secondary"):
        t = transition_totals(v1_all, v2_all, jid)
        lines.append(f"| {jid} | {t['n_v1_halusinasi']} | {t.get('to_faktual', 0)} | "
                      f"{t.get('to_penuh', 0)} | {t.get('to_abstain', 0)} |")
    lines.append("")
    lines.append(
        "Corrects the chat summary given alongside `docs/JUDGE_V2_LENIENCY_REVIEW.md` (which "
        "quoted only the HALUSINASI->FAKTUAL drill-down counts, 12 and 19, as if they were the "
        "full v1-HALUSINASI totals). The review doc's own matrices and section headers "
        "(\"primary (12 item(s))\", \"secondary (19 item(s))\") were already correct -- they "
        "label the drill-down specifically, not the totals -- so no edit to that file was needed."
    )
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"Written (section 1 only) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
