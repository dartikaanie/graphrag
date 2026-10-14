"""
Blinding / held-out guard for the Process page (Rule 5). The single place
that decides what the page may NOT show:

  * anything under the annotation, calibration, annotation-key and stage-2
    output locations (Amendment 4: calibration is in progress, annotator 1
    must stay blind; the confirmation set comes from stage-2 outputs);
  * item content (questions, answers, contexts, judge outputs/labels) of any
    run, in particular the official test sample (positions 1-384);
  * judge labels of calibration items and the annotation key.

For these the page gets aggregate status only (`calibration_status`,
`stage2_status`). Every other module asks this one: `is_blocked_path` before
reading a file, `check_run_metadata` before returning a run row.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.process.sources import REPO_ROOT

TEST_SAMPLE_POSITIONS = (1, 384)  # 1-based, docs/DECISION_C_SCORING.md "Development set"

ANNOTATION_DIR = REPO_ROOT / "llm" / "evaluation" / "annotation"
CALIBRATION_DIR = REPO_ROOT / "llm" / "evaluation" / "calibration"
ANNOTATION_KEY_DIR = Path.home() / "graphrag_annotation_keys"  # annotation_tool.DEFAULT_KEY_DIR
STAGE2_DIR = REPO_ROOT / "llm" / "c_graphrag" / "results" / "stage2"  # stage2_report.OUT_DIR

BLOCKED_DIRS = (ANNOTATION_DIR, CALIBRATION_DIR, ANNOTATION_KEY_DIR, STAGE2_DIR)

# Filename fragments that mark calibration/confirmation material wherever it
# lives (e.g. a judge output written next to other results).
BLOCKED_NAME_FRAGMENTS = ("calibration", "confirmation", "annotation", "annotator", "_key", "adjudicat", "stage2")

# The only fields a run row may carry -- metadata, never content.
RUN_METADATA_FIELDS = frozenset({
    "run_id", "file", "condition", "split", "alpha", "grounding", "fusion_mode", "c_retrieval_version",
    "trust_score_version", "config_hash", "prompt_version", "judge_version", "model", "n_sample",
    "status", "started_at_utc", "finished_at_utc", "commit", "git_dirty", "items_succeeded", "run_kind",
})


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def is_blocked_path(path: Path) -> bool:
    """True for any file the Process page must never read or list."""
    if any(_within(path, d) for d in BLOCKED_DIRS):
        return True
    name = path.name.lower()
    return any(fragment in name for fragment in BLOCKED_NAME_FRAGMENTS)


class GuardViolation(Exception):
    pass


def check_run_metadata(row: dict) -> dict:
    """Raise if a run row carries anything but whitelisted metadata."""
    extra = set(row) - RUN_METADATA_FIELDS
    if extra:
        raise GuardViolation(f"run row has non-metadata fields: {sorted(extra)}")
    return row


# ---------------------------------------------------------------- aggregates

def _judge_frozen(decisions_text: str) -> bool:
    """Amendment 4 Record: 'Judge version frozen: [ ]' stays unchecked until
    the judge is frozen (author fills in the version)."""
    for line in decisions_text.splitlines():
        if "Judge version frozen:" in line:
            return "[ ]" not in line
    return False


def calibration_status(decisions_text: str) -> dict:
    """Aggregate status of the Amendment 4 judge calibration. Reads only:
    whether the annotation export exists and its item counts/creation time
    (export_manifest.json), whether an adjudicated reference exists (file
    existence only), and the Amendment 4 Record checkbox. Never the
    annotation files, the key, or any judge label."""
    export = ANNOTATION_DIR / "calibration" / "export_manifest.json"
    exported = None
    if export.is_file():
        try:
            meta = json.loads(export.read_text(encoding="utf-8"))
            exported = {
                "created_at_utc": meta.get("created_at_utc"),
                "n_items": meta.get("n_items"),
                "n_items_annotator2": meta.get("n_items_annotator2"),
            }
        except (OSError, json.JSONDecodeError):
            exported = None
    adjudicated = (ANNOTATION_DIR / "calibration" / "reference_adjudicated.json").is_file()
    frozen = _judge_frozen(decisions_text)
    if frozen:
        status = "done"
    elif adjudicated:
        status = "adjudicated; judge not frozen"
    elif exported:
        status = "in progress"
    else:
        status = "not started"
    return {
        "status": status,
        "annotation_files_created": exported,
        "adjudicated_reference_exists": adjudicated,
        "judge_frozen": frozen,
        "source": "llm/evaluation/annotation/calibration/export_manifest.json (counts only); "
                  "docs/DECISION_C_SCORING.md Amendment 4 Record",
    }


def stage2_status(judge_frozen: bool) -> dict:
    """Stage-2 outputs, once they exist, are hidden until the confirmation
    set is annotated: only existence is reported, never names or counts."""
    exists = STAGE2_DIR.is_dir() and any(p.is_file() for p in STAGE2_DIR.iterdir())
    if exists:
        status = "outputs exist (hidden until the confirmation set is annotated)"
    elif not judge_frozen:
        status = "waiting for judge freeze (Amendment 4)"
    else:
        status = "not started"
    return {"status": status, "outputs_exist": exists}
