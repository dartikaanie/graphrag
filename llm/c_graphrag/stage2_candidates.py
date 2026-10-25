"""The alpha stage-2 candidates, DERIVED from the stage-1 report of record (no hand-written values).

Stage 1 (docs/DECISION_C_SCORING.md, Selection procedure) ranks the alphas; its top two go to stage 2. The report of
record is pinned by file name so a later report cannot change the candidates silently. Amendment 17: the report made
with C retrieval v4 (answerable-only anchors), top_two = [0.5, 0.25] at the time of writing; read from the file.

EXPLORATORY_ALPHAS are run the same way as a reference (pre-declared before stage 2) and are NEVER an input of the
selection rule. Light module (json only): imported by stage2_report, run_registry and frozen_parameters.
"""

import json
from pathlib import Path

STAGE1_REPORT_OF_RECORD = "stage1_anchors_report_20261025T165734Z.json"
STAGE1_REPORT_PATH = Path(__file__).resolve().parent / "results" / "anchors_check" / STAGE1_REPORT_OF_RECORD
EXPLORATORY_ALPHAS = (0.0,)


class Stage1ReportError(RuntimeError):
    pass


def stage1_top_two(path=STAGE1_REPORT_PATH) -> tuple:
    """The two best alphas of the stage-1 report, best first."""
    from llm import archive_paths

    path = archive_paths.resolve(path)
    if not Path(path).exists():
        raise Stage1ReportError(f"stage-1 report of record not found: {path}")
    sel = json.loads(Path(path).read_text())["selection"]
    if sel.get("keep_current") or len(sel.get("top_two", [])) != 2:
        raise Stage1ReportError(f"stage-1 report has no usable top_two: {sel}")
    return tuple(float(a) for a in sel["top_two"])


STAGE2_ALPHAS = stage1_top_two()
