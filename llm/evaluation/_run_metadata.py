"""
_run_metadata.py
=====================================
Resolves a condition result `.jsonl` file (e.g. condition_c_openai_gpt-
4o-mini_n10_seed42_uniform.jsonl) to its AUTHORITATIVE run parameters --
fusion_mode, c_retrieval_version/d_retrieval_version, and a dashboard-
style run_id -- by reading the SAME source the dashboard's Run Comparison
report reads: each condition's `logs/run_history.jsonl`, joined on
`output_path`. This is deliberately NOT inferred from the result
filename alone: filename prefix only tells us the condition letter, not
which fusion_mode variant produced a given Condition C file, which
matters for the thesis's ablation (uniform vs trust_weighted) and must
never be silently conflated.

run_id FORMULA IS A DELIBERATE, MANUALLY-SYNCED DUPLICATE of
backend/app/services/history_service.py::_history_id() -- NOT an import
from backend/. The dependency direction in this repo is backend -> llm/
(engine_service.py imports FROM llm/, never the reverse); importing
backend.app.services.history_service from here would also drag in its
own import of judge_lookup_service and, transitively, the FastAPI app
context, which llm/evaluation/ CLI scripts must stay free of (they run
standalone, no backend process required). If that formula ever changes
in history_service.py, _history_id() here must be updated to match, or
run_ids computed here stop lining up with the ones shown on the
dashboard's History page.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATHS: dict[str, Path] = {
    "A": REPO_ROOT / "llm" / "a_pure_llm" / "logs" / "run_history.jsonl",
    "B": REPO_ROOT / "llm" / "b_rag" / "logs" / "run_history.jsonl",
    "C": REPO_ROOT / "llm" / "c_graphrag" / "logs" / "run_history.jsonl",
    "D": REPO_ROOT / "llm" / "d_lightrag" / "logs" / "run_history.jsonl",
}

CONDITION_PREFIXES = {
    "condition_a_": "A",
    "condition_b_": "B",
    "condition_c_": "C",
    "condition_d_": "D",
}


class RunMetadataNotFoundError(Exception):
    """Raised when a result file's run parameters cannot be reliably
    resolved from run_history.jsonl -- callers must stop and surface this,
    never guess/fall back to filename-only inference for fusion_mode."""


# Short names used ONLY inside run_label (e.g. "C-uniform-grounded"), per
# docs/GROUNDING_FACTOR_UI.md's factorial table -- fusion_mode itself is
# ALWAYS stored/returned in full ("uniform"/"trust_weighted"); this is a
# display-only shortening.
_FUSION_MODE_SHORT = {"uniform": "uniform", "trust_weighted": "trust"}

# B's transitional hack (now removed from the generator scripts, see
# docs/GROUNDING_FACTOR_UI.md "run_label design conflict"): for a short
# window, B wrote its grounding state INTO fusion_mode (values "plain"/
# "grounded") because it had no dedicated field yet. Any run_history
# entry written during that window must still resolve correctly -- never
# rewritten, just read correctly here.
_B_LEGACY_FUSION_MODE_AS_GROUNDING = {"plain": "off", "grounded": "on"}

# Historical default grounding behavior BEFORE the "grounding"/
# "require_grounding" fields existed at all (used only when a record has
# NEITHER field, e.g. a run from before any of this -- including before
# the legacy B hack above). B never grounded by default; C/D always did.
_HISTORICAL_DEFAULT_GROUNDING = {"A": None, "B": "off", "C": "on", "D": "on"}


def derive_run_label(condition: str, record: dict) -> dict:
    """Single source of truth for the 3-axis run_label (condition +
    fusion_mode short name, for C + grounding on/off) -- used by BOTH
    this module and backend/app/services/history_service.py (manually
    synced, same pattern as _history_id() -- see module docstring).

    Returns {"run_label", "fusion_mode" (full value, never shortened),
    "grounding" ("on"/"off"/None for A), "grounding_inferred" (bool --
    True if "grounding"/"require_grounding" was missing/ambiguous and the
    value had to be inferred from older data or a historical default,
    rather than read directly)}.

    Resolution order for `grounding`:
      1. record["grounding"] ("on"/"off") if present -- the current,
         dedicated field. Not inferred.
      2. record["require_grounding"] (bool) if present. Not inferred.
      3. Condition B ONLY: record["fusion_mode"] in ("plain", "grounded")
         -- the removed transitional hack. Inferred (the SIGNAL is
         accurate, just not from the canonical field).
      4. Historical default per condition (B off, C/D on), if
         require_citation/grounding concept applies at all. Inferred.
    A never has a grounding axis (`grounding` is None, label is just "A").
    """
    if condition == "A":
        return {"run_label": "A", "fusion_mode": None, "grounding": None, "grounding_inferred": False, "is_dev": False}

    fusion_mode = record.get("fusion_mode")
    grounding_inferred = False

    if record.get("grounding") in ("on", "off"):
        grounding = record["grounding"]
    elif record.get("require_grounding") is not None:
        grounding = "on" if record["require_grounding"] else "off"
    elif condition == "B" and fusion_mode in _B_LEGACY_FUSION_MODE_AS_GROUNDING:
        grounding = _B_LEGACY_FUSION_MODE_AS_GROUNDING[fusion_mode]
        fusion_mode = None  # that value was never a real fusion_mode -- don't leak it as one
        grounding_inferred = True
    else:
        grounding = _HISTORICAL_DEFAULT_GROUNDING.get(condition)
        grounding_inferred = grounding is not None

    parts = [condition]
    if condition == "C" and fusion_mode:
        parts.append(_FUSION_MODE_SHORT.get(fusion_mode, fusion_mode))
    if condition == "C" and record.get("c_retrieval_version") == "v3" and record.get("alpha") is not None:
        parts.append(f"a={record['alpha']}")
    if grounding is not None:
        parts.append("grounded" if grounding == "on" else "plain")

    is_dev = record.get("sample_split") == "dev"
    label = "-".join(parts)
    if is_dev:
        label += " [dev]"

    return {
        "run_label": label,
        "fusion_mode": fusion_mode,
        "grounding": grounding,
        "grounding_inferred": grounding_inferred,
        "is_dev": is_dev,
    }


def infer_condition_from_filename(path) -> str:
    name = Path(path).name
    for prefix, condition in CONDITION_PREFIXES.items():
        if name.startswith(prefix):
            return condition
    raise RunMetadataNotFoundError(
        f"Cannot infer condition from filename '{name}' -- expected it to start with one of "
        f"{list(CONDITION_PREFIXES)}"
    )


def _history_id(condition: str, record: dict) -> str:
    """EXACT SAME formula as history_service.py::_history_id() -- see
    module docstring. Deterministic: same (condition, run_started_at,
    output_path) always hashes to the same id."""
    basis = f"{condition}|{record.get('run_started_at')}|{record.get('output_path')}"
    return f"{condition}-{hashlib.md5(basis.encode()).hexdigest()[:10]}"


def _load_history_records(history_path: Path) -> list[dict]:
    records = []
    if not history_path.exists():
        return records
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def resolve_run_metadata(result_file_path: str) -> dict:
    """Returns a dict with: run_id, condition, run_label, fusion_mode
    (full value, e.g. "trust_weighted", never shortened), grounding
    ("on"/"off"/None), grounding_inferred (bool), retrieval_version,
    generator_prompt_version, source_file, batch_id (may be None for
    older runs launched before batch_id was recorded).

    run_label is the 3-axis factorial label from derive_run_label() --
    e.g. "A", "B-plain", "B-grounded", "C-uniform-grounded",
    "C-trust-plain", "D-grounded" -- see docs/GROUNDING_FACTOR_UI.md.

    Raises RunMetadataNotFoundError if no run_history.jsonl entry's
    output_path matches this file -- callers must stop and tell the user
    rather than silently falling back to a guess. (grounding itself CAN
    be inferred when missing -- that's a documented, bounded fallback,
    not the same as this error, which fires when there's NO entry at all.)
    """
    result_path = Path(result_file_path).resolve()
    condition = infer_condition_from_filename(result_path)

    history_path = HISTORY_PATHS[condition]
    all_records = _load_history_records(history_path)
    if not all_records:
        raise RunMetadataNotFoundError(
            f"{history_path} does not exist or is empty -- cannot resolve run parameters for "
            f"'{result_file_path}'."
        )

    matches = []
    for record in all_records:
        output_path = record.get("output_path")
        if output_path and Path(output_path).resolve() == result_path:
            matches.append(record)

    if not matches:
        raise RunMetadataNotFoundError(
            f"No entry in {history_path} has output_path == '{result_path}'. This file's "
            f"fusion_mode/retrieval_version cannot be reliably determined -- refusing to guess. "
            f"Check that the file wasn't renamed/moved after its run_history entry was written."
        )

    # Re-runs can overwrite the same output_path more than once -- the
    # MOST RECENT entry describes what's actually on disk right now.
    latest = max(matches, key=lambda r: r.get("run_started_at") or "")

    if condition == "C":
        retrieval_version = latest.get("c_retrieval_version")
    elif condition == "D":
        retrieval_version = latest.get("d_retrieval_version")
    else:
        retrieval_version = None

    label_info = derive_run_label(condition, latest)

    return {
        "run_id": _history_id(condition, latest),
        "condition": condition,
        "run_label": label_info["run_label"],
        "fusion_mode": label_info["fusion_mode"],
        "grounding": label_info["grounding"],
        "grounding_inferred": label_info["grounding_inferred"],
        "retrieval_version": retrieval_version,
        "generator_prompt_version": latest.get("prompt_version"),
        "source_file": str(result_path),
        "batch_id": latest.get("batch_id"),
    }
