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
    """Returns a dict with: run_id, condition, run_label, fusion_mode,
    retrieval_version, generator_prompt_version, source_file.

    run_label is f"{condition}-{fusion_mode}" when fusion_mode is set
    (e.g. "C-uniform", "C-trust_weighted"), otherwise just the condition
    letter (e.g. "A", "B", "D" -- none of these currently have a
    fusion_mode/variant concept).

    Raises RunMetadataNotFoundError if no run_history.jsonl entry's
    output_path matches this file -- callers must stop and tell the user
    rather than silently falling back to a guess.
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

    fusion_mode = latest.get("fusion_mode")
    if condition == "C":
        retrieval_version = latest.get("c_retrieval_version")
    elif condition == "D":
        retrieval_version = latest.get("d_retrieval_version")
    else:
        retrieval_version = None

    run_label = f"{condition}-{fusion_mode}" if fusion_mode else condition

    return {
        "run_id": _history_id(condition, latest),
        "condition": condition,
        "run_label": run_label,
        "fusion_mode": fusion_mode,
        "retrieval_version": retrieval_version,
        "generator_prompt_version": latest.get("prompt_version"),
        "source_file": str(result_path),
    }
