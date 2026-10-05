"""logs/superseded_runs.jsonl -- run_history entries that were genuinely
VALID at the time (real generation happened, real answers), but have since
been replaced by a later re-run of the identical config under the new
config-hashed filename scheme (e.g. the 2026-10-05 pilot's 3 real
successes, C-trust-plain/C-uniform-plain/D-plain, superseded once their
configs were re-run with hashed filenames/manifests).

Deliberately SEPARATE from invalid_runs.jsonl: a superseded run was never
wrong, just outdated -- History/Compare hide it BY DEFAULT (so stale
numbers don't get compared against fresh ones by mistake) but a toggle
can bring it back into view, unlike an invalid run, which is always
hidden. run_history.jsonl itself is never rewritten/deleted either way.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERSEDED_RUNS_PATH = REPO_ROOT / "logs" / "superseded_runs.jsonl"


def load_superseded_run_ids() -> set[str]:
    if not SUPERSEDED_RUNS_PATH.exists():
        return set()
    ids: set[str] = set()
    with open(SUPERSEDED_RUNS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                continue
    return ids


def mark_superseded(run_id: str, condition: str, reason: str, **extra: Any) -> None:
    """Appends one record -- never overwrites/removes an existing one."""
    SUPERSEDED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "condition": condition, "reason": reason, **extra}
    with open(SUPERSEDED_RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
