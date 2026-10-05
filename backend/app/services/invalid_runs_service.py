"""logs/invalid_runs.jsonl -- a SEPARATE, append-only record of run_history
entries that are known-bad (e.g. the 2026-10-05 pilot's 6 stale-file resume
collisions: zero real generation happened, but append_run_history() still
wrote a normal-looking "no_results" entry for them). run_history.jsonl
itself is NEVER rewritten/deleted -- this file sits alongside it and the
dashboard's History/Compare views filter out any run_id listed here, so a
known-bad entry can never silently pass as real data without erasing the
audit trail of what actually happened.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
INVALID_RUNS_PATH = REPO_ROOT / "logs" / "invalid_runs.jsonl"


def load_invalid_run_ids() -> set[str]:
    if not INVALID_RUNS_PATH.exists():
        return set()
    ids: set[str] = set()
    with open(INVALID_RUNS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                continue
    return ids


def mark_invalid(run_id: str, condition: str, reason: str, **extra: Any) -> None:
    """Appends one record -- never overwrites/removes an existing one (if
    called twice for the same run_id, both records persist; readers only
    need the run_id to be PRESENT at least once to hide it)."""
    INVALID_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "condition": condition, "reason": reason, **extra}
    with open(INVALID_RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
