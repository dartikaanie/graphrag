"""logs/invalid_judge_runs.jsonl -- judge-v1 job entries (from
llm/evaluation/logs/judge_v1_run_history.jsonl) that are known-bad, e.g.
pre-dashboard smoke-test jobs (old pre-grounding run_labels, /tmp output
paths). SAME append-only pattern as invalid_runs_service.py/
superseded_runs_service.py, but keyed on judge-v1 JOB ids (see
judge_v1_lookup_service.py's _job_id()), a separate id space from
generation run_history's history_id.

Hidden by default, unconditionally (no toggle) -- unlike superseded runs,
these never represent real study data worth looking at again.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
INVALID_JUDGE_RUNS_PATH = REPO_ROOT / "logs" / "invalid_judge_runs.jsonl"


def load_invalid_judge_run_ids() -> set[str]:
    if not INVALID_JUDGE_RUNS_PATH.exists():
        return set()
    ids: set[str] = set()
    with open(INVALID_JUDGE_RUNS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["job_id"])
            except Exception:
                continue
    return ids


def mark_invalid(job_id: str, reason: str, **extra: Any) -> None:
    INVALID_JUDGE_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"job_id": job_id, "reason": reason, **extra}
    with open(INVALID_JUDGE_RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
