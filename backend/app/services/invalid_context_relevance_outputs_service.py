"""logs/invalid_context_relevance_outputs.jsonl -- context-relevance-v1
output *files* (under llm/evaluation/results/ctxrel_v1_*.jsonl) that are
known smoke-test/throwaway data, e.g. a manual --out path used for a
2-question smoke test. SAME append-only, hidden-by-default pattern as
invalid_judge_runs_service.py, but keyed on the output file's basename
(not a job_id) since a ctxrel-v1 smoke test is identified by which FILE
it wrote to, not by a judge_v1_run_history-style job record.

Used by engine_service._ctxrel_token_latency_stats()'s plan-table
estimation glob (over results/ctxrel_v1_*.jsonl) to skip a marked file --
dedup/resume/results-summary ALREADY only ever read the canonical
ctxrel_v1_<judge_id>_<prompt_version>.jsonl path (never a glob), so they
can never pick up a smoke-test output by construction; this service
closes the one remaining broad-glob path (token/latency estimation) so
smoke-test token counts never quietly bias a real plan-table estimate.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
INVALID_CTXREL_OUTPUTS_PATH = REPO_ROOT / "logs" / "invalid_context_relevance_outputs.jsonl"


def load_invalid_output_basenames() -> set[str]:
    if not INVALID_CTXREL_OUTPUTS_PATH.exists():
        return set()
    names: set[str] = set()
    with open(INVALID_CTXREL_OUTPUTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                names.add(json.loads(line)["output_file"])
            except Exception:
                continue
    return names


def mark_invalid(output_file: str, reason: str, **extra: Any) -> None:
    INVALID_CTXREL_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"output_file": output_file, "reason": reason, **extra}
    with open(INVALID_CTXREL_OUTPUTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
