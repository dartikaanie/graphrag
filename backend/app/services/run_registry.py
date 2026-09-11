"""In-memory run registry, persisted to JSON so GET /api/runs/{id} survives
a page refresh (PLAN_UI_UX.md §8 point 4). This does NOT resume execution
after a backend restart -- for a local, single-user thesis dashboard the
plan explicitly calls that out as unnecessary ("an in-memory registry plus
writing progress to a file is sufficient").
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).resolve().parents[3] / "backend" / ".runs"

_lock = threading.Lock()
_REGISTRY: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist(run_id: str) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / f"{run_id}.json", "w") as f:
        json.dump(_REGISTRY[run_id], f, default=str)


def create_run(condition: str, mode: str, params: dict[str, Any]) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _REGISTRY[run_id] = {
            "run_id": run_id,
            "condition": condition,
            "mode": mode,
            "status": "pending",
            "params": params,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "progress": {"current": 0, "total": 1 if mode == "single" else params.get("n_sample", 1)},
            "results": [],
            "summary": None,
            "error": None,
            "output_path": None,
        }
        _persist(run_id)
    return run_id


def update_run(run_id: str, **fields: Any) -> None:
    with _lock:
        if run_id not in _REGISTRY:
            return
        _REGISTRY[run_id].update(fields)
        _persist(run_id)


def append_result(run_id: str, result: dict[str, Any]) -> None:
    with _lock:
        if run_id not in _REGISTRY:
            return
        _REGISTRY[run_id]["results"].append(result)
        # Preserve whatever `total` engine_service already set (the real
        # sample_df size) -- don't recompute from params.n_sample, which is
        # meaningless for single-question mode and previously clobbered the
        # correct total on every appended result.
        existing_total = _REGISTRY[run_id]["progress"].get("total", _REGISTRY[run_id]["params"].get("n_sample", 1))
        _REGISTRY[run_id]["progress"] = {
            "current": len(_REGISTRY[run_id]["results"]),
            "total": existing_total,
        }
        _persist(run_id)


def get_run(run_id: str) -> dict[str, Any] | None:
    with _lock:
        if run_id in _REGISTRY:
            return dict(_REGISTRY[run_id])
    # Fall back to disk (e.g. after a backend restart) so a stale run_id
    # still resolves to its last known state instead of a bare 404.
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        with _lock:
            _REGISTRY[run_id] = data
        return data
    return None


def request_cancel(run_id: str) -> bool:
    with _lock:
        if run_id not in _REGISTRY:
            return False
        _REGISTRY[run_id]["cancel_requested"] = True
        _persist(run_id)
        return True


def is_cancelled(run_id: str) -> bool:
    with _lock:
        return bool(_REGISTRY.get(run_id, {}).get("cancel_requested"))
