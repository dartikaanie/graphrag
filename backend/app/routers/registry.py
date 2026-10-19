"""Read-only view of results/run_registry.json (llm/run_registry.py). Official
runs by default; dev-stage2 and archived runs only when asked for explicitly
(the dashboard's "show archive" toggle). No endpoint writes or calls an API."""

from fastapi import APIRouter, Query

from app.archive import REPO_ROOT  # noqa: F401  (puts the repo root on sys.path)
from llm import run_registry

router = APIRouter(prefix="/api/registry")


@router.get("")
def get_registry(
    include_dev_stage2: bool = Query(False, description="Also list alpha stage-2 dev runs"),
    include_archive: bool = Query(False, description="Also list pre-official archived runs (2026-10-19)"),
):
    reg = run_registry.load()
    statuses = (run_registry.STATUS_OFFICIAL,) + ((run_registry.STATUS_DEV_STAGE2,) if include_dev_stage2 else ())
    rows = run_registry.runs(statuses, include_archived=include_archive, reg=reg)
    return {
        "statuses": list(statuses) + ([run_registry.STATUS_ARCHIVED] if include_archive else []),
        "counts": {s: sum(1 for r in reg["runs"] if r["status"] == s) for s in run_registry.STATUSES},
        "budget": reg.get("budget"),
        "deepinfra_spent_usd": run_registry.deepinfra_spent(reg),
        "deepinfra_remaining_usd": run_registry.deepinfra_remaining(reg),
        "runs": sorted(rows, key=lambda r: r.get("created_at") or "", reverse=True),
    }
