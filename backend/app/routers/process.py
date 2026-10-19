"""Read-only API for the "Process & Parameters" page
(docs/agent_prompt_ui_process_page.md). No endpoint writes to Neo4j, starts
a run, or calls an LLM/judge API. Every response passes through
redact_obj (Rule 6); blinding rules live in app/process/guard.py (Rule 5)
and the log allowlist in app/process/logs.py (Rule 7).
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.process import live, logs, service
from app.process.redact import redact_obj

router = APIRouter(prefix="/api/process")


@router.get("/steps")
async def get_steps():
    return redact_obj(await run_in_threadpool(service.steps))


@router.get("/parameters")
async def get_parameters():
    return redact_obj(await run_in_threadpool(service.parameters))


@router.get("/decisions")
def get_decisions():
    return redact_obj(service.decisions())


@router.get("/runs")
def get_runs(include_archive: bool = Query(False, description="Also list pre-official archived runs (2026-10-19)")):
    return redact_obj(service.run_list(include_archive))


@router.get("/logs")
def get_logs(include_archive: bool = Query(False, description="Also list pre-official archived logs (2026-10-19)")):
    return redact_obj(service.log_list(include_archive))


@router.get("/logs/{root}/{name}")
def get_log(root: str, name: str, tail: int = Query(logs.DEFAULT_TAIL_LINES, ge=1, le=logs.MAX_TAIL_LINES)):
    try:
        return redact_obj(logs.read_tail(f"{root}/{name}", tail))
    except logs.LogAccessError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/live")
def get_live():
    return redact_obj(service.live_values())


@router.post("/live/refresh")
async def refresh_live():
    """Explicit, read-only Neo4j refresh (MATCH/RETURN only), cached for
    live.CACHE_TTL_SEC; returns "unavailable" instead of failing."""
    result = await run_in_threadpool(live.refresh)
    return redact_obj({"default": service.latest_verify(), "live": result, "queries": live.QUERIES})
