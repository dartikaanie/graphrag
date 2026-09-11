import asyncio
import json
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import RunCreateRequest, RunCreateResponse
from app.services import engine_service, graph_service, run_registry

router = APIRouter(prefix="/api/runs")


def _find_result_record(output_path: str | None, question_id: int) -> dict | None:
    if not output_path:
        return None
    try:
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("question_id") == question_id:
                    return record
    except FileNotFoundError:
        pass
    return None


@router.post("", response_model=RunCreateResponse)
def create_run(body: RunCreateRequest):
    condition = body.condition.upper()
    if condition not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="condition must be A, B, or C")
    if body.mode not in ("batch", "single"):
        raise HTTPException(status_code=400, detail="mode must be 'batch' or 'single'")
    if body.mode == "single" and not body.question_id:
        raise HTTPException(status_code=400, detail="question_id is required for single mode")

    params = body.model_dump()
    run_id = run_registry.create_run(condition, body.mode, params)

    # Runs are I/O-bound (LLM calls, DuckDB, Neo4j) and can take minutes for
    # n=384 -- a real background thread (not asyncio.create_task) so it
    # doesn't need to cooperatively yield inside sync/blocking library calls.
    thread = threading.Thread(target=engine_service.start_run, args=(run_id, condition, params), daemon=True)
    thread.start()

    return RunCreateResponse(run_id=run_id)


@router.get("/{run_id}")
def get_run(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/results")
def get_run_results(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run["results"]


@router.get("/{run_id}/results/{question_id}")
def get_run_result_detail(run_id: str, question_id: int):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    record = _find_result_record(run.get("output_path"), question_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found in this run's output")
    return record


@router.get("/{run_id}/results/{question_id}/graph")
def get_run_result_graph(run_id: str, question_id: int):
    """The actual retrieval path used for this one question (entity anchor
    -> graph traversal -> semantic expansion for Condition C), reconstructed
    from the provenance already persisted in its result record -- lets you
    verify what a run actually retrieved for a specific question, not just
    trust the aggregate metrics.
    """
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    record = _find_result_record(run.get("output_path"), question_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found in this run's output")
    return graph_service.get_run_result_graph(record)


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str):
    if not run_registry.request_cancel(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "cancel_requested"}


@router.get("/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE progress stream. Polls the in-memory/persisted run state rather
    than a pub/sub queue -- simple and reliable for a single-user local
    dashboard (PLAN_UI_UX.md §5.4), reconnect-safe since each event carries
    the full current state.
    """
    if not run_registry.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_stream():
        last_count = -1
        last_status = None
        while True:
            run = run_registry.get_run(run_id)
            if not run:
                break
            count = len(run["results"])
            if count != last_count or run["status"] != last_status:
                last_count, last_status = count, run["status"]
                yield f"data: {json.dumps(run, default=str)}\n\n"
            if run["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
