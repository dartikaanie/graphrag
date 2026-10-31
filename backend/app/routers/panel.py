"""Read-only dashboard endpoints for the 3-judge panel's official outputs
(hallucination panel + secondary metrics). See panel_results_service.py."""

from fastapi import APIRouter, HTTPException, Query

from app.services import panel_results_service as svc

router = APIRouter(prefix="/api/panel")


@router.get("/hallucination")
def get_hallucination_by_run():
    return {"runs": svc.hallucination_by_run()}


@router.get("/hallucination/{run_id}")
def get_hallucination_for_run(run_id: str):
    run = svc.hallucination_by_run().get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No official panel hallucination result for this run")
    return run


@router.get("/secondary")
def get_secondary_panel():
    return svc.secondary_panel()


@router.get("/evaluation")
def get_evaluation(ids: str = Query(..., description="Comma-separated history_id list")):
    """All panel metrics (hallucination, answer/context relevance) for the given
    runs: panel summary + per-judge breakdown + judge identity. A metric with no
    result for a run is null (rendered as 'not run', never hidden)."""
    run_ids = [i for i in ids.split(",") if i]
    if not run_ids:
        raise HTTPException(status_code=400, detail="ids must not be empty")
    return svc.evaluation_for_runs(run_ids)
