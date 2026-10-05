import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.services import batch_grouping_service, graph_service, history_service as svc

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from llm.manifest import MixedConfigHashError  # noqa: E402

router = APIRouter(prefix="/api/history")


@router.get("/batches")
def get_history_batches(show_superseded: bool = Query(False)):
    """Every run (across A/B/C/D), grouped into launch batches -- for the
    Hallucination Judge (judge-v1) page's run picker. Invalid runs are
    always excluded; superseded runs excluded unless show_superseded=True
    (same policy as GET /api/history). See batch_grouping_service.py."""
    items, _ = svc.list_history(None, None, None, page=1, page_size=100000, show_superseded=show_superseded)
    return {"batches": batch_grouping_service.group_into_batches(items)}


@router.get("")
def get_history(
    condition: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    show_superseded: bool = Query(False, description="Include runs superseded by a later re-run of the same config"),
):
    if condition and condition.upper() not in ("A", "B", "C", "D"):
        raise HTTPException(status_code=400, detail="condition must be A, B, C, or D")
    items, total = svc.list_history(condition, date_from, date_to, page, page_size, show_superseded)
    return {
        "items": items,
        "meta": {"page": page, "page_size": page_size, "total": total, "total_pages": svc.total_pages(total, page_size)},
        "active_paths": {k: str(v) for k, v in svc.HISTORY_PATHS.items()},
    }


@router.get("/compare/consistency")
def get_compare_consistency(ids: str = Query(..., description="Comma-separated history_id list, 2 or more")):
    history_ids = [i for i in ids.split(",") if i]
    if len(history_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 history ids are required (comma-separated in `ids`)")
    return svc.compute_sample_consistency(history_ids)


@router.get("/{history_id}")
def get_history_detail(history_id: str):
    try:
        detail = svc.get_history_detail(history_id)
    except MixedConfigHashError as e:
        # Output file mixes records from genuinely different configs/
        # prompt_versions under one filename -- never silently summarized.
        raise HTTPException(status_code=409, detail=str(e))
    if not detail:
        raise HTTPException(status_code=404, detail="History entry not found")
    return detail


@router.get("/{history_id}/results/{question_id}")
def get_history_result_detail(history_id: str, question_id: int):
    record = svc.get_history_result_detail(history_id, question_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found in this run's output")
    return record


@router.get("/{history_id}/results/{question_id}/graph")
def get_history_result_graph(history_id: str, question_id: int):
    record = svc.get_history_result_detail(history_id, question_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found in this run's output")
    return graph_service.get_run_result_graph(record)


@router.delete("/{history_id}")
def delete_history(history_id: str):
    if not svc.delete_history_record(history_id):
        raise HTTPException(status_code=404, detail="History entry not found")
    return {"status": "deleted", "history_id": history_id}
