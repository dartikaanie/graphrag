from fastapi import APIRouter, HTTPException, Query

from app.services import graph_service, history_service as svc

router = APIRouter(prefix="/api/history")


@router.get("")
def get_history(
    condition: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    if condition and condition.upper() not in ("A", "B", "C", "D"):
        raise HTTPException(status_code=400, detail="condition must be A, B, C, or D")
    items, total = svc.list_history(condition, date_from, date_to, page, page_size)
    return {
        "items": items,
        "meta": {"page": page, "page_size": page_size, "total": total, "total_pages": svc.total_pages(total, page_size)},
        "active_paths": {k: str(v) for k, v in svc.HISTORY_PATHS.items()},
    }


@router.get("/{history_id}")
def get_history_detail(history_id: str):
    detail = svc.get_history_detail(history_id)
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
