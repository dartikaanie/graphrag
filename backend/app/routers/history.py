from fastapi import APIRouter, HTTPException, Query

from app.services import history_service as svc

router = APIRouter(prefix="/api/history")


@router.get("")
def get_history(
    condition: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    if condition and condition.upper() not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="condition must be A, B, or C")
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
