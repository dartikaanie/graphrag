"""Read-only end-to-end report of the official evaluation (Amendment 14 draft, plan step 9):
llm/evaluation/official_report.py over the run registry and stored outputs. No endpoint calls a model API or
writes anything. mock=true serves synthetic data for the walkthrough; every such response has meta.mock = true."""

import sys

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.archive import REPO_ROOT

if str(REPO_ROOT / "llm" / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "llm" / "evaluation"))

import official_report as orp  # noqa: E402

router = APIRouter(prefix="/api/report")


def _store(mock: bool, include_dev_stage2: bool):
    return orp.MockStore() if mock else orp.RegistryStore(include_dev_stage2)


@router.get("")
def get_report(mock: bool = Query(False), include_dev_stage2: bool = Query(False)):
    return orp.build(_store(mock, include_dev_stage2))


@router.get("/export.md", response_class=PlainTextResponse)
def export_markdown(mock: bool = Query(False), include_dev_stage2: bool = Query(False)):
    return orp.markdown(orp.build(_store(mock, include_dev_stage2)))


@router.get("/items")
def get_items(run_id: str, mock: bool = Query(False), include_dev_stage2: bool = Query(False)):
    items = orp.item_list(_store(mock, include_dev_stage2), run_id)
    if items is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "mock": mock, "items": items}


@router.get("/item")
def get_item(run_id: str, question_id: int, mock: bool = Query(False), include_dev_stage2: bool = Query(False)):
    item = orp.item_detail(_store(mock, include_dev_stage2), run_id, question_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    return item
