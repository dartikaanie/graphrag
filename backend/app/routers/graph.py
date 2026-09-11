from fastapi import APIRouter, HTTPException, Query

from app.services import graph_service as svc

router = APIRouter(prefix="/api/graph")


@router.get("/partial")
def get_partial_graph(limit: int = Query(100, ge=1, le=500)):
    return svc.get_partial_graph(limit)


@router.get("/node/{node_type}/{node_id}")
def get_node_subgraph(node_type: str, node_id: str, hops: int = Query(2, ge=1, le=2)):
    try:
        result = svc.get_node_subgraph(node_type, node_id, hops)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"{node_type} '{node_id}' not found")
    return result
