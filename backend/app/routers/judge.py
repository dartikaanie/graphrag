import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.schemas import JudgeRunCreateRequest, JudgeRunCreateResponse
from app.services import engine_service, history_service, judge_lookup_service, run_registry

router = APIRouter(prefix="/api/judge-runs")

_CONDITION_FOLDER = {"A": "a_pure_llm", "B": "b_rag", "C": "c_graphrag", "D": "d_lightrag"}


@router.post("", response_model=JudgeRunCreateResponse)
def create_judge_run(body: JudgeRunCreateRequest):
    condition = body.condition.upper()
    if condition not in ("A", "B", "C", "D"):
        raise HTTPException(status_code=400, detail="condition must be A, B, C, or D")
    if not Path(body.input_path).is_absolute() and not (engine_service.REPO_ROOT / body.input_path).exists():
        raise HTTPException(status_code=400, detail=f"input_path not found: {body.input_path}")
    if body.kappa_validation and not (body.secondary_judge_provider and body.secondary_judge_model):
        raise HTTPException(
            status_code=400,
            detail="secondary_judge_provider and secondary_judge_model are required when kappa_validation is true",
        )

    params = body.model_dump()
    run_id = run_registry.create_run("JUDGE", "batch", params)

    # Same pattern as routers/runs.py create_run(): a real background thread
    # (not asyncio.create_task) since judging is I/O-bound (LLM calls) and
    # can take minutes.
    thread = threading.Thread(target=engine_service.run_judge_batch_dashboard, args=(run_id, params), daemon=True)
    thread.start()

    return JudgeRunCreateResponse(run_id=run_id)


@router.get("/available-inputs")
def available_inputs(condition: str):
    """List result files available to judge for one condition -- read from
    that condition's own run_history.jsonl (output_path field), NOT a
    folder glob, so this only ever lists files a real run actually
    produced (never stray/leftover files sitting in a results/ folder)."""
    condition = condition.upper()
    if condition not in history_service.HISTORY_PATHS:
        raise HTTPException(status_code=400, detail="condition must be A, B, C, or D")

    records = history_service._load_condition_history(condition)
    items = []
    seen_paths: set[str] = set()
    for r in records:
        raw_path = r.get("output_path")
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = engine_service._anchor_output_path(path, _CONDITION_FOLDER[condition])
        resolved = str(path)
        if resolved in seen_paths or not path.exists():
            continue
        seen_paths.add(resolved)
        items.append({
            "path": resolved,
            "filename": path.name,
            "provider": r.get("provider"),
            "model": r.get("model"),
            "n_sample": r.get("n_sample_target"),
            "seed": r.get("seed"),
            "mode": "single" if "single" in path.name else "batch",
        })
    items.sort(key=lambda i: i["filename"])
    return {"items": items}


@router.get("/for-input")
def judge_runs_for_input(input_path: str):
    """Key join endpoint: has this exact result file already been judged,
    and with what config(s)? Used by RunJudgePage to show cached results +
    offer "re-judge (force)" instead of a plain Run button, and by History
    Detail (via judge_lookup_service directly, server-side) for the same
    lookup."""
    return {"items": judge_lookup_service.get_judge_evaluations_for_input(input_path)}


@router.get("/{run_id}")
def get_judge_run(run_id: str):
    run = run_registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Judge run not found")
    return run
