"""Read-only, static run defaults the frontend needs -- distinct from
Settings (app/routers/settings.py), which is user-editable/persisted
config (API keys, Neo4j credentials, judge config). This is purely so the
frontend doesn't hard-code DEFAULT_OVERSAMPLE_POOL=1536 in several places
(RunConditionPage.tsx x4, RunAllConditionsPage.tsx) -- one backend
constant, one place the frontend reads it from.
"""

from fastapi import APIRouter

from app.services import engine_service

router = APIRouter(prefix="/api/config")


@router.get("/defaults")
def get_defaults():
    return {
        "default_oversample_pool": engine_service.DEFAULT_OVERSAMPLE_POOL,
        "max_planned_n_sample": engine_service.MAX_PLANNED_N_SAMPLE,
    }


@router.post("/release-faiss-cache")
def release_faiss_cache():
    """Frees the ~4GB shared FAISS index + embedding memmap Condition C/D
    load once and reuse (engine_service._faiss_cache) -- the next C/D run
    reloads it from disk. See Settings page: "Release cached FAISS index"."""
    return engine_service.release_faiss_cache()
