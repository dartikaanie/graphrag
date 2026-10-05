"""Read-only, static run defaults the frontend needs -- distinct from
Settings (app/routers/settings.py), which is user-editable/persisted
config (API keys, Neo4j credentials, judge config). This is purely so the
frontend doesn't hard-code DEFAULT_OVERSAMPLE_POOL=1536 in several places
(RunConditionPage.tsx x4, RunAllConditionsPage.tsx) -- one backend
constant, one place the frontend reads it from.
"""

import sys
from pathlib import Path

from fastapi import APIRouter

from app.services import engine_service

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm.c_graphrag.c_graphrag import C_RETRIEVAL_VERSION  # noqa: E402
from llm.d_lightrag.d_lightrag import D_RETRIEVAL_VERSION  # noqa: E402
from llm.prompts import PROMPT_VERSION  # noqa: E402

router = APIRouter(prefix="/api/config")


@router.get("/defaults")
def get_defaults():
    return {
        "default_oversample_pool": engine_service.DEFAULT_OVERSAMPLE_POOL,
        "max_planned_n_sample": engine_service.MAX_PLANNED_N_SAMPLE,
        # Current generation-prompt version + each condition's own retrieval-
        # code version -- NOT the same track: C has only ever needed one
        # version bump (v1->v2, the determinism/tie-break fix); D needed two
        # (v1->v2 the EMBED_SIM edge-type fix, v2->v3 the LIMIT-before-
        # answer-join + determinism fix) -- D simply accumulated one more
        # D-specific bug fix than C did. The "C=v2 vs D=v3" difference is
        # INTENTIONAL, not a leftover/stale value -- see docs/README.md §8.
        "prompt_version": PROMPT_VERSION,
        "c_retrieval_version": C_RETRIEVAL_VERSION,
        "d_retrieval_version": D_RETRIEVAL_VERSION,
    }


@router.post("/release-faiss-cache")
def release_faiss_cache():
    """Frees the ~4GB shared FAISS index + embedding memmap Condition C/D
    load once and reuse (engine_service._faiss_cache) -- the next C/D run
    reloads it from disk. See Settings page: "Release cached FAISS index"."""
    return engine_service.release_faiss_cache()
