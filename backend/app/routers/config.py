"""Read-only, static run defaults the frontend needs -- distinct from
Settings (app/routers/settings.py), which is user-editable/persisted
config (API keys, Neo4j credentials, judge config). This is purely so the
frontend doesn't hard-code DEFAULT_OVERSAMPLE_POOL=1536 in several places
(RunConditionPage.tsx x4, RunAllConditionsPage.tsx) -- one backend
constant, one place the frontend reads it from.
"""

import json
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

OFFICIAL_C_PARAMS_PATH = REPO_ROOT / "llm" / "c_graphrag" / "official_params.json"


def _official_c_params() -> dict | None:
    """The Factorial Batch page's "Official (n=384)" preset locks
    Condition C's v3 parameters to whatever docs/DECISION_C_SCORING.md's
    machine-readable block says (kept in sync by
    llm/c_graphrag/tests/test_official_params_sync.py) -- read fresh on
    every request rather than cached, since this file changes exactly
    once (when the decision record's Outcome section is filled in)."""
    if not OFFICIAL_C_PARAMS_PATH.exists():
        return None
    return json.loads(OFFICIAL_C_PARAMS_PATH.read_text())


_LOCKED_PARAM_KEYS = (
    "c_retrieval_version", "alpha", "sample_split", "max_hops", "edge_types",
    "use_author_trust", "accepted_only",
)


def check_official_c_params_lock(official_params: dict | None, requested: dict) -> dict:
    """Shared launch-refusal logic for the Factorial Batch page's
    "Official (n=384)" preset -- the single place the "is this allowed to
    launch" decision is made, so the frontend's warning and any future
    server-side enforcement can't silently drift apart.

    Refuses whenever `official_params["status"] == "pending_selection"`
    REGARDLESS of `c_retrieval_version` (even a plain v2 run is refused
    while the decision is pending -- the Official preset as a whole is
    locked, not just the v3 knobs), and otherwise refuses when any
    locked key in `requested` differs from `official_params`.

    Returns {"allowed": bool, "message": str | None, "mismatched_keys": list[str]}.
    `official_params=None` (file missing) always allows -- there's
    nothing to lock against yet, same as before this feature existed.
    """
    if official_params is None:
        return {"allowed": True, "message": None, "mismatched_keys": []}

    if official_params.get("status") == "pending_selection":
        return {
            "allowed": False,
            "message": (
                "Official preset is locked pending the C retrieval v3 decision -- "
                "see docs/DECISION_C_SCORING.md's Outcome section."
            ),
            "mismatched_keys": [],
        }

    mismatched = []
    for key in _LOCKED_PARAM_KEYS:
        official_value = official_params.get(key)
        requested_value = requested.get(key)
        if key == "edge_types":
            official_value = sorted(official_value) if official_value else official_value
            requested_value = sorted(requested_value) if requested_value else requested_value
        if official_value != requested_value:
            mismatched.append(key)

    if mismatched:
        return {
            "allowed": False,
            "message": f"Official preset parameters differ from docs/DECISION_C_SCORING.md on: {', '.join(mismatched)}.",
            "mismatched_keys": mismatched,
        }
    return {"allowed": True, "message": None, "mismatched_keys": []}


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
        "official_c_params": _official_c_params(),
    }


@router.post("/check-official-c-lock")
def check_official_c_lock(requested: dict):
    """Called by the Factorial Batch page before it will enable "Confirm
    & Launch" under the Official preset -- see
    check_official_c_params_lock()'s docstring for the refusal rules."""
    return check_official_c_params_lock(_official_c_params(), requested)


@router.post("/release-faiss-cache")
def release_faiss_cache():
    """Frees the ~4GB shared FAISS index + embedding memmap Condition C/D
    load once and reuse (engine_service._faiss_cache) -- the next C/D run
    reloads it from disk. See Settings page: "Release cached FAISS index"."""
    return engine_service.release_faiss_cache()
