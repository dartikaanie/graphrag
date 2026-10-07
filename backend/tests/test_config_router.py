"""Router-level test for /api/config/defaults -- in particular,
official_c_params, which the Factorial Batch page's "Official (n=384)"
preset reads to lock Condition C's v3 parameters to docs/
DECISION_C_SCORING.md's machine-readable block (see
llm/c_graphrag/tests/test_official_params_sync.py for the doc<->json
sync guarantee this endpoint relies on).
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PARAMS_PATH = REPO_ROOT / "llm" / "c_graphrag" / "official_params.json"


def test_defaults_endpoint_includes_official_c_params():
    resp = client.get("/api/config/defaults")
    assert resp.status_code == 200
    body = resp.json()
    assert "official_c_params" in body
    assert body["official_c_params"] == json.loads(OFFICIAL_PARAMS_PATH.read_text())


def test_official_c_params_has_expected_keys():
    resp = client.get("/api/config/defaults")
    params = resp.json()["official_c_params"]
    assert set(params) == {
        "status", "c_retrieval_version", "alpha", "sample_split", "max_hops", "edge_types",
        "use_author_trust", "accepted_only",
    }


def test_defaults_endpoint_returns_none_when_file_missing(monkeypatch):
    import app.routers.config as config_router

    monkeypatch.setattr(config_router, "OFFICIAL_C_PARAMS_PATH", Path("/nonexistent/official_params.json"))
    resp = client.get("/api/config/defaults")
    assert resp.status_code == 200
    assert resp.json()["official_c_params"] is None


# ---------------------------------------------------------------------
# check_official_c_params_lock() -- the Official preset's launch-refusal
# logic. Refuses whenever status is "pending_selection" REGARDLESS of
# c_retrieval_version, and otherwise refuses on any locked-param mismatch.
# ---------------------------------------------------------------------

from app.routers.config import check_official_c_params_lock  # noqa: E402

_PENDING = {
    "status": "pending_selection", "c_retrieval_version": None, "alpha": None,
    "sample_split": "test", "max_hops": 2,
    "edge_types": ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"],
    "use_author_trust": False, "accepted_only": False,
}

_LOCKED_RESOLVED_V3 = {
    "status": "locked", "c_retrieval_version": "v3", "alpha": 0.5,
    "sample_split": "test", "max_hops": 2,
    "edge_types": ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"],
    "use_author_trust": False, "accepted_only": False,
}


def test_lock_refuses_v2_request_while_pending_selection():
    """Even a request matching the pending defaults exactly is refused
    while status is "pending_selection" -- the Official preset as a
    whole is locked, not just the v3 knobs."""
    requested = {k: v for k, v in _PENDING.items() if k != "status"}
    requested["c_retrieval_version"] = "v2"
    result = check_official_c_params_lock(_PENDING, requested)
    assert result["allowed"] is False
    assert "DECISION_C_SCORING.md" in result["message"]


def test_lock_allows_exact_match_once_resolved():
    requested = {k: v for k, v in _LOCKED_RESOLVED_V3.items() if k != "status"}
    result = check_official_c_params_lock(_LOCKED_RESOLVED_V3, requested)
    assert result["allowed"] is True
    assert result["message"] is None


def test_lock_refuses_mismatched_alpha_once_resolved():
    requested = {k: v for k, v in _LOCKED_RESOLVED_V3.items() if k != "status"}
    requested["alpha"] = 0.75
    result = check_official_c_params_lock(_LOCKED_RESOLVED_V3, requested)
    assert result["allowed"] is False
    assert "alpha" in result["mismatched_keys"]


def test_lock_edge_types_order_insensitive():
    requested = {k: v for k, v in _LOCKED_RESOLVED_V3.items() if k != "status"}
    requested["edge_types"] = list(reversed(requested["edge_types"]))
    result = check_official_c_params_lock(_LOCKED_RESOLVED_V3, requested)
    assert result["allowed"] is True


def test_lock_allows_when_official_params_missing():
    result = check_official_c_params_lock(None, {"c_retrieval_version": "v3", "alpha": 0.9})
    assert result["allowed"] is True


def test_check_official_c_lock_endpoint_pending_selection(monkeypatch):
    import app.routers.config as config_router

    monkeypatch.setattr(config_router, "_official_c_params", lambda: _PENDING)
    resp = client.post("/api/config/check-official-c-lock", json={"c_retrieval_version": "v2"})
    assert resp.status_code == 200
    assert resp.json()["allowed"] is False
