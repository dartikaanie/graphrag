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
        "status", "c_retrieval_version", "alpha_primary", "alpha_ablation", "selection_record",
        "sample_split", "max_hops", "edge_types", "use_author_trust", "accepted_only",
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

_EDGES = ["EMBED_SIM", "HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR"]
_PENDING = {
    "status": "pending_selection", "c_retrieval_version": "v3", "alpha_primary": None, "alpha_ablation": None,
    "selection_record": {"path": "docs/DECISION_C_SCORING.md", "commit": None},
    "sample_split": "test", "max_hops": 2, "edge_types": _EDGES,
    "use_author_trust": False, "accepted_only": False,
}

_SELECTED = {
    **_PENDING, "status": "selected", "alpha_primary": 0.25, "alpha_ablation": 0.0,
    "selection_record": {"path": "docs/DECISION_C_SCORING.md", "commit": "abc1234"},
}


def _preset_request(official):
    return {k: v for k, v in official.items() if k not in ("status", "selection_record")}


def test_lock_refuses_request_while_pending_selection():
    """Even a request matching the pending defaults exactly is refused
    while status is not "selected" -- the Official preset as a whole is
    locked, not just the alpha knobs."""
    result = check_official_c_params_lock(_PENDING, _preset_request(_PENDING))
    assert result["allowed"] is False
    assert "DECISION_C_SCORING.md" in result["message"]


def test_lock_allows_exact_match_once_selected():
    result = check_official_c_params_lock(_SELECTED, _preset_request(_SELECTED))
    assert result["allowed"] is True
    assert result["message"] is None


def test_lock_refuses_mismatched_alpha_once_selected():
    requested = _preset_request(_SELECTED)
    requested["alpha_ablation"] = 0.5
    result = check_official_c_params_lock(_SELECTED, requested)
    assert result["allowed"] is False
    assert result["mismatched_keys"] == ["alpha_ablation"]


def test_lock_single_alpha_must_be_one_of_the_two():
    base = {"c_retrieval_version": "v3", "sample_split": "test"}
    assert check_official_c_params_lock(_SELECTED, {**base, "alpha": 0.0})["allowed"] is True
    assert check_official_c_params_lock(_SELECTED, {**base, "alpha": 0.25})["allowed"] is True
    result = check_official_c_params_lock(_SELECTED, {**base, "alpha": 0.5})
    assert result["allowed"] is False and result["mismatched_keys"] == ["alpha"]


def test_lock_refuses_v2_once_selected():
    result = check_official_c_params_lock(_SELECTED, {**_preset_request(_SELECTED), "c_retrieval_version": "v2"})
    assert result["mismatched_keys"] == ["c_retrieval_version"]


def test_lock_edge_types_order_insensitive():
    requested = _preset_request(_SELECTED)
    requested["edge_types"] = list(reversed(requested["edge_types"]))
    result = check_official_c_params_lock(_SELECTED, requested)
    assert result["allowed"] is True


def test_lock_allows_when_official_params_missing():
    result = check_official_c_params_lock(None, {"c_retrieval_version": "v3", "alpha_primary": 0.9})
    assert result["allowed"] is True


def test_check_official_c_lock_endpoint_pending_selection(monkeypatch):
    import app.routers.config as config_router

    monkeypatch.setattr(config_router, "_official_c_params", lambda: _PENDING)
    resp = client.post("/api/config/check-official-c-lock", json={"c_retrieval_version": "v2"})
    assert resp.status_code == 200
    assert resp.json()["allowed"] is False


import tempfile  # noqa: E402

_RUNS_TMP = Path(tempfile.mkdtemp(prefix="graphrag_runs_test_"))  # never the real backend/.runs

# ---------------------------------------------------------------------
# Official run matrix (llm/official_run_matrix.py) via /api/config/defaults,
# and the per-run launch rule in POST /api/runs (Amendment 10).
# ---------------------------------------------------------------------

def test_defaults_endpoint_returns_four_run_official_matrix(monkeypatch):
    import app.routers.config as config_router

    monkeypatch.setattr(config_router, "_official_c_params", lambda: _SELECTED)
    matrix = client.get("/api/config/defaults").json()["official_run_matrix"]
    assert [r["run"] for r in matrix] == ["R1", "R3", "R5", "R9"]  # Amendment 16: A, B-, C-primary-, D-grounded
    assert not any(r["legacy"] for r in matrix)
    c = {r["run"]: r for r in matrix if r["condition"] == "C"}
    assert {k: (v["alpha"], v["grounding"]) for k, v in c.items()} == {"R5": (0.25, True)}
    assert all(r["dashboard"]["c_retrieval_version"] == "v3" for r in c.values())


def test_defaults_endpoint_matrix_alphas_null_while_pending(monkeypatch):
    import app.routers.config as config_router

    monkeypatch.setattr(config_router, "_official_c_params", lambda: _PENDING)
    matrix = client.get("/api/config/defaults").json()["official_run_matrix"]
    assert [r["alpha"] for r in matrix if r["condition"] == "C"] == [None]


def _c_test_run(monkeypatch, official, tmp_path=None, **overrides):
    import llm.c_graphrag.official_params as op
    from app.services import run_registry

    monkeypatch.setattr(run_registry, "RUNS_DIR", _RUNS_TMP)

    monkeypatch.setattr(op, "load_official_params", lambda *a, **k: official)
    started = []
    import app.routers.runs as runs_router

    monkeypatch.setattr(runs_router.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())
    body = {"condition": "C", "mode": "batch", "n_sample": 384, "seed": 42, "sample_split": "test",
            "c_retrieval_version": "v3", "alpha": 0.25, **overrides}
    return client.post("/api/runs", json=body), started


def test_create_run_refuses_c_test_run_while_pending(monkeypatch):
    resp, started = _c_test_run(monkeypatch, _PENDING)
    assert resp.status_code == 400 and "pending_selection" in resp.json()["detail"] and not started


def test_create_run_refuses_c_test_run_v2_or_unlisted_alpha(monkeypatch):
    resp, _ = _c_test_run(monkeypatch, _SELECTED, c_retrieval_version="v2")
    assert resp.status_code == 400 and "v3" in resp.json()["detail"]
    resp, _ = _c_test_run(monkeypatch, _SELECTED, alpha=0.5)
    assert resp.status_code == 400 and "not an official alpha" in resp.json()["detail"]


def test_create_run_allows_official_c_alpha_and_dev_runs(monkeypatch):
    resp, started = _c_test_run(monkeypatch, _SELECTED, alpha=0.25)
    assert resp.status_code == 200 and len(started) == 1
    resp, started = _c_test_run(monkeypatch, _PENDING, sample_split="dev", c_retrieval_version="v2", alpha=None)
    assert resp.status_code == 200 and len(started) == 1


def test_create_run_refuses_legacy_cells_without_the_flag(monkeypatch):
    resp, started = _c_test_run(monkeypatch, _SELECTED, alpha=0.0)  # alpha_ablation
    assert resp.status_code == 400 and "legacy" in resp.json()["detail"] and not started
    resp, started = _c_test_run(monkeypatch, _SELECTED, alpha=0.25, require_grounding=False)  # C-primary-plain
    assert resp.status_code == 400 and "legacy" in resp.json()["detail"] and not started
    resp, started = _c_test_run(monkeypatch, _SELECTED, alpha=0.0, allow_legacy_cells=True)
    assert resp.status_code == 200 and len(started) == 1
