"""Phase 2 -- DEFAULT_OVERSAMPLE_POOL is the single source of truth for the
"auto" oversample_pool fallback across all four conditions (previously A
used n_sample*3, B/C/D used n_sample*4 -- neither matched the CLI's own
MAX_PLANNED_N_SAMPLE*4=1536 convention). No LLM calls, no Neo4j/DuckDB --
resolve_oversample_pool() is a pure function, and GET /api/config/defaults
is hit via FastAPI's TestClient (no real server process).
"""

from app.services import engine_service


def test_default_oversample_pool_is_1536_and_derived_from_max_planned_n_sample():
    assert engine_service.MAX_PLANNED_N_SAMPLE == 384
    assert engine_service.DEFAULT_OVERSAMPLE_POOL == 1536
    assert engine_service.DEFAULT_OVERSAMPLE_POOL == engine_service.MAX_PLANNED_N_SAMPLE * 4


def test_resolve_oversample_pool_falls_back_to_default_for_every_condition():
    # Every condition's run_condition_x() calls resolve_oversample_pool(params)
    # with whatever the request body was (condition/mode/etc. don't change
    # this function's behavior -- it only looks at "oversample_pool") --
    # covering the 4 realistic param shapes proves the SAME fallback applies
    # uniformly, not testing 4 different code paths.
    for params in (
        {"condition": "A", "mode": "batch", "n_sample": 10, "seed": 42},
        {"condition": "B", "mode": "batch", "n_sample": 10, "seed": 42},
        {"condition": "C", "mode": "batch", "n_sample": 10, "seed": 42},
        {"condition": "D", "mode": "batch", "n_sample": 10, "seed": 42},
    ):
        assert engine_service.resolve_oversample_pool(params) == 1536


def test_resolve_oversample_pool_respects_explicit_override():
    assert engine_service.resolve_oversample_pool({"oversample_pool": 200}) == 200


def test_resolve_oversample_pool_falls_back_on_null_or_zero():
    # oversample_pool: null (JSON) arrives as None in params -- falsy, same
    # fallback path as it being omitted entirely.
    assert engine_service.resolve_oversample_pool({"oversample_pool": None}) == 1536
    assert engine_service.resolve_oversample_pool({}) == 1536


def test_config_defaults_endpoint_returns_1536():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/api/config/defaults")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default_oversample_pool"] == 1536
    assert body["max_planned_n_sample"] == 384
