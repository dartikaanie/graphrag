"""
Live Neo4j values for the Process page (Rule 8). Never queried on page load:
the page shows the latest verify log by default (service.py), and these
read-only queries run only on an explicit "Refresh live values" request.
The result is cached in memory for CACHE_TTL_SEC; a refresh inside that
window returns the cached result. When Neo4j is unreachable the result says
so instead of failing the page.

All queries are MATCH ... RETURN only (no write clauses); test_process_guard
asserts that.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable

CACHE_TTL_SEC = 600
QUERY_TIMEOUT_SEC = 60

NODE_LABELS = ("Question", "Answer", "Tag", "User")
EDGE_TYPES = ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "TAGGED_WITH", "IS_RELATED_TO", "AUTHOR_TRUST", "TAG_COOCCUR",
              "EMBED_SIM")

# Per-label / per-type counts are answered from Neo4j's count store (no
# scan). The trustScoreVersion breakdowns scan Answer / Question nodes once.
QUERIES: dict[str, str] = {
    **{f"nodes:{label}": f"MATCH (n:{label}) RETURN count(n) AS n" for label in NODE_LABELS},
    **{f"edges:{t}": f"MATCH ()-[r:{t}]->() RETURN count(r) AS n" for t in EDGE_TYPES},
    "answer_trust_versions": "MATCH (a:Answer) RETURN a.trustScoreVersion AS version, count(*) AS n",
    "question_trust_versions": "MATCH (q:Question) RETURN q.trustScoreVersion AS version, count(*) AS n",
}

_lock = threading.Lock()
_cache: dict | None = None


def _default_runner(query: str) -> list[dict]:
    # lazy imports: the page loads without the neo4j driver connecting
    from neo4j import Query, READ_ACCESS

    from app.config import get_settings
    from app.db.neo4j_client import get_driver

    with get_driver().session(database=get_settings().neo4j_database, default_access_mode=READ_ACCESS) as session:
        return [r.data() for r in session.run(Query(query, timeout=QUERY_TIMEOUT_SEC))]


def _collect(runner: Callable[[str], list[dict]]) -> dict:
    rows = {name: runner(q) for name, q in QUERIES.items()}
    return {
        "nodes": {name.split(":", 1)[1]: r[0]["n"] for name, r in rows.items() if name.startswith("nodes:")},
        "edges": {name.split(":", 1)[1]: r[0]["n"] for name, r in rows.items() if name.startswith("edges:")},
        "answer_trust_versions": {str(r["version"]): r["n"] for r in rows["answer_trust_versions"]},
        "question_trust_versions": {str(r["version"]): r["n"] for r in rows["question_trust_versions"]},
    }


def get_cached() -> dict:
    with _lock:
        if _cache is None:
            return {"available": None, "as_of": None, "values": None, "message": "Not refreshed yet."}
        return {k: v for k, v in _cache.items() if k != "_t"}


def refresh(runner: Callable[[str], list[dict]] | None = None, force: bool = False) -> dict:
    global _cache
    with _lock:
        if (not force and _cache is not None and _cache.get("available")
                and time.monotonic() - _cache["_t"] < CACHE_TTL_SEC):
            return {k: v for k, v in _cache.items() if k != "_t"}
        try:
            values = _collect(runner or _default_runner)
            _cache = {"available": True, "as_of": datetime.now(timezone.utc).isoformat(), "values": values,
                      "message": None, "_t": time.monotonic()}
        except Exception as exc:  # any driver/connection error -> "unavailable", never a failed page
            _cache = {"available": False, "as_of": datetime.now(timezone.utc).isoformat(), "values": None,
                      "message": f"Neo4j unavailable ({type(exc).__name__})", "_t": time.monotonic()}
        return {k: v for k, v in _cache.items() if k != "_t"}


def clear_cache() -> None:
    global _cache
    with _lock:
        _cache = None
