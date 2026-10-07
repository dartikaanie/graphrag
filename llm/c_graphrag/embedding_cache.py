"""
embedding_cache.py
=====================================
On-disk cache for candidate-text embeddings (Step 3.4 of docs/
agent_prompt_c_retrieval_v3_devset.md), keyed by (answer_id, sha256(text)),
backed by SQLite (not JSON, per explicit requirement) -- so an α sweep
over the SAME dev questions doesn't re-encode identical candidate text
for every α value. Cheap, append-mostly, single-writer-per-run usage, so
a plain sqlite3 connection with WAL is enough (no ORM, no server).

Values are stored as raw float32 bytes (384 floats for all-MiniLM-L6-v2,
but the dimension isn't hardcoded here -- the cache just stores whatever
byte length was written and lets the caller reshape it).

FAIL-SAFE: the cache is only a speed-up. Any SQLite error (on open, get,
put, or close) logs ONE warning naming the cache path and disables the
cache for the rest of the run: get() then always misses and put() is a
no-op, so the caller simply computes embeddings directly. A cache error
can never stop a run. (It can never change a result either, provided the
caller's embeddings don't depend on batch composition -- see
c_graphrag.compute_candidate_similarities(), which encodes each text on
its own for exactly that reason.)

LOCATION: the default is a local, non-synced cache directory
(~/Library/Caches/graphrag on macOS, $XDG_CACHE_HOME or ~/.cache/graphrag
elsewhere), overridable with C_EMBEDDING_CACHE_PATH. Do NOT put it in a
folder synced by Google Drive / iCloud / Dropbox: Google Drive for
desktop hard-links new files into its .tmp.driveupload staging folder to
upload them, and Apple's SQLite treats a database whose link count
changes under an open connection as replaced (SQLITE_IOERR_VNODE,
reported by Python as "disk I/O error"). That is exactly what broke the
first stage-1 runs, whose cache lived inside the Drive-backed repo.
"""

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

CONNECT_TIMEOUT_SEC = 10.0


def default_cache_path() -> Path:
    """C_EMBEDDING_CACHE_PATH if set, else a local per-user cache dir that
    no backup/sync client watches by default."""
    env = os.getenv("C_EMBEDDING_CACHE_PATH")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "graphrag" / "embedding_cache.sqlite3"


DEFAULT_CACHE_PATH = default_cache_path()


def text_cache_key(answer_id: int, text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Usage:
        cache = EmbeddingCache(path)
        vec = cache.get(answer_id, text)          # None if not cached (or cache disabled)
        cache.put(answer_id, text, vec)            # vec: 1-D float32 ndarray
        cache.close()
    Safe to share one file across every α in a sweep -- a cache hit for
    one α is a cache hit for every α (the cache key never includes alpha,
    only the candidate's own (answer_id, text) -- the embedding itself
    doesn't depend on alpha at all).

    `disabled` / `disabled_reason` report whether a SQLite error turned
    the cache off during this run (see module docstring).
    """

    def __init__(self, path: Path | str | None = None, timeout: float = CONNECT_TIMEOUT_SEC, logger=print):
        self.path = Path(path) if path is not None else default_cache_path()
        self.hits = 0
        self.misses = 0
        self.disabled = False
        self.disabled_reason: str | None = None
        self._log = logger
        self._conn = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=timeout)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    answer_id INTEGER NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (answer_id, text_hash)
                )
                """
            )
            self._conn.commit()
        except (sqlite3.Error, OSError) as e:
            self._disable("open", e)

    def _disable(self, operation: str, exc: Exception) -> None:
        if not self.disabled:
            self._log(
                f"      [WARN] embedding cache DISABLED for the rest of this run -- SQLite error during "
                f"{operation} on cache file '{self.path}': {type(exc).__name__}: {exc}. Continuing without "
                f"the cache (embeddings are computed directly; results are unaffected). If this path is "
                f"inside a Google Drive/iCloud/Dropbox-synced folder, set C_EMBEDDING_CACHE_PATH to a local, "
                f"non-synced path (default: {default_cache_path()})."
            )
        self.disabled = True
        self.disabled_reason = f"{operation}: {type(exc).__name__}: {exc}"
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def get(self, answer_id: int, text: str) -> np.ndarray | None:
        if self.disabled:
            self.misses += 1
            return None
        key = text_cache_key(answer_id, text)
        try:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE answer_id = ? AND text_hash = ?",
                (int(answer_id), key),
            ).fetchone()
        except sqlite3.Error as e:
            self._disable("get", e)
            self.misses += 1
            return None
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return np.frombuffer(row[0], dtype=np.float32)

    def put(self, answer_id: int, text: str, vector: np.ndarray) -> None:
        if self.disabled:
            return
        key = text_cache_key(answer_id, text)
        vector = np.asarray(vector, dtype=np.float32)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (answer_id, text_hash, vector) VALUES (?, ?, ?)",
                (int(answer_id), key, vector.tobytes()),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._disable("put", e)

    def get_many(self, items: list[tuple[int, str]]) -> dict[tuple[int, str], np.ndarray]:
        """Batched lookup -- returns only the (answer_id, text) keys that
        were actually found (missing keys are simply absent, not None-
        valued), so the caller can compute `misses = [k for k in items if
        k not in found]` and only encode those."""
        found: dict[tuple[int, str], np.ndarray] = {}
        for answer_id, text in items:
            vec = self.get(answer_id, text)
            if vec is not None:
                found[(answer_id, text)] = vec
        return found

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error as e:
                self._disable("close", e)
