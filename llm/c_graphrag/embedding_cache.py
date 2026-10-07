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
"""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_CACHE_PATH = Path("embedding_cache.sqlite3")


def text_cache_key(answer_id: int, text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Usage:
        cache = EmbeddingCache(path)
        vec = cache.get(answer_id, text)          # None if not cached
        cache.put(answer_id, text, vec)            # vec: 1-D float32 ndarray
        cache.close()
    Safe to share one file across every α in a sweep -- a cache hit for
    one α is a cache hit for every α (the cache key never includes alpha,
    only the candidate's own (answer_id, text) -- the embedding itself
    doesn't depend on alpha at all).
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
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
        self.hits = 0
        self.misses = 0

    def get(self, answer_id: int, text: str) -> np.ndarray | None:
        key = text_cache_key(answer_id, text)
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE answer_id = ? AND text_hash = ?",
            (int(answer_id), key),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return np.frombuffer(row[0], dtype=np.float32)

    def put(self, answer_id: int, text: str, vector: np.ndarray) -> None:
        key = text_cache_key(answer_id, text)
        vector = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (answer_id, text_hash, vector) VALUES (?, ?, ?)",
            (int(answer_id), key, vector.tobytes()),
        )
        self._conn.commit()

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
        self._conn.close()
