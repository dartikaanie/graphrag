"""The embedding cache is a speed-up only: a SQLite error must never stop
a run or change a result (see embedding_cache.py's module docstring)."""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from c_graphrag import compute_candidate_similarities
from embedding_cache import EmbeddingCache, default_cache_path


class _BatchSensitiveModel:
    """Deterministic per text, PLUS a ~1e-7 perturbation that depends on
    the other texts in the batch -- the same kind of drift the real
    all-MiniLM-L6-v2 shows. A caller that batches would therefore get
    different vectors with vs. without the cache."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, convert_to_numpy=True, device="cpu", batch_size=32):
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(batch_size)
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            drift = 1e-7 * (len(chunk) - 1 + sum(len(t) for t in chunk) % 5)
            for t in chunk:
                seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
                v = np.random.default_rng(seed).normal(size=8).astype(np.float32)
                out.append(v + np.float32(drift))
        return np.stack(out)


class _FailingConn:
    """Wraps a real sqlite3 connection; raises 'disk I/O error' on every
    execute after `fail_after` successful ones (like the Google Drive
    hard-link problem: fine at first, then SQLITE_IOERR)."""

    def __init__(self, conn, fail_after):
        self._conn, self._left = conn, fail_after

    def execute(self, *a, **k):
        if self._left <= 0:
            raise sqlite3.OperationalError("disk I/O error")
        self._left -= 1
        return self._conn.execute(*a, **k)

    def commit(self):
        return self._conn.commit()

    def close(self):
        return self._conn.close()


QUERY = np.random.default_rng(0).normal(size=(1, 8)).astype(np.float32)
# Three "questions" whose candidate pools overlap in different orders --
# a shared text is first encoded while processing one question and later
# served from the cache for another.
POOLS = [
    ([1, 2, 3, 4], ["alpha text", "beta text longer", "gamma", "delta delta delta"]),
    ([3, 5, 2], ["gamma", "epsilon", "beta text longer"]),
    ([6, 1, 4, 5], ["zeta zeta", "alpha text", "delta delta delta", "epsilon"]),
]


def _sims(model, cache):
    return [compute_candidate_similarities(model, QUERY, texts, ids, cache=cache) for ids, texts in POOLS]


def _logger():
    lines = []
    return lines, lines.append


def test_similarities_identical_with_without_and_with_failing_cache(tmp_path):
    model = _BatchSensitiveModel()
    baseline = _sims(model, None)

    lines, log = _logger()
    cold = EmbeddingCache(tmp_path / "c.sqlite3", logger=log)
    assert _sims(model, cold) == baseline           # populating
    assert _sims(model, cold) == baseline           # every lookup a hit now
    assert cold.hits > 0 and not cold.disabled
    cold.close()

    for fail_after in (0, 1, 3, 7):                 # fail at get / put, early / mid-run
        lines, log = _logger()
        cache = EmbeddingCache(tmp_path / f"f{fail_after}.sqlite3", logger=log)
        cache._conn = _FailingConn(cache._conn, fail_after)
        assert _sims(model, cache) == baseline, f"fail_after={fail_after}"
        assert cache.disabled
        assert len(lines) == 1 and str(cache.path) in lines[0] and "disk I/O error" in lines[0]
        cache.close()


def test_candidates_are_encoded_one_text_per_call():
    model = _BatchSensitiveModel()
    compute_candidate_similarities(model, QUERY, POOLS[0][1], POOLS[0][0], cache=None)
    assert model.calls == [1]


def test_disabled_cache_never_raises_and_warns_once(tmp_path):
    lines, log = _logger()
    cache = EmbeddingCache(tmp_path / "c.sqlite3", logger=log)
    cache._conn = _FailingConn(cache._conn, 0)
    assert cache.get(1, "x") is None
    cache.put(1, "x", np.ones(3, dtype=np.float32))
    assert cache.get(2, "y") is None
    assert cache.get_many([(1, "x"), (2, "y")]) == {}
    cache.close()
    assert cache.disabled and cache.disabled_reason.startswith("get: OperationalError")
    assert len(lines) == 1


def test_open_failure_disables_instead_of_raising(tmp_path):
    lines, log = _logger()
    bad = tmp_path / "is_a_directory"
    bad.mkdir()
    cache = EmbeddingCache(bad, logger=log)
    assert cache.disabled and cache.get(1, "x") is None
    cache.put(1, "x", np.ones(3, dtype=np.float32))
    cache.close()
    assert len(lines) == 1 and str(bad) in lines[0]


def test_default_cache_path_is_local_and_outside_the_repo(monkeypatch):
    monkeypatch.delenv("C_EMBEDDING_CACHE_PATH", raising=False)
    path = default_cache_path()
    repo = Path(__file__).resolve().parents[3]
    assert repo not in path.parents
    assert path.name == "embedding_cache.sqlite3"
    monkeypatch.setenv("C_EMBEDDING_CACHE_PATH", "~/somewhere/cache.sqlite3")
    assert default_cache_path() == Path.home() / "somewhere" / "cache.sqlite3"


def test_connection_uses_a_timeout(tmp_path, monkeypatch):
    seen = {}
    real_connect = sqlite3.connect

    def spy(*a, **k):
        seen.update(k)
        return real_connect(*a, **k)

    monkeypatch.setattr(sqlite3, "connect", spy)
    EmbeddingCache(tmp_path / "c.sqlite3").close()
    assert seen.get("timeout", 0) > 0
