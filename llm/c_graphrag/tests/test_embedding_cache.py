"""Step 5: embedding cache hit/miss (SQLite-backed, per explicit
requirement -- not JSON)."""

import numpy as np

from embedding_cache import EmbeddingCache, text_cache_key


def test_miss_then_hit(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    assert cache.get(1, "hello") is None
    assert cache.misses == 1

    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.put(1, "hello", vec)
    retrieved = cache.get(1, "hello")
    assert cache.hits == 1
    np.testing.assert_array_almost_equal(retrieved, vec)
    cache.close()


def test_different_text_same_answer_id_different_key(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put(1, "text a", np.array([1.0, 0.0], dtype=np.float32))
    cache.put(1, "text b", np.array([0.0, 1.0], dtype=np.float32))
    a = cache.get(1, "text a")
    b = cache.get(1, "text b")
    np.testing.assert_array_almost_equal(a, [1.0, 0.0])
    np.testing.assert_array_almost_equal(b, [0.0, 1.0])
    cache.close()


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache1 = EmbeddingCache(path)
    cache1.put(42, "persisted", np.array([5.0, 6.0], dtype=np.float32))
    cache1.close()

    cache2 = EmbeddingCache(path)
    retrieved = cache2.get(42, "persisted")
    np.testing.assert_array_almost_equal(retrieved, [5.0, 6.0])
    cache2.close()


def test_get_many_returns_only_found_keys(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put(1, "a", np.array([1.0], dtype=np.float32))
    found = cache.get_many([(1, "a"), (2, "b")])
    assert (1, "a") in found
    assert (2, "b") not in found
    cache.close()


def test_sqlite_file_actually_created_not_json(tmp_path):
    path = tmp_path / "cache.sqlite3"
    cache = EmbeddingCache(path)
    cache.put(1, "x", np.array([1.0], dtype=np.float32))
    cache.close()
    assert path.exists()
    with open(path, "rb") as f:
        header = f.read(16)
    assert header.startswith(b"SQLite format 3")


def test_text_cache_key_deterministic():
    assert text_cache_key(1, "hello") == text_cache_key(1, "hello")
    assert text_cache_key(1, "hello") != text_cache_key(1, "world")
