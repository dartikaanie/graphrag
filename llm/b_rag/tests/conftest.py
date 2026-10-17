"""Adds the repo root to sys.path (same as llm/tests/conftest.py) and
provides a tiny on-disk KG-answer corpus for offline tests -- built with
the real build stages (encode/index) but from a synthetic parquet, so no
Neo4j and no external drive are needed."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_tiny_corpus import write_tiny_corpus  # noqa: E402


@pytest.fixture(scope="session")
def minilm():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory, minilm):
    """Fully built tiny corpus: (index_cache_dir, paths, rows)."""
    from llm.b_rag import build_kg_answer_index as build

    root = tmp_path_factory.mktemp("kg_index_cache")
    paths, rows = write_tiny_corpus(root)
    build.stage_encode(paths, "all-MiniLM-L6-v2", batch_size=16, device="cpu", unit_rows=64, model=minilm)
    build.stage_index(paths)
    return root, paths, rows
