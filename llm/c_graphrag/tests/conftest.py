"""conftest.py -- adds the repo root AND this directory (llm/c_graphrag/)
to sys.path, so tests can `import c_graphrag`/`import embedding_cache`
the same bare way c_graphrag.py's own CLI usage expects (run from within
this directory), while also being able to `from llm.manifest import ...`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

import pytest

from question_index_spy import QuestionIndexReadSpy


@pytest.fixture(scope="session", autouse=True)
def question_index_read_spy():
    """Integration sessions only: every faiss.read_index of the question index
    goes through the spy, which fails on a second read in this process."""
    if os.getenv("GRAPHRAG_INTEGRATION") != "1":
        yield None
        return
    import faiss

    spy = QuestionIndexReadSpy(faiss).install()
    yield spy
    spy.uninstall()


@pytest.fixture(scope="session")
def question_faiss(question_index_read_spy):
    """C's question index + embedding memmap, loaded ONCE per pytest process
    and shared by every integration test (c_graphrag.load_faiss_cache)."""
    from dotenv import load_dotenv

    import c_graphrag as cg

    load_dotenv(REPO_ROOT / ".env")
    return cg.load_faiss_cache(Path(os.environ["KG_WORKSPACE_DIR"]), lambda *a, **k: None)
