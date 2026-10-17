"""Condition B must never touch Neo4j at runtime (reports/b_corpus_parity.md):
Neo4j is only the offline export source of build_kg_answer_index.py."""

import ast
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
B_RUNTIME_FILES = [
    "llm/b_rag/b_condition_b_rag.py", "llm/b_rag/kg_corpus.py",
    "llm/common/__init__.py", "llm/common/text.py", "llm/common/leakage.py",
]


def _imported_modules(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("rel", B_RUNTIME_FILES)
def test_no_neo4j_import_anywhere_on_b_runtime_path(rel):
    assert "neo4j" not in _imported_modules(REPO_ROOT / rel)


def test_importing_b_does_not_load_neo4j():
    code = ("import sys; sys.path.insert(0, %r); import llm.b_rag.b_condition_b_rag, llm.b_rag.kg_corpus; "
            "print('neo4j' in sys.modules)" % str(REPO_ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


class _PoisonedNeo4j(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"Condition B touched neo4j.{name} at runtime")


def test_b_run_succeeds_with_neo4j_poisoned_and_no_env(tiny_corpus, minilm, monkeypatch, tmp_path):
    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import load_kg_answer_index

    for var in [v for v in list(__import__("os").environ) if v.startswith("NEO4J_")]:
        monkeypatch.delenv(var)
    monkeypatch.setitem(sys.modules, "neo4j", _PoisonedNeo4j("neo4j"))
    monkeypatch.setattr(b, "log", lambda *a, **k: None)
    monkeypatch.setattr(b.time, "sleep", lambda s: None)

    root, _, rows = tiny_corpus
    kg = load_kg_answer_index(root, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    sample_df = pd.DataFrame([{
        "Id": 503, "Title": next(r["question_title"] for r in rows if r["question_id"] == 503),
        "Body": "<p>How do I do this?</p>", "Tags": "<python>", "AcceptedAnswerId": 1013,
        "AcceptedAnswerBody": "Use sorted().", "n_tokens": 20, "ViewCount": 1, "Score": 1,
    }])

    def fake_llm(client, messages, model):
        return {"content": "Use sorted() [SO-500]."}

    records = b.process_sample(sample_df, None, fake_llm, minilm, kg, None, 5, "fake", tmp_path / "out.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["n_context_items_used"] == 5
    assert rec["query_text"] == f"{sample_df.Title[0]} {sample_df.Body[0]}"
    assert all(item["question_id"] != 503 for item in rec["retrieved_context"])
