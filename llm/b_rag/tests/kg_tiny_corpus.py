"""Tiny on-disk KG-answer corpus for offline tests (see conftest.py)."""

from pathlib import Path

TOPICS = ["python list sort", "git rebase conflict", "css flexbox center", "sql join null", "java stream map",
          "docker volume mount", "react state hook", "bash loop files", "regex lookahead", "numpy broadcasting"]


def write_tiny_corpus(root: Path, n_questions: int = 60, answers_per_question: int = 4, token_chunk_limit: int = 400):
    """Synthetic kg_answers.parquet + export_manifest.json in the layout
    build_kg_answer_index.py --stage export writes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from llm.b_rag import build_kg_answer_index as build
    from llm.b_rag.kg_corpus import CORPUS_VERSION, corpus_paths, ids_sha256, write_json_atomic
    from llm.common.text import get_tokenizer

    paths = corpus_paths(root, token_chunk_limit, "all-MiniLM-L6-v2")
    paths.root.mkdir(parents=True, exist_ok=True)
    records = []
    aid = 1000
    for q in range(n_questions):
        topic = TOPICS[q % len(TOPICS)]
        for k in range(answers_per_question):
            aid += 1
            records.append({
                "answer_id": aid, "question_id": 500 + q, "question_title": f"How to {topic} (case {q})?",
                "answer_body": f"<p>Answer {k} about {topic}: " + ("detail " * (5 + 40 * (k == 3))) + "</p>",
                "is_accepted": k == 0, "rel_type": "HAS_ACCEPTED_ANSWER" if k == 0 else "HAS_ANSWER",
            })
    rows = build.make_doc_rows(records, get_tokenizer(), token_chunk_limit)
    pq.write_table(pa.Table.from_pylist(rows, schema=build._parquet_schema()), paths.parquet, row_group_size=50)
    write_json_atomic(paths.export_manifest, {
        "corpus_version": CORPUS_VERSION, "token_chunk_limit": token_chunk_limit, "n_rows": len(rows),
        "n_answer_nodes": len(rows), "ids_sha256": ids_sha256([r["answer_id"] for r in rows]),
    })
    return paths, rows
