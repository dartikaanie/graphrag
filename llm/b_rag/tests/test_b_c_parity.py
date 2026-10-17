"""B/C parity on the REAL data (reports/b_corpus_parity.md). Dev split only
(positions 385..), never the test sample. Skipped when the built KG-answer
corpus / the parquet / the KG workspace are not available on this machine.

Tests that read Neo4j (the script-11 text check) additionally need
GRAPHRAG_INTEGRATION=1 -- reading Neo4j from a TEST is fine; Condition B
itself never does (test_b_no_neo4j_runtime.py).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
C_FIXTURE = REPO_ROOT / "llm" / "c_graphrag" / "tests" / "fixtures" / "c_retrieval_snapshot_dev.json"
INDEX_CACHE_DIR = REPO_ROOT / "llm" / "b_rag" / "index_cache"
N_DEV = 35
SEED, OVERSAMPLE_POOL, DEV_OFFSET = 42, 384 * 4, 385


def _env():
    from dotenv import dotenv_values

    return dotenv_values(REPO_ROOT / ".env")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def kg():
    from llm.b_rag.kg_corpus import corpus_paths, load_kg_answer_index

    paths = corpus_paths(INDEX_CACHE_DIR, 400, "all-MiniLM-L6-v2")
    if not paths.export_manifest.exists():
        pytest.skip("KG-answer corpus not built (build_kg_answer_index.py)")
    try:
        return load_kg_answer_index(INDEX_CACHE_DIR, 400, "all-MiniLM-L6-v2", log=lambda *a: None)
    except SystemExit as e:
        pytest.skip(f"KG-answer index not complete: {e}")


@pytest.fixture(scope="module")
def dev_sample():
    env = _env()
    qp, ap = env.get("QUESTIONS_PARQUET"), env.get("ANSWERS_PARQUET")
    if not (qp and ap and os.path.exists(qp) and os.path.exists(ap)):
        pytest.skip("questions/answers parquet not available")
    import duckdb

    from llm.b_rag import b_condition_b_rag as b
    from llm.sample_split import sample_questions_split

    b.log = lambda *a, **k: None
    con = duckdb.connect()
    cand = b.filter_by_token_limit(b.get_candidate_questions(con, qp, ap, OVERSAMPLE_POOL, SEED))
    df = sample_questions_split(cand, N_DEV, SEED, "dev", DEV_OFFSET)
    assert len(df) == N_DEV
    return df


@pytest.fixture(scope="module")
def question_memmap():
    """C's precomputed question embeddings (11_densify_embedding_similarity.py)."""
    kg_dir = _env().get("KG_WORKSPACE_DIR")
    if not (kg_dir and (Path(kg_dir) / "embed_checkpoint.json").exists()):
        pytest.skip("KG workspace (C's question embeddings) not available")
    kg_dir = Path(kg_dir)
    n = json.loads((kg_dir / "embed_checkpoint.json").read_text())["next_row"]
    ids = np.load(kg_dir / "question_embeddings_ivf_ids.npy")
    emb = np.memmap(kg_dir / "question_embeddings.f32", dtype="float32", mode="r", shape=(n, 384))
    return emb, {int(q): i for i, q in enumerate(ids)}


@pytest.fixture(scope="module")
def cg():
    c_dir = str(REPO_ROOT / "llm" / "c_graphrag")
    if c_dir not in sys.path:
        sys.path.insert(0, c_dir)
    import c_graphrag

    return c_graphrag


@pytest.fixture(scope="module")
def minilm_cpu():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# ---------------------------------------------------------------- corpus

def test_vectors_eq_parquet_rows_eq_exported_answer_nodes(kg):
    import pyarrow.parquet as pq

    from llm.b_rag.kg_corpus import read_json

    export = read_json(kg.paths.export_manifest)
    n_parquet = pq.ParquetFile(kg.paths.parquet).metadata.num_rows
    assert kg.ntotal == n_parquet == export["n_rows"] == export["n_answer_nodes"]
    assert export["n_orphan_answers"] == 0 and export["n_multi_parent_answers"] == 0


def test_candidate_text_identical_between_b_and_c(kg):
    """Every answer C retrieved in the pre-refactor snapshot (12 dev
    questions x 4 configs): B's text for that answer_id == C's chunk_text."""
    snap = json.loads(C_FIXTURE.read_text())
    c_sha = {}
    for records in snap["configs"].values():
        for rec in records:
            for item in rec["retrieved"]:
                c_sha.setdefault(item["answer_id"], set()).add(item["chunk_text_sha256"])
    assert all(len(v) == 1 for v in c_sha.values())  # C itself is consistent per answer_id
    docs = kg.fetch_docs(sorted(c_sha))
    assert set(docs) == set(c_sha), "an answer C retrieves is missing from B's corpus"
    mismatched = [aid for aid, shas in c_sha.items() if _sha(docs[aid]["doc_text"]) not in shas]
    assert not mismatched and len(c_sha) >= 20


# ---------------------------------------------------------------- query text

def test_query_text_identical_b_vs_c(dev_sample, cg):
    """B's query text == the text C actually passed to embed_query() before
    the refactor (snapshot), and == C's expression for all 35 dev questions."""
    from llm.common.text import build_query_text

    snap = json.loads(C_FIXTURE.read_text())
    c_text = {r["question_id"]: r["first_embed_query_text"] for r in snap["configs"]["v2_embed_query"]}
    rows = {int(r.Id): r for r in dev_sample.itertuples()}
    for qid, text in c_text.items():
        assert build_query_text(rows[qid].Title, rows[qid].Body) == text
    for r in dev_sample.itertuples():
        assert build_query_text(r.Title, r.Body) == f"{r.Title} {str(r.Body)[:1000]}"  # C's pre-refactor expression


def test_query_embedding_b_equals_c(dev_sample, cg, minilm_cpu, question_memmap):
    """B's query vector vs C's: (a) C's embed_query() fallback on the same
    text -- identical computation; (b) C's main path, the precomputed
    question vector from script 11 -- within 1e-6."""
    from llm.b_rag.kg_corpus import encode_query
    from llm.common.text import build_query_text

    emb, id2row = question_memmap
    worst = 0.0
    for r in dev_sample.itertuples():
        text = build_query_text(r.Title, r.Body)
        b_vec = encode_query(minilm_cpu, text)
        np.testing.assert_allclose(b_vec, cg.embed_query(text, minilm_cpu), atol=1e-6, rtol=0)
        assert int(r.Id) in id2row
        diff = float(np.abs(b_vec[0] - emb[id2row[int(r.Id)]]).max())
        worst = max(worst, diff)
        assert diff <= 1e-6, (r.Id, diff)
    print(f"max |B query vec - C precomputed vec| over {len(dev_sample)} dev questions: {worst:.2e}")


@pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1", reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
def test_query_text_matches_script11_neo4j_text(dev_sample):
    """Script 11 encoded title + " " + body[:1000] from the Question node --
    same string as B's query for every dev question."""
    from neo4j import GraphDatabase

    from llm.common.text import build_query_text

    env = _env()
    driver = GraphDatabase.driver(env["NEO4J_URI"].strip(), auth=(env.get("NEO4J_USER", "neo4j").strip(),
                                                                  env["NEO4J_PASSWORD"].strip().strip('"').strip("'")))
    try:
        with driver.session(database=env.get("NEO4J_DATABASE", "neo4j").strip(), default_access_mode="READ") as s:
            neo = {r["id"]: f"{r['title'] or ''} {(r['body'] or '')[:1000]}" for r in s.run(
                "UNWIND $ids AS i MATCH (q:Question {id: i}) RETURN q.id AS id, q.title AS title, q.body AS body",
                ids=[int(i) for i in dev_sample["Id"]])}
    finally:
        driver.close()
    assert len(neo) == len(dev_sample)
    for r in dev_sample.itertuples():
        assert build_query_text(r.Title, r.Body) == neo[int(r.Id)]


# ---------------------------------------------------------------- leakage

def test_dev_results_exclude_own_thread_and_keep_top_k(kg, dev_sample, minilm_cpu):
    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import encode_query
    from llm.common.leakage import assert_no_leakage, build_exclusion
    from llm.common.text import build_query_text

    threads = kg.thread_answer_ids(dev_sample["Id"].astype(int).tolist())
    would_leak = 0
    for r in dev_sample.itertuples():
        qid = int(r.Id)
        excl = build_exclusion(qid, threads[qid], r.AcceptedAnswerId)
        assert int(r.AcceptedAnswerId) in threads[qid]  # the accepted answer is in B's corpus...
        text = build_query_text(r.Title, r.Body)
        q_emb = encode_query(minilm_cpu, text)
        out = b.retrieve_context_kg(minilm_cpu, kg, text, 5, excl, q_emb)
        assert len(out) == 5
        assert_no_leakage(out, excl)  # ...but never retrieved
        unfiltered = [int(kg.question_ids[row]) for row, _ in kg.search(q_emb, 5)]
        would_leak += qid in unfiltered
    print(f"{would_leak}/{len(dev_sample)} dev questions would retrieve their own thread without exclusion")
    assert would_leak > 0  # the exclusion is doing real work on this corpus
