"""Expanded leakage exclusion on the REAL data (Amendment 8; reports/leakage_audit.md).
Reads Neo4j, the parquet and the KG workspace -- opt-in with GRAPHRAG_INTEGRATION=1.
Dev split (positions 385..) and non-test pool questions only; the test sample is
never used as a query."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1",
                                reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "llm" / "d_lightrag"))
MODES = ("thread", "dup+neardup0.95", "dup+linked+neardup0.95")
N_DEV_TEST = 20


@pytest.fixture(scope="module")
def env():
    import duckdb
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    import c_graphrag as cg

    load_dotenv(REPO_ROOT / ".env")
    cg.log = lambda *a, **k: None
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    qp, ap = os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"]
    pool = cg.filter_by_token_limit(cg.get_candidate_questions(con, qp, ap, 384 * 4, 42))
    perm = pool.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dev = perm.iloc[384:384 + N_DEV_TEST].reset_index(drop=True)
    non_test = perm.iloc[384:].reset_index(drop=True)
    driver, database = cg.connect_neo4j(lambda *a, **k: None)
    faiss = cg.load_faiss_cache(Path(os.environ["KG_WORKSPACE_DIR"]), lambda *a, **k: None)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    yield {"cg": cg, "con": con, "ap": ap, "dev": dev, "non_test": non_test, "driver": driver,
           "database": database, "faiss": faiss, "model": model}
    driver.close()


def _thread(env, ids):
    return env["cg"].get_all_answer_ids_for_questions(env["con"], env["ap"], [int(i) for i in ids])


def _run(cond, env, df, mode):
    from llm.common.leakage import LeakageContext

    mod = env["cg"] if cond == "C" else __import__("d_lightrag")
    mod.log = lambda *a, **k: None
    index, ids_arr, emb, id_to_row = env["faiss"]
    with tempfile.TemporaryDirectory() as tmp:
        recs, _ = mod.process_sample(
            df, None, None, env["model"], env["driver"], env["database"], index, ids_arr, emb, id_to_row,
            _thread(env, df["Id"]), 5, 3, 3, 400, "none", Path(tmp) / "o.jsonl", retrieval_only=True,
            leakage=LeakageContext.from_mode(mode), **({"c_retrieval_version": "v3", "alpha": 0.5} if cond == "C" else {}))
    return recs


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cond", ["C", "D"])
def test_dev_retrieval_disjoint_from_exclusion_set(env, cond, mode):
    from llm.common.leakage import LeakageContext

    ctx = LeakageContext.from_mode(mode)
    recs = _run(cond, env, env["dev"], mode)
    thread = _thread(env, env["dev"]["Id"])
    acc = dict(zip(env["dev"]["Id"].astype(int), env["dev"]["AcceptedAnswerId"].astype(int)))
    assert len(recs) == N_DEV_TEST
    for r in recs:
        ex = ctx.exclusion_for(r["question_id"], thread.get(r["question_id"], []), acc[r["question_id"]])
        assert not any(ex.excludes(i["question_id"], i["answer_id"]) for i in r["retrieved_context"])
        assert r["leakage"]["mode"] == mode and r["leakage"]["final_k"] == len(r["retrieved_context"])


@pytest.mark.parametrize("mode", MODES)
def test_dev_retrieval_disjoint_from_exclusion_set_b(env, mode):
    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import encode_query
    from llm.common.leakage import LeakageContext
    from llm.common.text import build_query_text

    kg = _kg()
    ctx = LeakageContext.from_mode(mode)
    threads = kg.thread_answer_ids(env["dev"]["Id"].astype(int).tolist())
    for r in env["dev"].itertuples():
        ex = ctx.exclusion_for(int(r.Id), threads[int(r.Id)], r.AcceptedAnswerId)
        text = build_query_text(r.Title, r.Body)
        out = b.retrieve_context_kg(env["model"], kg, text, 5, ex, encode_query(env["model"], text))
        assert len(out) == 5 and not any(ex.excludes(i["question_id"], i["answer_id"]) for i in out)


_KG = {}


def _kg():
    from llm.b_rag.kg_corpus import load_kg_answer_index

    if "kg" not in _KG:
        _KG["kg"] = load_kg_answer_index(REPO_ROOT / "llm" / "b_rag" / "index_cache", 400, "all-MiniLM-L6-v2",
                                         log=lambda *a: None)
    return _KG["kg"]


@pytest.fixture(scope="module")
def dup_case(env):
    """A NON-TEST pool question with a real IS_RELATED_TO Duplicate in the KG."""
    from llm.common.leakage import LeakageContext, default_mode

    ctx = LeakageContext.from_mode(default_mode())
    for r in env["non_test"].itertuples():
        q = int(r.Id)
        if q in ctx.table and ctx.table.counts(q)["n_dup"] > 0:
            rq, ra = ctx.table.related(q)
            return {"row": env["non_test"][env["non_test"]["Id"] == r.Id].reset_index(drop=True),
                    "q": q, "dups": sorted(rq - {q}), "dup_answers": ra, "ctx": ctx}
    pytest.skip("no non-test pool question with a Duplicate")


def test_real_duplicate_is_never_anchor_q2_expansion_or_answer_in_c(env, dup_case):
    recs = _run("C", env, dup_case["row"], "dup+neardup0.95")
    r = recs[0]
    dups = set(dup_case["dups"])
    assert not dups & set(r["anchor_question_ids"])
    assert not dups & set(r["expansion_qids"])
    assert not dups & {i["question_id"] for i in r["retrieved_context"]}
    assert not set(dup_case["dup_answers"]) & {i["answer_id"] for i in r["retrieved_context"]}
    # q2: traverse from a question linked to the duplicate -- the duplicate is never passed through
    with env["driver"].session(database=env["database"]) as s:
        via = [rec["x"] for rec in s.run(
            "UNWIND $d AS d MATCH (x:Question)-[:IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM]->(dq:Question {id: d}) "
            "WHERE NOT x.id IN $d AND x.id <> $q RETURN DISTINCT x.id AS x ORDER BY x LIMIT 3",
            d=list(dups), q=dup_case["q"])]
    if via:
        ex = dup_case["ctx"].exclusion_for(dup_case["q"])
        cands = env["cg"].traverse_graph(env["driver"], env["database"], via, set(ex.question_ids), set(ex.answer_ids))
        assert not dups & {int(c["via_question_id"]) for c in cands}
        assert not set(ex.answer_ids) & {int(c["answer_id"]) for c in cands}


def test_real_duplicate_is_never_anchor_or_answer_in_d_and_b(env, dup_case):
    from llm.b_rag import b_condition_b_rag as b
    from llm.b_rag.kg_corpus import encode_query
    from llm.common.text import build_query_text

    dups = set(dup_case["dups"])
    r = _run("D", env, dup_case["row"], "dup+neardup0.95")[0]
    assert not dups & set(r["anchor_question_ids"])
    assert not dups & {i["question_id"] for i in r["retrieved_context"]}
    row = dup_case["row"].iloc[0]
    kg = _kg()
    ex = dup_case["ctx"].exclusion_for(dup_case["q"], kg.thread_answer_ids([dup_case["q"]])[dup_case["q"]],
                                       row["AcceptedAnswerId"])
    text = build_query_text(row["Title"], row["Body"])
    out = b.retrieve_context_kg(env["model"], kg, text, 5, ex, encode_query(env["model"], text))
    assert len(out) == 5 and not dups & {i["question_id"] for i in out}
    assert not set(dup_case["dup_answers"]) & {i["answer_id"] for i in out}


def test_exclusion_sets_parquet_is_deterministic(tmp_path):
    """Rebuilding gives a byte-identical file (same sha256 as the committed one)."""
    from llm.common.leakage import DEFAULT_EXCLUSION_SETS_PATH

    out = tmp_path / "exclusion_sets.parquet"
    subprocess.run([sys.executable, str(REPO_ROOT / "llm" / "leakage" / "build_exclusion_sets.py"),
                    "--output", str(out)], check=True, cwd=REPO_ROOT, capture_output=True)
    committed = json.loads(DEFAULT_EXCLUSION_SETS_PATH.with_suffix(".meta.json").read_text())["sha256"]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == committed == \
        hashlib.sha256(DEFAULT_EXCLUSION_SETS_PATH.read_bytes()).hexdigest()
