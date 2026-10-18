"""Amendment 9 (answerable-only semantic expansion) on the REAL KG -- dev split
only. Reads Neo4j + parquet + KG workspace: opt-in with GRAPHRAG_INTEGRATION=1."""

import json
import os
import tempfile
from pathlib import Path

import pytest

import c_retrieval_snapshot as snapshot

pytestmark = pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1",
                                reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
REPO_ROOT = Path(__file__).resolve().parents[3]
N_DEV_TEST = 20


@pytest.fixture(scope="module")
def env(question_faiss):
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
    dev = cg.sample_questions_split(pool, N_DEV_TEST, 42, split="dev", dev_offset=385)
    thread = cg.get_all_answer_ids_for_questions(con, ap, dev["Id"].astype(int).tolist())
    driver, database = cg.connect_neo4j(lambda *a, **k: None)
    faiss = question_faiss  # one copy per pytest process (conftest.py)
    yield {"cg": cg, "dev": dev, "thread": thread, "driver": driver, "database": database, "faiss": faiss,
           "model": SentenceTransformer("all-MiniLM-L6-v2", device="cpu")}
    driver.close()


def _run(env, require_answer: bool):
    from llm.common.leakage import LeakageContext, default_mode

    index, ids_arr, emb, id_to_row = env["faiss"]
    with tempfile.TemporaryDirectory() as tmp:
        recs, _ = env["cg"].process_sample(
            env["dev"], None, None, env["model"], env["driver"], env["database"], index, ids_arr, emb, id_to_row,
            env["thread"], 5, 3, 3, 400, "none", Path(tmp) / "o.jsonl", retrieval_only=True,
            c_retrieval_version="v3", alpha=0.5, leakage=LeakageContext.from_mode(default_mode()),
            expansion_require_answer=require_answer)
    return recs


@pytest.fixture(scope="module")
def runs(env):
    return {"on": _run(env, True), "on_again": _run(env, True), "off": _run(env, False)}


def test_every_expansion_question_has_a_non_excluded_answer(env, runs):
    from llm.common.leakage import LeakageContext, default_mode

    ctx = LeakageContext.from_mode(default_mode())
    acc = dict(zip(env["dev"]["Id"].astype(int), env["dev"]["AcceptedAnswerId"].astype(int)))
    for r in runs["on"]:
        q = r["question_id"]
        ex = ctx.exclusion_for(q, env["thread"].get(q, []), acc[q])
        assert r["expansion_qids_empty"] == []
        assert r["n_expansion_answerable"] == len(r["expansion_qids"])
        if r["expansion_qids"]:
            with env["driver"].session(database=env["database"]) as s:
                ok = {rec["q"] for rec in s.run(
                    "UNWIND $qs AS q MATCH (:Question {id: q})-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer) "
                    "WHERE NOT a.id IN $ex RETURN DISTINCT q", qs=r["expansion_qids"], ex=list(ex.answer_ids))}
            assert ok == set(r["expansion_qids"])
        assert not any(ex.excludes(i["question_id"], i["answer_id"]) for i in r["retrieved_context"])


def test_deterministic_across_runs(runs):
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("retrieval_latency_sec",)} for r in rs]  # noqa: E731
    assert json.dumps(strip(runs["on"]), sort_keys=True, default=str) == \
        json.dumps(strip(runs["on_again"]), sort_keys=True, default=str)


def test_only_the_expansion_stage_changes(runs):
    """Anchors and the traversal candidate pool are identical with the switch on or off."""
    off = {r["question_id"]: r for r in runs["off"]}
    for r in runs["on"]:
        o = off[r["question_id"]]
        assert r["anchor_question_ids"] == o["anchor_question_ids"]
        assert r["graph_candidate_answer_ids"] == o["graph_candidate_answer_ids"]


def test_switch_off_reproduces_pre_refactor_snapshot(question_faiss):
    expected = json.loads(snapshot.FIXTURE.read_text())
    actual = json.loads(json.dumps(snapshot.take_snapshot(expansion_require_answer=False, faiss_cache=question_faiss), sort_keys=True))
    for label in expected["configs"]:
        assert actual["configs"][label] == expected["configs"][label], label


def test_switch_on_matches_its_own_snapshot_and_keeps_anchor_traversal(question_faiss):
    """New snapshot (separate file) is reproduced; vs the old snapshot, anchors and traversal
    candidates are unchanged -- only expansion (and so the fused top-5) may differ."""
    from llm.common.leakage import default_mode

    new = json.loads(snapshot.FIXTURE_EXPANSION_ANSWERED.read_text())
    actual = json.loads(json.dumps(snapshot.take_snapshot(leakage_mode=default_mode(), expansion_require_answer=True,
                                                          with_stages=True, faiss_cache=question_faiss), sort_keys=True))
    assert actual == new
    old_stages = json.loads(json.dumps(snapshot.take_snapshot(leakage_mode=default_mode(), expansion_require_answer=False,
                                                              with_stages=True, faiss_cache=question_faiss), sort_keys=True))
    for label in new["configs"]:
        for a, b in zip(new["configs"][label], old_stages["configs"][label]):
            assert a["anchor_question_ids"] == b["anchor_question_ids"]
            assert a["graph_candidate_answer_ids"] == b["graph_candidate_answer_ids"]
            assert a["expansion_qids_empty"] == []
