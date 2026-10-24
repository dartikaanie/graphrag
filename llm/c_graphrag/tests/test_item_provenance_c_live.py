"""Acceptance check for C's item provenance on the REAL KG (dev split only, read-only):
for a handful of dev questions, every retrieved item validates against the
schema, every 2-hop item has a 3-step path anchor -> q2 -> answer whose anchor
is one of the record's anchors, every semantic-expansion item has a non-null
seed in path[0], and EVERY edge of every path (and alt_path) exists in Neo4j
with the recorded weight. Reads Neo4j + parquet + KG workspace: opt-in with
GRAPHRAG_INTEGRATION=1."""

import os
import tempfile
from pathlib import Path

import pytest

from llm.common import item_provenance as ip

pytestmark = pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1",
                                reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
REPO_ROOT = Path(__file__).resolve().parents[3]
N_DEV = 8


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
    dev = cg.sample_questions_split(pool, N_DEV, 42, split="dev", dev_offset=385)
    answers = cg.get_accepted_answers(con, ap, dev["AcceptedAnswerId"].dropna().unique().tolist())
    dev = dev.merge(answers, on="AcceptedAnswerId", how="left").dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
    thread = cg.get_all_answer_ids_for_questions(con, ap, dev["Id"].astype(int).tolist())
    driver, database = cg.connect_neo4j(lambda *a, **k: None)
    yield {"cg": cg, "dev": dev, "thread": thread, "driver": driver, "database": database, "faiss": question_faiss,
           "model": SentenceTransformer("all-MiniLM-L6-v2", device="cpu")}
    driver.close()


def _records(env, **kwargs):
    from llm.common.leakage import LeakageContext, default_mode

    index, ids_arr, emb, id_to_row = env["faiss"]
    with tempfile.TemporaryDirectory() as tmp:
        recs, _ = env["cg"].process_sample(
            env["dev"], None, None, env["model"], env["driver"], env["database"], index, ids_arr, emb, id_to_row,
            env["thread"], 5, 3, 3, 400, "none", Path(tmp) / "o.jsonl", retrieval_only=True,
            leakage=LeakageContext.from_mode(default_mode()), expansion_require_answer=True, **kwargs)
    return recs


@pytest.mark.parametrize("kwargs", [dict(c_retrieval_version="v2"), dict(c_retrieval_version="v3", alpha=0.5)],
                         ids=["v2", "v3_alpha0.5"])
def test_c_paths_are_complete_and_exist_in_the_kg(env, kwargs, capsys):
    lookup = ip.neo4j_edge_lookup(env["driver"], env["database"])
    stats = {"items": 0, "hop1": 0, "hop2": 0, "semantic": 0, "alt_paths": 0, "edges_checked": 0}
    for rec in _records(env, **kwargs):
        assert ip.validate_record(rec) == [], rec["question_id"]
        anchors = {a if isinstance(a, int) else a["question_id"] for a in rec["anchor_question_ids"]}
        for item in rec["retrieved_context"]:
            stats["items"] += 1
            assert ip.validate_item_in_kg(item, lookup) == [], (rec["question_id"], item["answer_id"])
            for path in [item["path"]] + [a["path"] for a in item["alt_paths"]]:
                stats["edges_checked"] += len(path) - 1
            stats["alt_paths"] += len(item["alt_paths"])
            if item["source_stage"] == "graph_traversal":
                assert item["path"][0]["node_id"] == item["anchor_id"] in anchors
                if item["hop"] == 2:
                    stats["hop2"] += 1
                    assert len(item["path"]) == 3
                else:
                    stats["hop1"] += 1
                    assert len(item["path"]) == 2
            else:
                stats["semantic"] += 1
                assert item["path"][0]["node_id"] is not None and item["path"][0]["edge_type_to_next"] == "SEMANTIC_SIM"
    with capsys.disabled():
        print(f"\n[live provenance C {kwargs}] {stats}")
    assert stats["items"] > 0
