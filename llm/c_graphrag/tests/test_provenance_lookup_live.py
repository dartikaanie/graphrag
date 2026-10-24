"""End to end on the REAL KG for C and D: a 12-question dev run through the real process_sample
(real Neo4j + FAISS retrieval, fake LLM that cites the first and the last retrieved source), then
llm/provenance_lookup.py reads the stored run back by run_id:
  * every cited source gets a path, and every edge of every shown path exists in Neo4j;
  * the file log has one [latency] line per question carrying the record's own numbers
    (llm/latency_report.py reads them back);
  * the record validates against the v3 schema (incl. the not_selected / below_cutoff trace).
Dev split only, read-only. Opt-in: GRAPHRAG_INTEGRATION=1."""

import json
import os
import sys
from pathlib import Path

import pytest

from llm import latency_report
from llm.common import item_provenance as ip
from llm.provenance_lookup import lookup_provenance, render_lookup

pytestmark = pytest.mark.skipif(os.getenv("GRAPHRAG_INTEGRATION") != "1",
                                reason="reads Neo4j; set GRAPHRAG_INTEGRATION=1")
REPO_ROOT = Path(__file__).resolve().parents[3]
N_DEV = 12


@pytest.fixture(scope="module")
def env(question_faiss):
    import duckdb
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    import c_graphrag as cg

    sys.path.insert(0, str(REPO_ROOT / "llm" / "d_lightrag"))
    import d_lightrag as dl

    load_dotenv(REPO_ROOT / ".env")
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    qp, ap = os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"]
    cg.log = lambda *a, **k: None
    pool = cg.filter_by_token_limit(cg.get_candidate_questions(con, qp, ap, 384 * 4, 42))
    dev = cg.sample_questions_split(pool, N_DEV, 42, split="dev", dev_offset=385)
    answers = cg.get_accepted_answers(con, ap, dev["AcceptedAnswerId"].dropna().unique().tolist())
    dev = dev.merge(answers, on="AcceptedAnswerId", how="left").dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
    thread = cg.get_all_answer_ids_for_questions(con, ap, dev["Id"].astype(int).tolist())
    driver, database = cg.connect_neo4j(lambda *a, **k: None)
    yield {"cg": cg, "dl": dl, "dev": dev, "thread": thread, "driver": driver, "database": database,
           "faiss": question_faiss, "model": SentenceTransformer("all-MiniLM-L6-v2", device="cpu")}
    driver.close()


def _fake_llm(client, messages, model):
    labels = [line[4:line.index("]")] for line in messages[-1]["content"].split("\n\n") if line.startswith("[SO-")]
    cited = [labels[0], labels[-1]] if labels else []
    return {"content": "Based on " + " and ".join(f"[SO-{i}]" for i in cited) if cited else "no context"}


def _run(env, cond, tmp_path, monkeypatch):
    from llm.common.leakage import LeakageContext, default_mode

    index, ids_arr, emb, id_to_row = env["faiss"]
    mod = env["cg"] if cond == "C" else env["dl"]
    lines = []
    monkeypatch.setattr(mod, "log", lines.append)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    out = tmp_path / f"condition_{cond.lower()}_live.jsonl"
    run_id = f"{cond * 8}-0000-0000-0000-000000000000"
    leakage = LeakageContext.from_mode(default_mode())
    if cond == "C":
        recs, _ = mod.process_sample(env["dev"], None, _fake_llm, env["model"], env["driver"], env["database"], index,
                                     ids_arr, emb, id_to_row, env["thread"], 5, 3, 3, 400, "fake", out,
                                     c_retrieval_version="v3", alpha=0.5, leakage=leakage,
                                     expansion_require_answer=True, run_id=run_id)
    else:
        recs, _ = mod.process_sample(env["dev"], None, _fake_llm, env["model"], env["driver"], env["database"], index,
                                     ids_arr, emb, id_to_row, env["thread"], 5, 3, 3, 400, "fake", out,
                                     leakage=leakage, run_id=run_id)
    return out, run_id, recs, lines


@pytest.mark.parametrize("cond", ["C", "D"])
def test_lookup_end_to_end_on_12_dev_questions(env, cond, tmp_path, capsys, monkeypatch):
    out, run_id, recs, lines = _run(env, cond, tmp_path, monkeypatch)
    assert len(recs) == N_DEV
    edge_lookup = ip.neo4j_edge_lookup(env["driver"], env["database"])
    stats = {"questions": 0, "cited_sources": 0, "paths_checked": 0, "not_cited": 0, "with_semantic_link": 0}
    for rec in recs:
        assert ip.validate_record(rec) == [], rec["question_id"]
        res = lookup_provenance(run_id, rec["question_id"], search_dirs=[tmp_path])
        assert res["status"] == "ok" and res["condition"] == cond
        assert res["cited"], "the fake LLM cites the first and last retrieved source"
        assert {c["source_id"] for c in res["cited"]} == set(rec["valid_cited_source_ids"])
        for entry in res["cited"]:
            for v in entry["items"]:
                assert v["path_text"].startswith(("Question", "Answer")) and v["path_text"].endswith(f"Answer {v['answer_id']}")
                assert ip.validate_path_in_kg(v["path"], edge_lookup) == []
                stats["paths_checked"] += 1
                stats["with_semantic_link"] += v["has_semantic_link"]
        assert len(res["not_cited"]) + sum(len(c["items"]) for c in res["cited"]) == len(rec["retrieved_context"])
        assert render_lookup(res).splitlines()[-1].startswith("Note: the path shows how the source was retrieved")
        stats["questions"] += 1
        stats["cited_sources"] += len(res["cited"])
        stats["not_cited"] += len(res["not_cited"])
    # latency: one line per question, same numbers as the record
    lat = [m for m in (latency_report.LINE.search(x) for x in lines) if m]
    assert len(lat) == N_DEV and {m["cond"] for m in lat} == {cond}
    by_q = {int(m["qid"]): m for m in lat}
    for rec in recs:
        m = by_q[rec["question_id"]]
        assert float(m["retrieval"]) == pytest.approx(rec["retrieval_latency_sec"], abs=2e-3)
        assert float(m["total"]) == pytest.approx(rec["end_to_end_latency_sec"], abs=2e-3)
    log_file = tmp_path / "run.log"
    log_file.write_text("\n".join(lines))
    with capsys.disabled():
        print(f"\n[live lookup {cond}] {stats}")
        print(latency_report.format_report(latency_report.read_latencies([log_file])))
        first = lookup_provenance(run_id, recs[0]["question_id"], search_dirs=[tmp_path])
        print(render_lookup(first))
