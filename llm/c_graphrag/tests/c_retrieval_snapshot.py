"""Snapshot of Condition C's retrieval on dev questions, used to prove that
the shared-module refactor (llm/common/: query text, doc truncation,
leakage exclusion -- reports/b_corpus_parity.md) did not change C.

`take_snapshot()` runs C's own process_sample(retrieval_only=True) on the
first N dev questions (positions 385.., never the test split) under
several configs, and records every text C passes to embed_query() plus the
retrieved context (ids, scores, sha256 of chunk_text). The fixture in
fixtures/c_retrieval_snapshot_dev.json was produced with the code BEFORE
the refactor; test_c_behavior_unchanged.py re-runs it after.

    python llm/c_graphrag/tests/c_retrieval_snapshot.py --write   # regenerate (pre-refactor only)

Needs Neo4j + the parquet/KG workspace from .env -- read-only.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
C_DIR = REPO_ROOT / "llm" / "c_graphrag"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "c_retrieval_snapshot_dev.json"
# Amendment 9 snapshot (EXPANSION_REQUIRE_ANSWER=True, default leakage mode) -- a separate file;
# the pre-refactor FIXTURE above is never overwritten by it
FIXTURE_EXPANSION_ANSWERED = FIXTURE.with_name("c_retrieval_snapshot_dev_expansion_answered.json")
N_DEV = 12
SEED = 42
OVERSAMPLE_POOL = 384 * 4
DEV_OFFSET = 385

# (label, process_sample kwargs, use precomputed question embeddings)
CONFIGS = [
    ("v2_precomputed", dict(c_retrieval_version="v2"), True),
    ("v2_embed_query", dict(c_retrieval_version="v2"), False),
    ("v3_a0.5_precomputed", dict(c_retrieval_version="v3", alpha=0.5), True),
    ("v3_a0.5_embed_query", dict(c_retrieval_version="v3", alpha=0.5), False),
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def take_snapshot(n_dev: int = N_DEV, leakage_mode: str | None = None,
                  expansion_require_answer: bool = False, with_stages: bool = False,
                  faiss_cache: tuple | None = None) -> dict:
    """`leakage_mode` (llm/common/leakage.py): None = process_sample()'s
    default (MODE_THREAD, the exclusion the fixture was taken with).
    `expansion_require_answer` (Amendment 9): False = the expansion the
    fixture was taken with. `with_stages`: also record each question's
    anchor ids, traversal-candidate answer ids and expansion Questions (not
    part of the original fixture's shape). `faiss_cache`: the tuple
    cg.load_faiss_cache() returns, already loaded -- tests pass the one
    session-wide copy (conftest.py) so a pytest process never holds the
    ~4.15 GB question index twice; None loads it here (CLI use)."""
    for p in (REPO_ROOT, C_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    import duckdb
    from sentence_transformers import SentenceTransformer

    import c_graphrag as cg
    from llm.common import item_provenance as ip

    cg.log = lambda *a, **k: None
    qp, ap = os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"]
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    cand = cg.filter_by_token_limit(cg.get_candidate_questions(con, qp, ap, OVERSAMPLE_POOL, SEED))
    sample_df = cg.sample_questions_split(cand, n_dev, SEED, split="dev", dev_offset=DEV_OFFSET)
    answers_df = cg.get_accepted_answers(con, ap, sample_df["AcceptedAnswerId"].dropna().unique().tolist())
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
    all_answer_ids_map = cg.get_all_answer_ids_for_questions(con, ap, sample_df["Id"].astype(int).tolist())

    driver, database = cg.connect_neo4j(lambda *a, **k: None)
    if faiss_cache is None:
        faiss_cache = cg.load_faiss_cache(Path(os.environ["KG_WORKSPACE_DIR"]), lambda *a, **k: None)
    index, ids_arr, embeddings, id_to_row = faiss_cache
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    from llm.common.leakage import LeakageContext

    leakage = LeakageContext.from_mode(leakage_mode) if leakage_mode else None

    original_embed_query = cg.embed_query
    out = {"dev_question_ids": sample_df["Id"].astype(int).tolist(), "configs": {}}
    try:
        for label, kwargs, precomputed in CONFIGS:
            texts_by_q: dict[int, list[str]] = {}
            current = {"qid": None}

            def recording_embed_query(text, sbert_model):
                texts_by_q.setdefault(current["qid"], []).append(text)
                return original_embed_query(text, sbert_model)

            cg.embed_query = recording_embed_query
            records = []
            with tempfile.TemporaryDirectory() as tmp:
                for _, row in sample_df.iterrows():
                    current["qid"] = int(row["Id"])
                    recs, _ = cg.process_sample(
                        sample_df[sample_df["Id"] == row["Id"]], None, None, model, driver, database,
                        index, ids_arr, embeddings, id_to_row if precomputed else {}, all_answer_ids_map,
                        5, 3, 3, 400, "none", Path(tmp) / "out.jsonl", retrieval_only=True, leakage=leakage,
                        expansion_require_answer=expansion_require_answer, **kwargs,
                    )
                    records += recs
            out["configs"][label] = [{
                "question_id": r["question_id"],
                "embed_query_texts_sha256": [_sha(t) for t in texts_by_q.get(r["question_id"], [])],
                "first_embed_query_text": (texts_by_q.get(r["question_id"]) or [None])[0],
                # items as they were before per-item provenance (llm/common/item_provenance.py): the
                # provenance keys are additive, so the pre-provenance fixtures are compared without them
                "retrieved": [{k: v for k, v in ip.legacy_view(item, drop_rank=label.startswith("v2")).items()
                               if k != "chunk_text"} | {"chunk_text_sha256": _sha(item["chunk_text"])}
                              for item in r["retrieved_context"]],
                "candidate_pool_size": r["candidate_pool_size"],
                **({"anchor_question_ids": r["anchor_question_ids"],
                    "graph_candidate_answer_ids": r["graph_candidate_answer_ids"],
                    "expansion_qids": r["expansion_qids"], "expansion_qids_empty": r["expansion_qids_empty"]}
                   if with_stages else {}),
            } for r in records]
    finally:
        cg.embed_query = original_embed_query
        driver.close()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite the committed fixture")
    parser.add_argument("--write-expansion-answered", action="store_true",
                        help=f"write the Amendment 9 snapshot to {FIXTURE_EXPANSION_ANSWERED.name} (old fixture untouched)")
    args = parser.parse_args()
    if args.write_expansion_answered:
        sys.path.insert(0, str(REPO_ROOT))
        from llm.common.leakage import default_mode

        snap = take_snapshot(leakage_mode=default_mode(), expansion_require_answer=True, with_stages=True)
        FIXTURE_EXPANSION_ANSWERED.write_text(json.dumps(snap, indent=1, sort_keys=True))
        print(f"wrote {FIXTURE_EXPANSION_ANSWERED}")
        sys.exit(0)
    snap = take_snapshot()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(snap, indent=1, sort_keys=True))
        print(f"wrote {FIXTURE}")
    else:
        print(json.dumps(snap, indent=1)[:2000])
