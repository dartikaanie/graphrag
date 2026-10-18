"""
build_exclusion_sets.py
=====================================
Offline precompute of the per-question leakage exclusion sets
(llm/common/leakage.py) for the whole sampling pool, written to
results/leakage/exclusion_sets.parquet -- one row per (question_id, mode):

    question_id, mode, excluded_q_ids, excluded_a_ids, n_dup, n_linked, n_neardup

excluded_q_ids = {q} ∪ related questions of the mode; excluded_a_ids = every
answer of those questions (+ q's thread answers from the answers parquet, the
source C/D already use). n_dup / n_linked / n_neardup count the related
questions of each kind INCLUDED by that mode (0 when the mode leaves the kind
out). Rows are sorted by (mode, question_id) and id lists are sorted, so the
file is byte-identical across runs on the same data.

Sources (read-only):
  - pool: the shared sampler (get_candidate_questions + filter_by_token_limit,
    oversample_pool = MAX_PLANNED_N_SAMPLE * 4 = 1536, seed 42) -- test and dev
    questions alike; only ids are used.
  - Duplicate / Linked: IS_RELATED_TO {link_type}, either direction (Neo4j).
  - near-duplicates: C's question embeddings (Title + " " + Body[:1000],
    11_densify_embedding_similarity.py) and its IVF index -- the index
    anchor_via_vector_search() uses; no new index is built. Candidates from
    the IVF search (--near-dup-top-n, --nprobe) are re-scored exactly with the
    stored vectors; cosine >= threshold, q itself excluded.
  - answers of every involved question: HAS_ANSWER | HAS_ACCEPTED_ANSWER (Neo4j).

Condition B never runs this (it has no Neo4j at runtime); it only reads the parquet.

    python llm/leakage/build_exclusion_sets.py            # default modes, full pool
    python llm/leakage/build_exclusion_sets.py --question-ids 123 456   # add questions outside the pool
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from llm.common.question_index import query_vector, read_question_index  # noqa: E402
from llm.common.leakage import (DEFAULT_EXCLUSION_SETS_PATH, DEFAULT_NEAR_DUP_COSINE, MODE_THREAD,  # noqa: E402
                                leakage_mode)

MAX_PLANNED_N_SAMPLE = 384
OVERSAMPLE_POOL = MAX_PLANNED_N_SAMPLE * 4
SEED = 42
EMBED_DIM = 384
DEFAULT_MODES = (
    MODE_THREAD,
    leakage_mode(True, False, None),                      # dup
    leakage_mode(True, False, DEFAULT_NEAR_DUP_COSINE),    # dup+neardup0.95 (default)
    leakage_mode(True, True, DEFAULT_NEAR_DUP_COSINE),     # dup+linked+neardup0.95
)


def parse_mode(mode: str) -> dict:
    """mode string -> {"dup", "linked", "neardup"} (neardup: threshold or None)."""
    if mode == MODE_THREAD:
        return {"dup": False, "linked": False, "neardup": None}
    parts = mode.split("+")
    if parts[0] != "dup":
        raise ValueError(f"unsupported mode '{mode}'")
    out = {"dup": True, "linked": False, "neardup": None}
    for p in parts[1:]:
        if p == "linked":
            out["linked"] = True
        elif p.startswith("neardup"):
            out["neardup"] = float(p[len("neardup"):])
        else:
            raise ValueError(f"unsupported mode part '{p}' in '{mode}'")
    if leakage_mode(True, out["linked"], out["neardup"]) != mode:
        raise ValueError(f"non-canonical mode '{mode}'")
    return out


def compute_rows(question_ids, modes, thread_answers: dict, neighbors: dict, near_dups: dict,
                 answers_of: dict) -> list[dict]:
    """Pure assembly of the parquet rows.

    thread_answers: q -> answer ids of q's thread (any source)
    neighbors:      q -> [(other_qid, link_type)]  (IS_RELATED_TO, either direction)
    near_dups:      q -> [(other_qid, cosine)]     (q itself never listed)
    answers_of:     question id -> its answer ids (KG)
    """
    rows = []
    for mode in modes:
        spec = parse_mode(mode)
        for q in sorted(int(x) for x in question_ids):
            dup = {o for o, t in neighbors.get(q, []) if t == "Duplicate" and o != q} if spec["dup"] else set()
            linked = {o for o, t in neighbors.get(q, []) if t == "Linked" and o != q} if spec["linked"] else set()
            neardup = ({o for o, c in near_dups.get(q, []) if c >= spec["neardup"] and o != q}
                       if spec["neardup"] is not None else set())
            ex_q = {q} | dup | linked | neardup
            ex_a = set(thread_answers.get(q, [])) | set(answers_of.get(q, []))
            for o in ex_q - {q}:
                ex_a |= set(answers_of.get(o, []))
            rows.append({
                "question_id": q, "mode": mode,
                "excluded_q_ids": sorted(ex_q), "excluded_a_ids": sorted(int(a) for a in ex_a),
                "n_dup": len(dup), "n_linked": len(linked), "n_neardup": len(neardup),
            })
    rows.sort(key=lambda r: (r["mode"], r["question_id"]))
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("question_id", pa.int64()), ("mode", pa.string()),
        ("excluded_q_ids", pa.list_(pa.int64())), ("excluded_a_ids", pa.list_(pa.int64())),
        ("n_dup", pa.int32()), ("n_linked", pa.int32()), ("n_neardup", pa.int32()),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), tmp, compression="zstd")
    tmp.replace(path)


# ---------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------

def load_pool(questions_parquet: str, answers_parquet: str, seed: int = SEED,
              oversample_pool: int = OVERSAMPLE_POOL):
    import duckdb

    from llm.b_rag import b_condition_b_rag as b

    b.log = lambda *a, **k: None
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    pool = b.filter_by_token_limit(b.get_candidate_questions(con, questions_parquet, answers_parquet,
                                                             oversample_pool, seed))
    ids = pool["Id"].astype(int).tolist()
    # thread answers from the answers parquet (the map C/D build their exclusion from)
    df = con.execute(f"""
        SELECT ParentId AS question_id, Id AS answer_id
        FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
              FROM read_parquet('{answers_parquet}') WHERE ParentId IN ({','.join(map(str, ids))}))
        WHERE rn = 1
    """).df()
    con.close()
    thread = {}
    for q, a in zip(df["question_id"], df["answer_id"]):
        thread.setdefault(int(q), []).append(int(a))
    return ids, thread


def fetch_neighbors(driver, database, question_ids) -> dict:
    out = {}
    with driver.session(database=database, default_access_mode="READ") as s:
        for r in s.run("""
            UNWIND $ids AS qid
            MATCH (q:Question {id: qid})-[r:IS_RELATED_TO]-(o:Question)
            RETURN qid, o.id AS other, r.link_type AS link_type
            ORDER BY qid, other, link_type
        """, ids=[int(q) for q in question_ids]):
            out.setdefault(int(r["qid"]), []).append((int(r["other"]), r["link_type"]))
    return out


def fetch_answers(driver, database, question_ids) -> dict:
    out = {}
    with driver.session(database=database, default_access_mode="READ") as s:
        for r in s.run("""
            UNWIND $ids AS qid
            MATCH (q:Question {id: qid})-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer)
            RETURN qid, a.id AS aid ORDER BY qid, aid
        """, ids=sorted({int(q) for q in question_ids})):
            out.setdefault(int(r["qid"]), []).append(int(r["aid"]))
    return out


def find_near_dups(question_ids, kg_workspace_dir: Path, threshold: float, top_n: int = 100,
                   nprobe: int = 32, query_texts: dict | None = None) -> dict:
    """q -> [(other_qid, exact cosine)] with cosine >= threshold, via C's IVF
    question index; candidates re-scored with the stored (L2-normalized)
    vectors. A question missing from the embedding cache is encoded on CPU
    from query_texts[q] (build_query_text), as C does."""
    import faiss

    n_total = json.loads((kg_workspace_dir / "embed_checkpoint.json").read_text())["next_row"]
    ids_arr = np.load(kg_workspace_dir / "question_embeddings_ivf_ids.npy")
    emb = np.memmap(kg_workspace_dir / "question_embeddings.f32", dtype="float32", mode="r", shape=(n_total, EMBED_DIM))
    index = read_question_index(kg_workspace_dir / "question_embeddings_ivf.faiss")  # STOPs unless nprobe == 32
    faiss.extract_index_ivf(index).nprobe = nprobe
    id_to_row = {int(q): i for i, q in enumerate(ids_arr)}
    model = None
    out = {}
    for q in sorted(int(x) for x in question_ids):
        if q in id_to_row:
            qv = query_vector(emb, id_to_row[q])
        else:
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            from llm.b_rag.kg_corpus import encode_query

            qv = encode_query(model, (query_texts or {})[q])
        _, rows = index.search(qv, top_n)
        found = []
        for row in rows[0]:
            if row == -1:
                continue
            other = int(ids_arr[row])
            if other == q:
                continue
            cos = float(np.dot(qv[0], emb[row]))
            if cos >= threshold:
                found.append((other, round(cos, 6)))
        out[q] = sorted(found)
    return out


def main():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--question-ids", nargs="*", type=int, default=[], help="extra questions outside the pool")
    parser.add_argument("--output", default=str(DEFAULT_EXCLUSION_SETS_PATH))
    parser.add_argument("--near-dup-top-n", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=32, help="IVF nprobe for the near-dup search (script 11's default)")
    args = parser.parse_args()
    specs = [parse_mode(m) for m in args.modes]
    t0 = time.time()

    pool, thread = load_pool(os.environ["QUESTIONS_PARQUET"], os.environ["ANSWERS_PARQUET"])
    question_ids = sorted(set(pool) | set(args.question_ids))
    print(f"[pool] {len(pool)} pool questions + {len(set(args.question_ids) - set(pool))} extra")

    from neo4j import GraphDatabase

    pw = (os.getenv("NEO4J_PASSWORD") or "").strip().strip('"').strip("'")
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687").strip(),
                                  auth=(os.getenv("NEO4J_USER", "neo4j").strip(), pw))
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    try:
        neighbors = fetch_neighbors(driver, database, question_ids)
        thresholds = [s["neardup"] for s in specs if s["neardup"] is not None]
        near_dups = (find_near_dups(question_ids, Path(os.environ["KG_WORKSPACE_DIR"]), min(thresholds),
                                    args.near_dup_top_n, args.nprobe) if thresholds else {})
        involved = set(question_ids) | {o for v in neighbors.values() for o, _ in v} | {o for v in near_dups.values() for o, _ in v}
        answers_of = fetch_answers(driver, database, involved)
    finally:
        driver.close()

    rows = compute_rows(question_ids, args.modes, thread, neighbors, near_dups, answers_of)
    out = Path(args.output)
    write_rows(rows, out)
    import hashlib

    manifest = {
        "modes": args.modes, "n_questions": len(question_ids), "n_pool": len(pool),
        "pool": {"oversample_pool": OVERSAMPLE_POOL, "seed": SEED, "sampler": "get_candidate_questions + filter_by_token_limit"},
        "near_dup": {"top_n": args.near_dup_top_n, "nprobe": args.nprobe, "index": "question_embeddings_ivf.faiss (C)",
                     "rescored": "exact inner product of stored L2-normalized vectors"},
        "neo4j_database": database, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - t0, 1),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(manifest, indent=2))
    for m in args.modes:
        mr = [r for r in rows if r["mode"] == m]
        print(f"[{m}] q with dup>0: {sum(r['n_dup'] > 0 for r in mr)}, linked>0: {sum(r['n_linked'] > 0 for r in mr)}, "
              f"neardup>0: {sum(r['n_neardup'] > 0 for r in mr)} / {len(mr)}")
    print(f"wrote {out} (sha256 {manifest['sha256'][:12]})")


if __name__ == "__main__":
    main()
