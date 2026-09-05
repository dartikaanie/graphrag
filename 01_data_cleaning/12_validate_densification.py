"""
12_validate_densification.py
==============================
Validasi hasil densifikasi Knowledge Graph (setelah 10_densify_tag_
cooccurrence.py & 11_densify_embedding_similarity.py dijalankan) terhadap
TARGET VALIDASI yang ditentukan di awal:

  LEVEL KORPUS
    - % Question isolated turun dari 91,96% jadi <=30%
    - mean degree >=3, median degree >=1
    (dihitung atas gabungan edge Question-Question: IS_RELATED_TO +
    TAG_COOCCUR + EMBED_SIM -- outdegree, konsisten dgn desain directed
    top-K per node di kedua skrip densifikasi)

  LEVEL SAMPEL EVALUASI
    - dari top-k hasil vector search untuk N pertanyaan evaluasi
      (default 384, seed=42), >=80% dari node hasil retrieval punya
      minimal 1 edge outgoing
    - rata-rata node terjangkau dalam 2-hop (dari node pertanyaan
      evaluasi) >=5

  NON-FUNGSIONAL
    - Latency traversal 2-hop <=15 detik (NF3) -- diukur per query,
      dilaporkan rata-rata & p95

CATATAN PENTING
-----------------
Skrip ini memakai FAISS index yang di-cache oleh 11_densify_embedding_
similarity.py (--stage edges) sebagai proxy "vector search" -- pipeline
retrieval resmi Kondisi C (poin #2 di rencana penelitian) belum dibangun
saat skrip ini ditulis. Kalau pipeline retrieval resmi sudah ada nanti,
pertimbangkan reuse index yang sama supaya angka validasi konsisten.

Sample evaluasi diambil LANGSUNG dari Neo4j (Question yang punya
HAS_ACCEPTED_ANSWER), BUKAN dari parquet -- supaya skrip ini berdiri
sendiri tanpa dependency ke path parquet. Untuk mereproduksi SAMPLE ID
YANG SAMA PERSIS dengan pilot Kondisi A/B/C nanti, pakai
--question-ids-file (satu Id per baris).

CARA PAKAI
----------
    python 12_validate_densification.py                       # n=384 default
    python 12_validate_densification.py --n-sample 30 --seed 42  # pilot cepat
    python 12_validate_densification.py --question-ids-file eval_ids.txt
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

WORKSPACE_DIR = Path("_kg_workspace")
EMBED_DIM = 384
DENSIFY_REL_TYPES = "IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM"

DEFAULT_N_SAMPLE = 384
DEFAULT_SEED = 42
DEFAULT_TOP_K = 5
TARGET_ISOLATED_PCT = 30.0
TARGET_MEAN_DEGREE = 3.0
TARGET_MEDIAN_DEGREE = 1.0
TARGET_SAMPLE_OUTGOING_PCT = 80.0
TARGET_AVG_2HOP_REACHABLE = 5.0
TARGET_LATENCY_SEC = 15.0


def log(msg):
    print(msg, flush=True)


def connect_neo4j():
    import os
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if password:
        password = password.strip().strip('"').strip("'")
    if not password:
        raise ValueError("NEO4J_PASSWORD tidak ditemukan di .env (root project)")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver, database


# ---------------------------------------------------------------------
# LEVEL KORPUS
# ---------------------------------------------------------------------

def validate_corpus_level(driver, database):
    log("\n" + "=" * 78)
    log("LEVEL KORPUS -- distribusi degree Question (gabungan IS_RELATED_TO + "
        "TAG_COOCCUR + EMBED_SIM)")
    log("=" * 78)

    with driver.session(database=database) as session:
        total_q = session.run("MATCH (q:Question) RETURN count(q) AS n").single()["n"]

        # apoc-free: hitung outdegree per Question lewat OPTIONAL MATCH + count
        result = session.run(
            f"""
            MATCH (q:Question)
            OPTIONAL MATCH (q)-[r:{DENSIFY_REL_TYPES}]->()
            WITH q, count(r) AS deg
            RETURN deg
            """
        )
        degrees = [record["deg"] for record in result]

    degrees = np.array(degrees)
    n_isolated = int((degrees == 0).sum())
    pct_isolated = n_isolated / total_q * 100
    mean_deg = float(degrees.mean())
    median_deg = float(np.median(degrees))

    log(f"Total Question           : {total_q:,}")
    log(f"Question isolated (deg=0): {n_isolated:,} ({pct_isolated:.2f}%) "
        f"[target: <= {TARGET_ISOLATED_PCT}%] {'PASS' if pct_isolated <= TARGET_ISOLATED_PCT else 'FAIL'}")
    log(f"Mean degree              : {mean_deg:.4f} "
        f"[target: >= {TARGET_MEAN_DEGREE}] {'PASS' if mean_deg >= TARGET_MEAN_DEGREE else 'FAIL'}")
    log(f"Median degree            : {median_deg:.4f} "
        f"[target: >= {TARGET_MEDIAN_DEGREE}] {'PASS' if median_deg >= TARGET_MEDIAN_DEGREE else 'FAIL'}")

    return {
        "total_question": total_q,
        "pct_isolated": pct_isolated,
        "mean_degree": mean_deg,
        "median_degree": median_deg,
    }


# ---------------------------------------------------------------------
# Sample pertanyaan evaluasi (langsung dari Neo4j, atau dari file eksplisit)
# ---------------------------------------------------------------------

def get_evaluation_sample(driver, database, n_sample, seed, question_ids_file):
    if question_ids_file:
        ids = [int(line.strip()) for line in Path(question_ids_file).read_text().splitlines() if line.strip()]
        log(f"\n[sample] {len(ids):,} Question id dimuat dari {question_ids_file}")
        return ids

    with driver.session(database=database) as session:
        # rand() Cypher bawaan Neo4j (bukan reservoir-sample deterministic
        # seperti DuckDB) -- cukup untuk validasi ad-hoc. Untuk sample RESMI
        # yang harus identik dengan Kondisi A/B (OVERSAMPLE_POOL, seed DuckDB),
        # gunakan --question-ids-file dari hasil sample_questions() resmi.
        result = session.run(
            """
            MATCH (q:Question)-[:HAS_ACCEPTED_ANSWER]->()
            WITH q, rand() AS r
            ORDER BY r
            LIMIT $n
            RETURN q.id AS id
            """,
            n=n_sample,
        )
        ids = [record["id"] for record in result]
    log(f"\n[sample] {len(ids):,} Question id diambil random dari Neo4j "
        f"(HAS_ACCEPTED_ANSWER) -- BUKAN reservoir-sample seed={seed}, "
        f"pakai --question-ids-file untuk sample resmi yang reproducible.")
    return ids


# ---------------------------------------------------------------------
# LEVEL SAMPEL EVALUASI -- vector search (proxy via FAISS cache dari
# 11_densify_embedding_similarity.py) + outgoing edge check
# ---------------------------------------------------------------------

def load_faiss_cache(logger_log):
    import faiss

    index_path = WORKSPACE_DIR / "question_embeddings_ivf.faiss"
    ids_path = WORKSPACE_DIR / "question_embeddings_ivf_ids.npy"
    emb_path = WORKSPACE_DIR / "question_embeddings.f32"
    ckpt_path = WORKSPACE_DIR / "embed_checkpoint.json"

    if not (index_path.exists() and ids_path.exists() and emb_path.exists()):
        raise SystemExit(
            "[ABORT] Cache FAISS/embedding tidak ditemukan di _kg_workspace/. "
            "Jalankan dulu 11_densify_embedding_similarity.py --stage all "
            "sebelum validasi level sampel."
        )

    import json
    n_total = json.loads(ckpt_path.read_text())["next_row"]
    ids_arr = np.load(ids_path)
    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_total, EMBED_DIM))
    index = faiss.read_index(str(index_path))
    logger_log(f"[faiss] Index & embedding cache dimuat: {index.ntotal:,} vektor")
    return index, ids_arr, embeddings


def validate_sample_level(driver, database, question_ids, top_k):
    log("\n" + "=" * 78)
    log(f"LEVEL SAMPEL EVALUASI -- {len(question_ids):,} pertanyaan, top-{top_k} vector search")
    log("=" * 78)

    index, cached_ids, embeddings = load_faiss_cache(log)
    id_to_row = {int(qid): i for i, qid in enumerate(cached_ids)}

    missing = [qid for qid in question_ids if qid not in id_to_row]
    if missing:
        log(f"[warn] {len(missing):,}/{len(question_ids):,} Question sample tidak ada di embedding "
            f"cache (kemungkinan ditambahkan setelah encoding terakhir) -- dilewati dari retrieval check.")
    question_ids = [qid for qid in question_ids if qid in id_to_row]

    retrieved_node_ids = set()
    per_query_reachable = []
    latencies = []

    with driver.session(database=database) as session:
        for qid in question_ids:
            row = id_to_row[qid]
            q_emb = embeddings[row:row + 1]
            _, indices = index.search(q_emb, top_k + 1)
            neighbor_ids = [int(cached_ids[idx]) for idx in indices[0] if idx != -1 and idx != row][:top_k]
            retrieved_node_ids.update(neighbor_ids)

            t0 = time.time()
            result = session.run(
                f"""
                MATCH (start:Question {{id: $id}})
                CALL {{
                    WITH start
                    MATCH (start)-[:{DENSIFY_REL_TYPES}*1..2]->(n)
                    RETURN DISTINCT n
                }}
                RETURN count(n) AS reachable
                """,
                id=qid,
            )
            reachable = result.single()["reachable"]
            latencies.append(time.time() - t0)
            per_query_reachable.append(reachable)

    if retrieved_node_ids:
        with driver.session(database=database) as session:
            result = session.run(
                f"""
                UNWIND $ids AS qid
                MATCH (q:Question {{id: qid}})
                OPTIONAL MATCH (q)-[r:{DENSIFY_REL_TYPES}]->()
                WITH q, count(r) AS deg
                RETURN deg
                """,
                ids=list(retrieved_node_ids),
            )
            retrieved_degrees = np.array([record["deg"] for record in result])
    else:
        retrieved_degrees = np.array([])

    n_retrieved = len(retrieved_degrees)
    n_with_outgoing = int((retrieved_degrees > 0).sum())
    pct_outgoing = (n_with_outgoing / n_retrieved * 100) if n_retrieved else 0.0

    avg_reachable = statistics.mean(per_query_reachable) if per_query_reachable else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0)

    log(f"\nNode unik hasil top-{top_k} retrieval (gabungan {len(question_ids):,} query): {n_retrieved:,}")
    log(f"  -> punya >=1 edge outgoing: {n_with_outgoing:,} ({pct_outgoing:.2f}%) "
        f"[target: >= {TARGET_SAMPLE_OUTGOING_PCT}%] "
        f"{'PASS' if pct_outgoing >= TARGET_SAMPLE_OUTGOING_PCT else 'FAIL'}")
    log(f"Rata-rata node terjangkau dalam 2-hop (per query): {avg_reachable:.2f} "
        f"[target: >= {TARGET_AVG_2HOP_REACHABLE}] "
        f"{'PASS' if avg_reachable >= TARGET_AVG_2HOP_REACHABLE else 'FAIL'}")
    log(f"Latency traversal 2-hop -- rata-rata: {avg_latency * 1000:.1f}ms, "
        f"p95: {p95_latency * 1000:.1f}ms, MAX: {max(latencies, default=0)*1000:.1f}ms "
        f"[target rata-rata: <= {TARGET_LATENCY_SEC}s] "
        f"{'PASS' if avg_latency <= TARGET_LATENCY_SEC else 'FAIL'}")

    return {
        "n_retrieved_unique": n_retrieved,
        "pct_with_outgoing": pct_outgoing,
        "avg_2hop_reachable": avg_reachable,
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95_latency,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--question-ids-file", default=None,
                         help="Opsional: file 1 Question id per baris, utk sample yang reproducible "
                              "(mis. hasil sample_questions() resmi Kondisi A/B)")
    parser.add_argument("--skip-sample-level", action="store_true",
                         help="Hanya jalankan validasi level korpus (skip FAISS/vector search)")
    args = parser.parse_args()

    driver, database = connect_neo4j()
    try:
        corpus_results = validate_corpus_level(driver, database)

        sample_results = None
        if not args.skip_sample_level:
            question_ids = get_evaluation_sample(driver, database, args.n_sample, args.seed,
                                                   args.question_ids_file)
            sample_results = validate_sample_level(driver, database, question_ids, args.top_k)

        log("\n" + "=" * 78)
        log("RINGKASAN")
        log("=" * 78)
        all_pass = (
            corpus_results["pct_isolated"] <= TARGET_ISOLATED_PCT
            and corpus_results["mean_degree"] >= TARGET_MEAN_DEGREE
            and corpus_results["median_degree"] >= TARGET_MEDIAN_DEGREE
        )
        if sample_results:
            all_pass = all_pass and (
                sample_results["pct_with_outgoing"] >= TARGET_SAMPLE_OUTGOING_PCT
                and sample_results["avg_2hop_reachable"] >= TARGET_AVG_2HOP_REACHABLE
                and sample_results["avg_latency_sec"] <= TARGET_LATENCY_SEC
            )
        log(f"Semua target validasi terpenuhi: {'YA' if all_pass else 'TIDAK'}")
        if not all_pass:
            log("-> Kalau belum tercapai setelah maksimal 2 iterasi penyesuaian threshold "
                "(--jaccard-threshold / --max-df di skrip 10, --cosine-threshold di skrip 11), "
                "terima angka terbaik & dokumentasikan sebagai limitation di Bab VI (sesuai keputusan "
                "metodologis yang sudah ditetapkan).")
    finally:
        driver.close()


if __name__ == "__main__":
    main()