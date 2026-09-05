"""
10_densify_tag_cooccurrence.py
================================
Densifikasi Knowledge Graph (Kondisi C) — TAMBAH edge TAG_COOCCUR
(Question-Question) berdasarkan Jaccard similarity atas tag SHARED,
ke KG yang SUDAH ADA di Neo4j, TANPA menyentuh Question/Answer/Tag/
HAS_ANSWER/HAS_ACCEPTED_ANSWER/TAGGED_WITH/IS_RELATED_TO/AUTHOR_TRUST
yang sudah dibangun oleh 7_build_knowledge_graph.py.

LATAR BELAKANG
--------------
91,96% Question node terisolasi dari IS_RELATED_TO (graf PostLinks terlalu
jarang untuk multi-hop traversal GraphRAG yang bermakna). Skrip ini adalah
sumber densifikasi PERTAMA dari dua yang direncanakan (lihat juga
11_densify_embedding_similarity.py untuk sumber KEDUA).

SUMBER DATA -- PENTING
------------------------
Tag per-Question TIDAK di-re-parse dari Tags parquet -- ditarik LANGSUNG
dari edge TAGGED_WITH yang sudah ada di Neo4j (MATCH (q)-[:TAGGED_WITH]->(t)).
Ini menjamin konsistensi 100% dengan graf aktual (tidak ada risiko mismatch
scope antara parquet vs Neo4j kalau 7_build_knowledge_graph.py pernah
dijalankan dengan filter --tags/--limit di masa lalu, meski saat ini
sudah full-build).

METODOLOGI
----------
1. Tarik semua edge TAGGED_WITH (question_id, tag_name) dari Neo4j.
2. Hitung document frequency (df) tiap tag = jumlah Question unik yang
   memakainya.
3. "Tag informatif" = tag dengan df <= --max-df (default 10.000) --
   analog stopword TF-IDF, mencegah tag mega-populer (mis. 'python',
   'javascript') memicu combinatorial explosion candidate pair.
4. Untuk tiap pasang Question yang share >=1 tag informatif, hitung
   Jaccard = |tag_A ∩ tag_B| / |tag_A ∪ tag_B| (union/intersection
   dihitung HANYA atas himpunan tag informatif -- tag mega-populer yang
   dibuang di langkah 3 tidak ikut menghitung union/intersection).
5. Filter Jaccard >= --jaccard-threshold (default 0.3).
6. Per Question (sebagai node SUMBER), ranking tetangga by Jaccard desc,
   cap top --top-k (default 20) -- EDGE DIRECTED (q1 -> q2), bukan
   gabungan dua sisi. Ini konsisten dengan makna "outgoing edge" di
   target validasi densifikasi (lihat 12_validate_densification.py) dan
   dengan pola IS_RELATED_TO yang juga directional.
7. Tulis edge TAG_COOCCUR ke Neo4j (MERGE, idempotent, batched UNWIND --
   pola sama dengan write_batched() di 7_build_knowledge_graph.py).

SAFETY CHECK -- UKURAN CANDIDATE PAIR
----------------------------------------
Self-join tag co-occurrence bisa meledak kalau --max-df kurang ketat
(mis. tag dengan df=9999 sendirian menghasilkan ~9999*9998 pasangan
directed). SEBELUM full join dijalankan, skrip menghitung estimasi upper-
bound jumlah pasangan (SUM(df*(df-1)) per tag informatif) dan ABORT kalau
melebihi --max-candidate-pairs (default 500 juta), kecuali eksplisit pakai
--confirm-large-join. Kalau estimasi terlalu besar: turunkan --max-df,
BUKAN --confirm-large-join begitu saja (bisa OOM/berjam-jam di RAM 8GB).

INSTALL DEPENDENCY
-------------------
    pip install neo4j duckdb pandas python-dotenv

SETUP .env (root project -- NEO4J_* sudah ada di sana, uncomment kalau perlu)
----------------------------------------------------------------------------
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j

CARA PAKAI
----------
    # Cek dulu tanpa nulis ke Neo4j (lihat estimasi & angka Jaccard)
    python 10_densify_tag_cooccurrence.py --dry-run

    # Run sesungguhnya dengan default sesuai spesifikasi penelitian
    python 10_densify_tag_cooccurrence.py

    # Kalau safety check abort, coba max-df lebih ketat dulu
    python 10_densify_tag_cooccurrence.py --max-df 5000 --dry-run
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
DEFAULT_MAX_DF = 10_000
DEFAULT_JACCARD_THRESHOLD = 0.3
DEFAULT_TOP_K = 20
DEFAULT_TRUST_WEIGHT = 0.6  # "menengah" -- di antara IS_RELATED_TO (1.0/cosine) dan EMBED_SIM (rencana lebih rendah)
DEFAULT_MAX_CANDIDATE_PAIRS = 500_000_000
BATCH_SIZE_DEFAULT = 2000

import logging


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_densify_tag_cooccurrence.log"

    logger = logging.getLogger("densify_tag_cooccurrence")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path


def step(logger, n, title):
    logger.info("=" * 78)
    logger.info(f"STEP {n} - {title}")
    logger.info("=" * 78)


# ---------------------------------------------------------------------
# Neo4j connection (pola sama dengan 4_test_infra.py / 7_build_knowledge_graph.py)
# ---------------------------------------------------------------------

def connect_neo4j(logger):
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    if password:
        password = password.strip().strip('"').strip("'")
    if not password:
        raise ValueError("NEO4J_PASSWORD tidak ditemukan di .env (root project)")

    logger.info(f"Neo4j URI: {uri} | User: {user} | Database: {database}")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    logger.info("Terhubung ke Neo4j.")
    return driver, database


def write_batched(driver, database, cypher: str, df: pd.DataFrame, batch_size: int, logger, label: str):
    """Identik write_batched() di 7_build_knowledge_graph.py -- UNWIND batch
    write, konversi ke dict PER BATCH (bukan sekaligus di awal) supaya log
    progress muncul sejak batch pertama & memori puncak kecil."""
    total = len(df)
    if total == 0:
        logger.info(f"  [{label}] tidak ada baris untuk ditulis, dilewati.")
        return

    total_batches = max(1, math.ceil(total / batch_size))
    log_every = max(1, total_batches // 100)

    written = 0
    t0 = time.time()
    with driver.session(database=database) as session:
        for batch_num, i in enumerate(range(0, total, batch_size), start=1):
            batch = df.iloc[i:i + batch_size].to_dict("records")
            session.execute_write(lambda tx, b=batch: tx.run(cypher, rows=b))
            written += len(batch)

            if batch_num % log_every == 0 or written == total:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (total - written) / rate if rate > 0 else float("nan")
                logger.info(
                    f"  [{label}] {written:,}/{total:,} baris tertulis "
                    f"({rate:,.0f} baris/detik, ETA ~{eta / 60:.1f} menit)"
                )


# ---------------------------------------------------------------------
# STEP 1 - Tarik edge TAGGED_WITH (question_id, tag_name) dari Neo4j
# ---------------------------------------------------------------------

def export_tagged_with(driver, database, logger) -> pd.DataFrame:
    with driver.session(database=database) as session:
        count = session.run(
            "MATCH ()-[r:TAGGED_WITH]->() RETURN count(r) AS n"
        ).single()["n"]
        logger.info(f"Edge TAGGED_WITH di Neo4j: {count:,} -- menarik semua ke DuckDB...")

        result = session.run(
            "MATCH (q:Question)-[:TAGGED_WITH]->(t:Tag) RETURN q.id AS question_id, t.name AS tag_name"
        )
        rows = []
        t0 = time.time()
        for i, record in enumerate(result, start=1):
            rows.append((record["question_id"], record["tag_name"]))
            if i % 1_000_000 == 0:
                elapsed = time.time() - t0
                logger.info(f"  ... {i:,} baris ditarik ({i / elapsed:,.0f} baris/detik)")

    df = pd.DataFrame(rows, columns=["question_id", "tag_name"])
    logger.info(f"Total TAGGED_WITH ditarik: {len(df):,}")
    return df


# ---------------------------------------------------------------------
# STEP 2-6 - Hitung Jaccard di DuckDB (tag informatif, candidate pair,
#            safety check, ranking top-K directed)
# ---------------------------------------------------------------------

def compute_tag_cooccurrence(con: duckdb.DuckDBPyConnection, tagged_with: pd.DataFrame,
                              max_df: int, jaccard_threshold: float, top_k: int,
                              max_candidate_pairs: int, confirm_large_join: bool,
                              logger) -> pd.DataFrame:
    con.register("tagged_with_raw", tagged_with)

    step_local = "[tag-freq]"
    con.execute("""
        CREATE OR REPLACE TABLE tag_freq AS
        SELECT tag_name, COUNT(DISTINCT question_id) AS df
        FROM tagged_with_raw
        GROUP BY tag_name
    """)
    n_tags_total = con.execute("SELECT COUNT(*) FROM tag_freq").fetchone()[0]
    n_tags_informative = con.execute(f"SELECT COUNT(*) FROM tag_freq WHERE df <= {max_df}").fetchone()[0]
    logger.info(f"{step_local} Tag total: {n_tags_total:,} | Tag informatif (df<={max_df:,}): "
                f"{n_tags_informative:,} ({n_tags_informative / n_tags_total * 100:.1f}%)")

    con.execute(f"""
        CREATE OR REPLACE TABLE informative_question_tags AS
        SELECT tw.question_id, tw.tag_name
        FROM tagged_with_raw tw
        JOIN tag_freq tf ON tw.tag_name = tf.tag_name
        WHERE tf.df <= {max_df}
    """)
    con.execute("""
        CREATE OR REPLACE TABLE question_tag_count AS
        SELECT question_id, COUNT(*) AS n_tags
        FROM informative_question_tags
        GROUP BY question_id
    """)
    n_q_eligible = con.execute("SELECT COUNT(*) FROM question_tag_count").fetchone()[0]
    logger.info(f"{step_local} Question dengan >=1 tag informatif: {n_q_eligible:,}")

    # --- Safety check: estimasi upper-bound jumlah directed candidate pair ---
    est = con.execute("""
        SELECT SUM(cnt * (cnt - 1))
        FROM (
            SELECT tag_name, COUNT(*) AS cnt
            FROM informative_question_tags
            GROUP BY tag_name
        )
    """).fetchone()[0] or 0
    logger.info(f"[safety-check] Estimasi upper-bound directed candidate pair: {int(est):,} "
                f"(SUM(df*(df-1)) per tag informatif -- overcount kalau ada tag ganda, aman sbg batas atas)")

    if est > max_candidate_pairs and not confirm_large_join:
        raise SystemExit(
            f"[ABORT] Estimasi candidate pair ({int(est):,}) melebihi --max-candidate-pairs "
            f"({max_candidate_pairs:,}). Self-join sebesar ini berisiko OOM/berjam-jam di RAM 8GB.\n"
            f"  -> SARAN: turunkan --max-df (saat ini {max_df:,}) supaya tag lebih ketat difilter, "
            f"lalu coba lagi dengan --dry-run.\n"
            f"  -> Kalau memang mau lanjut apa adanya, pakai --confirm-large-join (TIDAK disarankan "
            f"untuk percobaan pertama)."
        )

    step_local = "[candidate-pairs]"
    logger.info(f"{step_local} Menjalankan self-join (bisa lama tergantung ukuran estimasi di atas)...")
    t0 = time.time()
    con.execute("""
        CREATE OR REPLACE TABLE tag_pairs AS
        SELECT a.question_id AS src, b.question_id AS dst, COUNT(*) AS intersection
        FROM informative_question_tags a
        JOIN informative_question_tags b
          ON a.tag_name = b.tag_name AND a.question_id != b.question_id
        GROUP BY a.question_id, b.question_id
    """)
    n_pairs = con.execute("SELECT COUNT(*) FROM tag_pairs").fetchone()[0]
    logger.info(f"{step_local} Selesai dalam {time.time() - t0:.1f}s -- {n_pairs:,} pasangan directed "
                f"(sebelum filter Jaccard)")

    step_local = "[jaccard]"
    con.execute(f"""
        CREATE OR REPLACE TABLE tag_jaccard AS
        SELECT tp.src, tp.dst, tp.intersection,
               (c1.n_tags + c2.n_tags - tp.intersection) AS union_size,
               CAST(tp.intersection AS DOUBLE) / (c1.n_tags + c2.n_tags - tp.intersection) AS jaccard
        FROM tag_pairs tp
        JOIN question_tag_count c1 ON tp.src = c1.question_id
        JOIN question_tag_count c2 ON tp.dst = c2.question_id
        WHERE CAST(tp.intersection AS DOUBLE) / (c1.n_tags + c2.n_tags - tp.intersection) >= {jaccard_threshold}
    """)
    n_above_threshold = con.execute("SELECT COUNT(*) FROM tag_jaccard").fetchone()[0]
    logger.info(f"{step_local} Pasangan dengan Jaccard >= {jaccard_threshold}: {n_above_threshold:,}")

    step_local = "[top-k]"
    con.execute(f"""
        CREATE OR REPLACE TABLE tag_cooccur_edges AS
        SELECT src AS question_id, dst AS related_question_id, jaccard
        FROM (
            SELECT src, dst, jaccard,
                   ROW_NUMBER() OVER (PARTITION BY src ORDER BY jaccard DESC, dst) AS rnk
            FROM tag_jaccard
        )
        WHERE rnk <= {top_k}
    """)
    n_final = con.execute("SELECT COUNT(*) FROM tag_cooccur_edges").fetchone()[0]
    n_src_with_edge = con.execute("SELECT COUNT(DISTINCT question_id) FROM tag_cooccur_edges").fetchone()[0]
    logger.info(f"{step_local} Edge TAG_COOCCUR final (cap top-{top_k}/node): {n_final:,} "
                f"dari {n_src_with_edge:,} Question sumber unik")

    result = con.execute("SELECT question_id, related_question_id, jaccard FROM tag_cooccur_edges").fetchdf()
    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-df", type=int, default=DEFAULT_MAX_DF,
                         help=f"Tag dengan df > nilai ini dibuang dari inverted index (default {DEFAULT_MAX_DF:,})")
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD,
                         help=f"Ambang Jaccard minimum (default {DEFAULT_JACCARD_THRESHOLD})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                         help=f"Cap tetangga per node (default {DEFAULT_TOP_K})")
    parser.add_argument("--trust-weight", type=float, default=DEFAULT_TRUST_WEIGHT,
                         help=f"Bobot trust kategorikal utk edge TAG_COOCCUR (default {DEFAULT_TRUST_WEIGHT}, "
                              f"'menengah' -- di antara IS_RELATED_TO dan EMBED_SIM)")
    parser.add_argument("--max-candidate-pairs", type=int, default=DEFAULT_MAX_CANDIDATE_PAIRS,
                         help=f"Safety check: abort kalau estimasi candidate pair melebihi ini "
                              f"(default {DEFAULT_MAX_CANDIDATE_PAIRS:,})")
    parser.add_argument("--confirm-large-join", action="store_true",
                         help="Lewati safety check ukuran candidate pair (TIDAK disarankan)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--memory-limit", default="2GB",
                         help="Batas memori DuckDB, sama konvensi dgn 7_build_knowledge_graph.py")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                         help="Hitung semua edge, TIDAK menulis ke Neo4j")
    args = parser.parse_args()

    logger, log_path = setup_logging()
    logger.info("DENSIFIKASI KG -- TAG_COOCCUR (Jaccard similarity)")
    logger.info(f"Log file: {log_path}")
    logger.info(f"[config] max_df={args.max_df:,} jaccard_threshold={args.jaccard_threshold} "
                f"top_k={args.top_k} trust_weight={args.trust_weight}")
    t_start = time.time()

    workspace_dir = Path("_kg_workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = workspace_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = workspace_dir / "_workspace_tag_cooccur.duckdb"

    con = duckdb.connect(database=str(db_path))
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    logger.info(f"[config] memory_limit={args.memory_limit} threads={args.threads} "
                f"temp_directory={tmp_dir} workspace_db={db_path}")

    step(logger, 1, "Koneksi Neo4j")
    driver, database = connect_neo4j(logger)

    try:
        step(logger, 2, "Tarik edge TAGGED_WITH dari Neo4j")
        tagged_with = export_tagged_with(driver, database, logger)

        step(logger, 3, "Hitung tag co-occurrence (DuckDB: tag informatif -> candidate pair -> Jaccard -> top-K)")
        edges = compute_tag_cooccurrence(
            con, tagged_with, args.max_df, args.jaccard_threshold, args.top_k,
            args.max_candidate_pairs, args.confirm_large_join, logger,
        )

        if args.dry_run:
            logger.info("--dry-run aktif: TIDAK menulis ke Neo4j. Selesai.")
            logger.info(f"Contoh 10 edge teratas by Jaccard:\n"
                        f"{edges.sort_values('jaccard', ascending=False).head(10).to_string(index=False)}")
            return

        step(logger, 4, "Tulis edge TAG_COOCCUR ke Neo4j")
        edges = edges.assign(trust_weight=args.trust_weight)
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (q1:Question {id: row.question_id}), (q2:Question {id: row.related_question_id})
            MERGE (q1)-[r:TAG_COOCCUR]->(q2)
            SET r.weight = row.trust_weight, r.jaccard = row.jaccard
            """,
            edges, args.batch_size, logger, "TAG_COOCCUR",
        )

        step(logger, 5, "Verifikasi -- hitung ulang edge TAG_COOCCUR langsung dari Neo4j")
        with driver.session(database=database) as session:
            n = session.run("MATCH ()-[r:TAG_COOCCUR]->() RETURN count(r) AS n").single()["n"]
            logger.info(f"  Neo4j TAG_COOCCUR: {n:,}")

    finally:
        driver.close()

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info(f"SELESAI dalam {elapsed / 60:.1f} menit")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
