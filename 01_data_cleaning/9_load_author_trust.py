"""
9_load_author_trust.py
========================
Tambahkan User node + edge AUTHOR_TRUST ke Knowledge Graph yang SUDAH ADA
di Neo4j, TANPA mengulang seluruh 7_build_knowledge_graph.py.

KENAPA SCRIPT TERPISAH
------------------------
7_build_knowledge_graph.py mengerjakan semuanya sekaligus (Question, Answer,
Tag, IS_RELATED_TO -- termasuk komputasi embedding cosine similarity yang
paling lama, ~2+ jam). Kalau KG utama sudah berhasil dibangun dan yang
kurang cuma AUTHOR_TRUST (mis. karena users.parquet baru selesai dikonversi
belakangan dari 8_convert_users_xml.py), re-run penuh 7_build_knowledge_graph.py
buang-buang waktu berjam-jam untuk sesuatu yang sebenarnya cepat.

Script ini HANYA:
    1. Ambil semua Answer.id yang SUDAH ADA di Neo4j
    2. Cari OwnerUserId per answer_id dari answers_raw_union.parquet
    3. Load reputation dari users.parquet (atau FilteredUsers.csv fallback)
    4. Hitung bobot AUTHOR_TRUST (formula sama: normalize(log(reputation)))
    5. Tulis User node + edge AUTHOR_TRUST (MERGE, idempotent)

TIDAK menyentuh Question/Answer/Tag/HAS_ANSWER/IS_RELATED_TO yang sudah ada.

CARA PAKAI
----------
    python 9_load_author_trust.py
    python 9_load_author_trust.py --dry-run   # cek angka dulu tanpa nulis ke Neo4j
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")

import logging


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_load_author_trust.log"

    logger = logging.getLogger("load_author_trust")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--users-parquet", default="../00_datasource/merged/users.parquet")
    parser.add_argument("--raw-dir", default="../00_datasource/raw",
                         help="Fallback lokasi FilteredUsers.csv kalau users.parquet tidak ada")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.answers_parquet:
        print("[ERROR] ANSWERS_PARQUET harus diset di .env atau lewat --answers-parquet")
        sys.exit(1)

    logger, log_path = setup_logging()
    logger.info("LOAD AUTHOR_TRUST -- tambahan ke KG yang sudah ada (tanpa full rebuild)")
    logger.info(f"Log file: {log_path}")
    t_start = time.time()

    workspace_dir = Path("_kg_workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = workspace_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = workspace_dir / "_workspace.duckdb"

    con = duckdb.connect(database=str(db_path))
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")

    step(logger, 1, "Koneksi Neo4j + ambil Answer.id yang sudah ada")
    driver, database = connect_neo4j(logger)
    with driver.session(database=database) as session:
        existing_answer_ids = [r["id"] for r in session.run("MATCH (a:Answer) RETURN a.id AS id")]
    logger.info(f"Answer node di Neo4j saat ini: {len(existing_answer_ids):,}")

    if not existing_answer_ids:
        logger.warning("Tidak ada Answer node di Neo4j. Jalankan 7_build_knowledge_graph.py dulu.")
        driver.close()
        return

    step(logger, 2, "Cari OwnerUserId per answer_id dari answers_raw_union.parquet")
    con.register("existing_answers", pd.DataFrame({"id": existing_answer_ids}))
    answer_owners = con.execute(
        f"""
        SELECT a.Id AS answer_id, ANY_VALUE(a.OwnerUserId) AS owner_user_id
        FROM read_parquet('{args.answers_parquet}') a
        JOIN existing_answers e ON a.Id = e.id
        GROUP BY a.Id
        """
    ).fetchdf()
    con.unregister("existing_answers")
    logger.info(f"Answer dengan OwnerUserId ditemukan: {len(answer_owners):,} / {len(existing_answer_ids):,}")

    step(logger, 3, "Load reputation (users.parquet -> fallback FilteredUsers.csv)")
    unique_owner_ids = pd.unique(answer_owners["owner_user_id"].dropna())
    unique_owner_ids = pd.Series([int(i) for i in unique_owner_ids])

    users_parquet = Path(args.users_parquet)
    users_csv = Path(args.raw_dir) / "additional-metadata" / "FilteredUsers.csv"

    con.register("scope_users", pd.DataFrame({"id": unique_owner_ids}))
    if users_parquet.exists():
        logger.info(f"Sumber reputation: {users_parquet} (dump penuh)")
        users = con.execute(
            f"""
            SELECT u.id AS user_id, u.reputation AS reputation
            FROM read_parquet('{users_parquet.as_posix()}') u
            JOIN scope_users s ON u.id = s.id
            """
        ).fetchdf()
    elif users_csv.exists():
        logger.info(f"Sumber reputation: {users_csv} (fallback subset SORD)")
        users = con.execute(
            f"""
            SELECT u.Id AS user_id, u.Reputation AS reputation
            FROM read_csv_auto('{users_csv.as_posix()}', ignore_errors=true) u
            JOIN scope_users s ON u.Id = s.id
            """
        ).fetchdf()
    else:
        logger.warning(f"Tidak ditemukan {users_parquet} maupun {users_csv} -- berhenti.")
        con.unregister("scope_users")
        driver.close()
        return
    con.unregister("scope_users")
    logger.info(f"User dengan reputation ditemukan: {len(users):,} / {len(unique_owner_ids):,} unik dibutuhkan")

    step(logger, 4, "Hitung bobot AUTHOR_TRUST -- normalize(log(reputation)), clip P99")
    users = users.copy()
    users["reputation"] = users["reputation"].fillna(0).clip(lower=0)
    p99 = users["reputation"].quantile(0.99)
    if p99 <= 0:
        p99 = users["reputation"].max() or 1
    users["weight"] = (np.log1p(users["reputation"]) / np.log1p(p99)).clip(0, 1)
    logger.info(f"P99 reputation (basis normalisasi): {p99:,.0f}")

    trust_df = answer_owners.dropna(subset=["owner_user_id"]).merge(
        users, left_on="owner_user_id", right_on="user_id", how="inner"
    )[["answer_id", "user_id", "weight"]]
    logger.info(f"Edge AUTHOR_TRUST siap ditulis: {len(trust_df):,}")

    if args.dry_run:
        logger.info("--dry-run aktif: TIDAK menulis ke Neo4j. Selesai.")
        driver.close()
        return

    try:
        step(logger, 5, "Tulis User node")
        user_df = users.rename(columns={"user_id": "id"})[["id", "reputation"]]
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MERGE (u:User {id: row.id})
            SET u.reputation = row.reputation
            """,
            user_df, args.batch_size, logger, "User",
        )

        step(logger, 6, "Tulis edge AUTHOR_TRUST")
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (a:Answer {id: row.answer_id}), (u:User {id: row.user_id})
            MERGE (a)-[r:AUTHOR_TRUST]->(u)
            SET r.weight = row.weight
            """,
            trust_df, args.batch_size, logger, "AUTHOR_TRUST",
        )

        step(logger, 7, "Verifikasi")
        with driver.session(database=database) as session:
            n_user = session.run("MATCH (u:User) RETURN count(u) AS n").single()["n"]
            n_trust = session.run("MATCH ()-[r:AUTHOR_TRUST]->() RETURN count(r) AS n").single()["n"]
            logger.info(f"  Neo4j User        : {n_user:,}")
            logger.info(f"  Neo4j AUTHOR_TRUST: {n_trust:,}")
    finally:
        driver.close()

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info(f"SELESAI dalam {elapsed:.1f}s")
    logger.info("=" * 78)


if __name__ == "__main__":
    sys.exit(main())