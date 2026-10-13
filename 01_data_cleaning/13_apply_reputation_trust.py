"""
13_apply_reputation_trust.py
==============================
Menerapkan keputusan Amendment 5 (docs/DECISION_C_SCORING.md): memasukkan
reputasi kontributor ke trustScore simpul Answer di Neo4j.

    trustScore_baru = clip(trustScore_lama - 0.2*0.5 + 0.2*author_weight, 0, 1)

author_weight = bobot edge AUTHOR_TRUST (log1p(rep)/log1p(rep_P99), dipotong
[0,1]); 0 bila jawaban tidak punya edge AUTHOR_TRUST. Rumus ini IDENTIK dengan
varian `repfix` yang dievaluasi oleh llm/c_graphrag/reputation_stage1_recheck.py
(diuji di tests/test_apply_reputation_trust.py). Sekaligus mengisi
authorReputation (sebelumnya 0 untuk semua jawaban) dan menandai setiap
Answer dengan trustScoreVersion = "v2-reputation".

KEAMANAN
--------
- Default = --dry-run: hanya membaca dan menampilkan ringkasan, TIDAK menulis.
- --apply: sebelum menulis, nilai lama (id, trustScore, authorReputation)
  disimpan ke _kg_workspace/answer_trust_backup_pre_amendment5.parquet.
  Backup TIDAK pernah ditimpa. Nilai baru SELALU dihitung dari backup,
  sehingga menjalankan --apply berulang kali (mis. setelah terputus) aman
  dan menghasilkan nilai yang sama.
- --restore: mengembalikan trustScore dan authorReputation dari backup dan
  menghapus trustScoreVersion.
- Pemeriksaan asumsi sebelum backup pertama: tidak ada Answer dengan
  authorReputation != 0 dan trustScore minimum = 0.100 (kondisi graf yang
  dievaluasi pada Amendment 5). Kalau tidak terpenuhi, skrip berhenti.

Disarankan: buat dump database dulu di Neo4j Desktop (... > Dump) sebagai
cadangan kedua, dan tutup Finder pada folder logs/ selama proses.

CARA PAKAI (dari folder 01_data_cleaning/)
------------------------------------------
    python 13_apply_reputation_trust.py              # dry-run
    python 13_apply_reputation_trust.py --apply      # tulis ke Neo4j + verifikasi
    python 13_apply_reputation_trust.py --restore    # batalkan
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
REPUTATION_COEF = 0.2
STORED_REPUTATION_NORM = 0.5
VERSION = "v2-reputation"
BACKUP_NAME = "answer_trust_backup_pre_amendment5.parquet"


def compute_new_trust(old_trust: np.ndarray, author_weight: np.ndarray) -> np.ndarray:
    """trust' = clip(trust - 0.2*0.5 + 0.2*w, 0, 1); w kosong -> 0."""
    w = np.nan_to_num(np.asarray(author_weight, dtype=float), nan=0.0)
    old = np.asarray(old_trust, dtype=float)
    return np.clip(old - REPUTATION_COEF * STORED_REPUTATION_NORM + REPUTATION_COEF * w, 0.0, 1.0)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_apply_reputation_trust.log"
    logger = logging.getLogger("apply_reputation_trust")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, log_path


def connect_neo4j(logger):
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip().strip('"').strip("'")
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()
    if not password:
        raise ValueError("NEO4J_PASSWORD tidak ditemukan di .env (root project)")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    logger.info(f"Terhubung ke Neo4j: {uri} | database: {database}")
    return driver, database


def read_answers(driver, database) -> pd.DataFrame:
    with driver.session(database=database) as s:
        rows = s.run(
            """
            MATCH (a:Answer)
            OPTIONAL MATCH (a)-[r:AUTHOR_TRUST]->(u:User)
            WITH a, max(r.weight) AS w, max(u.reputation) AS user_rep
            RETURN a.id AS answer_id, a.trustScore AS trust, a.authorReputation AS rep_stored,
                   a.trustScoreVersion AS version, w AS author_weight, user_rep
            """
        ).data()
    return pd.DataFrame(rows)


def write_batched(driver, database, cypher: str, df: pd.DataFrame, logger, label: str,
                  batch_size: int = 10_000, **params):
    total, written, t0 = len(df), 0, time.time()
    with driver.session(database=database) as s:
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i + batch_size].to_dict("records")
            s.execute_write(lambda tx, b=batch: tx.run(cypher, rows=b, **params).consume())
            written += len(batch)
            if (i // batch_size) % 10 == 0 or written == total:
                rate = written / max(time.time() - t0, 1e-9)
                logger.info(f"  [{label}] {written:,}/{total:,} ({rate:,.0f}/detik)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--kg-workspace-dir", default="_kg_workspace")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--restore", action="store_true")
    args = p.parse_args()

    logger, log_path = setup_logging()
    mode = "apply" if args.apply else "restore" if args.restore else "dry-run"
    logger.info(f"Mode: {mode} | log: {log_path}")
    backup_path = Path(args.kg_workspace_dir) / BACKUP_NAME
    driver, database = connect_neo4j(logger)

    try:
        df = read_answers(driver, database)
        logger.info(f"Answer dibaca: {len(df):,} | punya AUTHOR_TRUST: {df['author_weight'].notna().sum():,} "
                    f"| sudah {VERSION}: {(df['version'] == VERSION).sum():,}")

        if args.restore:
            if not backup_path.exists():
                logger.info(f"[STOP] Backup tidak ditemukan: {backup_path}")
                return 1
            bk = pd.read_parquet(backup_path)
            write_batched(driver, database,
                          """UNWIND $rows AS row MATCH (a:Answer {id: row.answer_id})
                             SET a.trustScore = row.trust, a.authorReputation = row.rep_stored
                             REMOVE a.trustScoreVersion""",
                          bk[["answer_id", "trust", "rep_stored"]], logger, "restore")
            logger.info("Restore selesai.")
            return 0

        if backup_path.exists():
            bk = pd.read_parquet(backup_path)
            logger.info(f"Memakai backup yang sudah ada sebagai nilai lama: {backup_path} ({len(bk):,} baris)")
        else:
            n_rep = int((df["rep_stored"].fillna(0) != 0).sum())
            tmin = float(df["trust"].min())
            ok = n_rep == 0 and abs(tmin - REPUTATION_COEF * STORED_REPUTATION_NORM) < 1e-3
            logger.info(f"Cek asumsi: authorReputation != 0 = {n_rep:,}; trustScore min = {tmin:.4f} -> "
                        f"{'OK' if ok else 'GAGAL'}")
            if not ok:
                logger.info("[STOP] Graf tidak dalam kondisi yang dievaluasi pada Amendment 5.")
                return 1
            bk = df[["answer_id", "trust", "rep_stored"]].copy()

        merged = bk.merge(df[["answer_id", "author_weight", "user_rep"]], on="answer_id", how="left")
        if len(merged) != len(df):
            logger.info(f"[STOP] Jumlah Answer di graf ({len(df):,}) != backup ({len(merged):,}).")
            return 1
        merged["trust_new"] = compute_new_trust(merged["trust"].values, merged["author_weight"].values)
        merged["user_rep"] = merged["user_rep"].fillna(0)
        delta = merged["trust_new"] - merged["trust"]
        logger.info(f"trustScore lama : min {merged['trust'].min():.4f} max {merged['trust'].max():.4f} "
                    f"mean {merged['trust'].mean():.4f}")
        logger.info(f"trustScore baru : min {merged['trust_new'].min():.4f} max {merged['trust_new'].max():.4f} "
                    f"mean {merged['trust_new'].mean():.4f}")
        logger.info(f"Perubahan       : min {delta.min():+.4f} max {delta.max():+.4f} mean {delta.mean():+.4f}; "
                    f"tanpa AUTHOR_TRUST (w=0, turun 0.1): {merged['author_weight'].isna().sum():,}")

        if not args.apply:
            logger.info("Dry-run: tidak ada yang ditulis. Jalankan dengan --apply untuk menerapkan.")
            return 0

        if not backup_path.exists():
            bk.to_parquet(backup_path, index=False)
            logger.info(f"Backup ditulis: {backup_path}")

        write_batched(driver, database,
                      """UNWIND $rows AS row MATCH (a:Answer {id: row.answer_id})
                         SET a.trustScore = row.trust_new, a.authorReputation = row.user_rep,
                             a.trustScoreVersion = $version""",
                      merged[["answer_id", "trust_new", "user_rep"]], logger, "apply", version=VERSION)

        after = read_answers(driver, database).merge(merged[["answer_id", "trust_new"]], on="answer_id")
        max_diff = float((after["trust"] - after["trust_new"]).abs().max())
        n_ver = int((after["version"] == VERSION).sum())
        ok = math.isclose(max_diff, 0.0, abs_tol=1e-9) and n_ver == len(merged)
        logger.info(f"Verifikasi: selisih maks {max_diff:.2e}; {n_ver:,}/{len(merged):,} bertanda {VERSION} -> "
                    f"{'OK' if ok else 'GAGAL'}")
        return 0 if ok else 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
