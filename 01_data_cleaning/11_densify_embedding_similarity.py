"""
11_densify_embedding_similarity.py
====================================
Densifikasi Knowledge Graph (Kondisi C) — TAMBAH edge EMBED_SIM
(Question-Question) berdasarkan cosine similarity embedding Title+Body
(all-MiniLM-L6-v2), ke KG yang SUDAH ADA di Neo4j. Sumber densifikasi
KEDUA (lihat juga 10_densify_tag_cooccurrence.py untuk sumber PERTAMA).

KENAPA DUA TAHAP TERPISAH (encode lalu index+edges)
-------------------------------------------------------
Tidak ada embedding per-Question yang tersimpan permanen di KG saat ini --
7_build_knowledge_graph.py HANYA menghitung embedding on-the-fly untuk
pasangan IS_RELATED_TO tipe 'Linked' (tidak disimpan). Untuk 2,68 juta
Question, encoding penuh SentenceTransformer bisa makan 1-3+ jam di RAM
8GB -- kalau proses ini crash di tengah jalan (mati listrik, macOS sleep,
dll.), rebuild dari nol sangat mahal. Karena itu:

    TAHAP A (--stage encode) : encode SEMUA Question -> simpan ke memmap
                                 numpy on-disk + checkpoint id terakhir.
                                 RESUMABLE: jalankan ulang command yang
                                 sama, otomatis lanjut dari checkpoint.
    TAHAP B (--stage edges)  : build FAISS ANN index dari embedding hasil
                                 Tahap A -> cari top-K tetangga per node
                                 (cosine >= threshold) -> tulis ke Neo4j.

    --stage all (default)    : jalankan A lalu B berurutan dalam 1 command
                                 (skip Tahap A otomatis kalau sudah lengkap).

METODOLOGI
----------
1. Tarik semua (Question.id, title, body) dari Neo4j via keyset pagination
   (WHERE id > $last_id ORDER BY id LIMIT batch -- efisien dgn index
   constraint q.id UNIQUE, BUKAN SKIP/LIMIT yang O(skip) di Neo4j).
2. Encode teks = title + " " + body[:1000] (SAMA PERSIS dgn potongan teks
   yang dipakai 7_build_knowledge_graph.py utk edge IS_RELATED_TO 'Linked'
   -- supaya representasi embedding konsisten across the KG).
3. Simpan embedding ke memmap float32 (N x 384) + array id sejajar, plus
   file checkpoint JSON (last_id, next_row) -- resumable.
4. Build FAISS index ANN (IndexIVFFlat, trained on subsample) dari seluruh
   embedding -- exact search (IndexFlatIP) TIDAK feasible untuk 2,68 juta
   x 2,68 juta query (~74 jam estimasi), ANN wajib.
5. Query top-K (default 10) tetangga per node (cosine >= threshold default
   0.7), EXCLUDE self, cap top-K -- EDGE DIRECTED per node sumber (pola
   sama dgn 10_densify_tag_cooccurrence.py, konsisten dgn makna "outgoing"
   di target validasi).
6. Tulis edge EMBED_SIM ke Neo4j (MERGE, idempotent, batched UNWIND).

INSTALL DEPENDENCY
-------------------
    pip install neo4j duckdb pandas numpy sentence-transformers torch faiss-cpu python-dotenv

SETUP .env (root project, NEO4J_* sudah ada di sana)
----------------------------------------------------------------------------
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j

CARA PAKAI
----------
    # Full run (encode + build edges), resumable kalau terhenti
    python 11_densify_embedding_similarity.py

    # Hanya encode dulu (mis. mau jalan semalaman, cek progress besok)
    python 11_densify_embedding_similarity.py --stage encode

    # Encoding sudah selesai, lanjut ke index+edges saja
    python 11_densify_embedding_similarity.py --stage edges

    # Dev/testing cepat (subset kecil, tanpa nulis ke Neo4j)
    python 11_densify_embedding_similarity.py --limit 5000 --dry-run
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
WORKSPACE_DIR = Path("_kg_workspace")
EMBED_DIM = 384  # all-MiniLM-L6-v2
DEFAULT_TOP_K = 10
DEFAULT_COSINE_THRESHOLD = 0.7
DEFAULT_TRUST_WEIGHT = 0.4  # "terendah" -- di bawah TAG_COOCCUR (0.6) dan IS_RELATED_TO (1.0/cosine)
BATCH_SIZE_DEFAULT = 2000
PULL_BATCH_SIZE = 5000
ENCODE_BATCH_SIZE = 64
ENCODE_LOG_CHUNK = 20_000

import logging


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_densify_embedding_similarity.log"

    logger = logging.getLogger("densify_embedding_similarity")
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
    """Identik write_batched() di 7_build_knowledge_graph.py."""
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


def select_device(logger):
    """Coba MPS (Apple Silicon GPU) dulu -- signifikan lebih cepat dari CPU
    untuk encoding 2,68 juta teks di MacBook M2. Fallback ke CPU kalau
    tidak tersedia/error."""
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("[device] MPS (Apple Silicon GPU) tersedia -- dipakai untuk encoding.")
            return "mps"
    except Exception as e:
        logger.info(f"[device] Gagal cek MPS ({e}), fallback ke CPU.")
    logger.info("[device] Pakai CPU untuk encoding.")
    return "cpu"


# ---------------------------------------------------------------------
# TAHAP A - Encode embedding (resumable, keyset pagination dari Neo4j)
# ---------------------------------------------------------------------

def get_question_count(driver, database, limit=None) -> int:
    with driver.session(database=database) as session:
        n = session.run("MATCH (q:Question) RETURN count(q) AS n").single()["n"]
    return min(n, limit) if limit else n


def stage_encode(driver, database, args, logger):
    from sentence_transformers import SentenceTransformer
    import torch

    emb_path = WORKSPACE_DIR / "question_embeddings.f32"
    ids_path = WORKSPACE_DIR / "question_embedding_ids.npy"
    ckpt_path = WORKSPACE_DIR / "embed_checkpoint.json"

    n_total = get_question_count(driver, database, args.limit)
    logger.info(f"Total Question untuk di-encode: {n_total:,}")

    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        last_id = ckpt["last_id"]
        next_row = ckpt["next_row"]
        logger.info(f"[resume] Checkpoint ditemukan: {next_row:,} sudah ter-encode "
                     f"(last_id={last_id}). Melanjutkan...")
        emb_mmap = np.memmap(emb_path, dtype="float32", mode="r+", shape=(n_total, EMBED_DIM))
        ids_arr = np.load(ids_path)
        assert len(ids_arr) == n_total, (
            f"[ERROR] ids array ({len(ids_arr)}) tidak cocok dengan n_total saat ini ({n_total}). "
            f"Kemungkinan jumlah Question di Neo4j berubah sejak checkpoint dibuat -- hapus "
            f"{WORKSPACE_DIR}/question_embeddings.f32 & question_embedding_ids.npy untuk mulai ulang."
        )
    else:
        logger.info("[fresh] Tidak ada checkpoint -- mulai encoding dari awal.")
        last_id = -1
        next_row = 0
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        emb_mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_total, EMBED_DIM))
        ids_arr = np.full(n_total, -1, dtype="int64")

    if next_row >= n_total:
        logger.info("Encoding sudah lengkap (checkpoint menunjukkan semua baris selesai).")
        del emb_mmap
        return

    device = select_device(logger)
    torch.set_num_threads(args.embed_threads)
    model = SentenceTransformer(args.embed_model, device=device)

    limit_clause = f" AND q.id <= {args.limit}" if False else ""  # limit sudah dihandle via n_total scope
    t0 = time.time()
    row = next_row
    with driver.session(database=database) as session:
        while True:
            result = session.run(
                f"""
                MATCH (q:Question)
                WHERE q.id > $last_id
                RETURN q.id AS id, q.title AS title, q.body AS body
                ORDER BY q.id
                LIMIT $batch
                """,
                last_id=last_id, batch=PULL_BATCH_SIZE,
            )
            batch_records = [(r["id"], r["title"] or "", r["body"] or "") for r in result]
            if not batch_records:
                break
            if args.limit and row >= args.limit:
                break

            ids_batch = [r[0] for r in batch_records]
            texts = [f"{r[1]} {r[2][:1000]}" for r in batch_records]

            embeddings = model.encode(
                texts, show_progress_bar=False, batch_size=ENCODE_BATCH_SIZE, convert_to_numpy=True,
            ).astype("float32")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            embeddings = embeddings / norms  # pre-normalize -> cosine == inner product nanti

            n_batch = len(ids_batch)
            emb_mmap[row:row + n_batch] = embeddings
            ids_arr[row:row + n_batch] = ids_batch
            row += n_batch
            last_id = ids_batch[-1]

            if row % ENCODE_LOG_CHUNK < PULL_BATCH_SIZE or row >= n_total:
                emb_mmap.flush()
                np.save(ids_path, ids_arr)
                ckpt_path.write_text(json.dumps({"last_id": int(last_id), "next_row": int(row)}))
                elapsed = time.time() - t0
                rate = (row - next_row) / elapsed if elapsed > 0 else 0
                eta = (n_total - row) / rate if rate > 0 else float("nan")
                logger.info(f"  ... {row:,}/{n_total:,} ter-encode & checkpoint tersimpan "
                            f"({rate:.1f} teks/detik, ETA ~{eta / 60:.1f} menit)")

    emb_mmap.flush()
    np.save(ids_path, ids_arr)
    ckpt_path.write_text(json.dumps({"last_id": int(last_id), "next_row": int(row)}))
    logger.info(f"Encoding selesai: {row:,}/{n_total:,} Question ter-encode.")
    del emb_mmap


# ---------------------------------------------------------------------
# TAHAP B - Build FAISS ANN index + query top-K + tulis edge
# ---------------------------------------------------------------------

def stage_edges(driver, database, args, logger):
    import faiss

    emb_path = WORKSPACE_DIR / "question_embeddings.f32"
    ids_path = WORKSPACE_DIR / "question_embedding_ids.npy"
    ckpt_path = WORKSPACE_DIR / "embed_checkpoint.json"

    if not (emb_path.exists() and ids_path.exists() and ckpt_path.exists()):
        raise SystemExit(
            "[ABORT] Belum ada embedding tersimpan. Jalankan dulu dengan --stage encode "
            "(atau --stage all)."
        )

    ckpt = json.loads(ckpt_path.read_text())
    n_total = ckpt["next_row"]
    ids_arr = np.load(ids_path)[:n_total]
    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(len(ids_arr), EMBED_DIM))[:n_total]
    embeddings = np.ascontiguousarray(embeddings)
    logger.info(f"Embedding dimuat: {n_total:,} vektor (dim={EMBED_DIM})")

    faiss_index_path = WORKSPACE_DIR / "question_embeddings_ivf.faiss"
    faiss_ids_path = WORKSPACE_DIR / "question_embeddings_ivf_ids.npy"

    if faiss_index_path.exists() and not args.rebuild_index:
        logger.info(f"[faiss] Memuat index dari cache -> {faiss_index_path}")
        index = faiss.read_index(str(faiss_index_path))
        index.nprobe = args.nprobe
        logger.info(f"[faiss] Index dimuat: {index.ntotal:,} vektor, nprobe={args.nprobe} "
                     f"(pakai --rebuild-index kalau embedding berubah)")
    else:
        nlist = max(1, min(4096, int(math.sqrt(n_total)) * 2))
        logger.info(f"[faiss] Membangun IndexIVFFlat (nlist={nlist}, metric=inner-product/cosine "
                    f"krn embedding sudah dinormalisasi)...")
        quantizer = faiss.IndexFlatIP(EMBED_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)

        train_size = min(n_total, max(nlist * 40, 100_000))
        train_idx = np.random.default_rng(42).choice(n_total, size=train_size, replace=False)
        t0 = time.time()
        index.train(embeddings[train_idx])
        logger.info(f"[faiss] Index trained pada {train_size:,} sampel dalam {time.time() - t0:.1f}s")

        index.add(embeddings)
        index.nprobe = args.nprobe
        logger.info(f"[faiss] Index terisi {index.ntotal:,} vektor, nprobe={args.nprobe}")

        faiss.write_index(index, str(faiss_index_path))
        np.save(faiss_ids_path, ids_arr)
        logger.info(f"[faiss] Index di-cache -> {faiss_index_path} (dipakai ulang oleh "
                     f"12_validate_densification.py & nanti pipeline retrieval Kondisi C)")

    # top_k+1 karena hasil pertama biasanya diri sendiri (exclude di post-processing)
    step_local = "[search]"
    logger.info(f"{step_local} Mencari top-{args.top_k} tetangga per node (cosine >= {args.cosine_threshold})...")
    t0 = time.time()
    search_batch = 10_000
    edge_rows = []
    for i in range(0, n_total, search_batch):
        q_emb = embeddings[i:i + search_batch]
        scores, indices = index.search(q_emb, args.top_k + 1)
        for local_row, (score_row, idx_row) in enumerate(zip(scores, indices)):
            src_id = ids_arr[i + local_row]
            kept = 0
            for score, idx in zip(score_row, idx_row):
                if idx == -1 or idx == i + local_row:
                    continue  # exclude self / hasil kosong
                if score < args.cosine_threshold:
                    continue
                edge_rows.append((int(src_id), int(ids_arr[idx]), float(score)))
                kept += 1
                if kept >= args.top_k:
                    break
        if (i // search_batch) % 20 == 0:
            elapsed = time.time() - t0
            done = min(i + search_batch, n_total)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else float("nan")
            logger.info(f"  ... {done:,}/{n_total:,} query selesai ({rate:,.0f}/detik, "
                        f"ETA ~{eta / 60:.1f} menit)")

    edges = pd.DataFrame(edge_rows, columns=["question_id", "related_question_id", "cosine_sim"])
    n_src_with_edge = edges["question_id"].nunique()
    logger.info(f"{step_local} Edge EMBED_SIM ditemukan: {len(edges):,} dari "
                f"{n_src_with_edge:,} Question sumber unik")

    if args.dry_run:
        logger.info("--dry-run aktif: TIDAK menulis ke Neo4j.")
        if len(edges):
            logger.info(f"Contoh 10 edge teratas by cosine_sim:\n"
                        f"{edges.sort_values('cosine_sim', ascending=False).head(10).to_string(index=False)}")
        return

    step(None, 0, "") if False else None
    logger.info("Menulis edge EMBED_SIM ke Neo4j...")
    edges = edges.assign(trust_weight=args.trust_weight)
    write_batched(
        driver, database,
        """
        UNWIND $rows AS row
        MATCH (q1:Question {id: row.question_id}), (q2:Question {id: row.related_question_id})
        MERGE (q1)-[r:EMBED_SIM]->(q2)
        SET r.weight = row.trust_weight, r.cosine_sim = row.cosine_sim
        """,
        edges, args.batch_size, logger, "EMBED_SIM",
    )

    with driver.session(database=database) as session:
        n = session.run("MATCH ()-[r:EMBED_SIM]->() RETURN count(r) AS n").single()["n"]
        logger.info(f"  Neo4j EMBED_SIM: {n:,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "encode", "edges"], default="all")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--cosine-threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    parser.add_argument("--trust-weight", type=float, default=DEFAULT_TRUST_WEIGHT,
                         help=f"Bobot trust kategorikal utk edge EMBED_SIM (default {DEFAULT_TRUST_WEIGHT}, "
                              f"'terendah' di antara 3 sumber trust)")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embed-threads", type=int, default=4)
    parser.add_argument("--nprobe", type=int, default=32,
                         help="Jumlah cluster IVF yang dicek saat search (recall vs speed trade-off)")
    parser.add_argument("--rebuild-index", action="store_true",
                         help="Paksa rebuild FAISS index meski cache sudah ada")
    parser.add_argument("--limit", type=int, default=None, help="Dev/testing: batasi jumlah Question")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger, log_path = setup_logging()
    logger.info("DENSIFIKASI KG -- EMBED_SIM (cosine similarity embedding Title+Body)")
    logger.info(f"Log file: {log_path}")
    logger.info(f"[config] stage={args.stage} top_k={args.top_k} cosine_threshold={args.cosine_threshold} "
                f"trust_weight={args.trust_weight}")
    t_start = time.time()

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    driver, database = connect_neo4j(logger)
    try:
        if args.stage in ("all", "encode"):
            step(logger, 1, "Encode embedding Title+Body (resumable)")
            stage_encode(driver, database, args, logger)

        if args.stage in ("all", "edges"):
            step(logger, 2, "Build FAISS ANN index + cari top-K tetangga + tulis edge EMBED_SIM")
            stage_edges(driver, database, args, logger)
    finally:
        driver.close()

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info(f"SELESAI dalam {elapsed / 60:.1f} menit")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()