"""
build_kg_answer_index.py
=====================================
Offline build of Condition B's retrieval index over EVERY Answer node of
the knowledge graph Condition C retrieves from (reports/b_corpus_parity.md).
This is the ONLY B code that opens a Neo4j connection, and only in
--stage export, read-only. Condition B at runtime reads the files written
here and never touches Neo4j.

    Neo4j --export--> kg_answers.parquet --encode--> embeddings.f32 --index--> IndexFlatIP + meta

STAGES
------
  export : keyset-paginate (WHERE a.id > $last_id ORDER BY a.id LIMIT $batch)
           over every Answer with its parent Question (HAS_ANSWER |
           HAS_ACCEPTED_ANSWER), write answer_id, question_id, q.title,
           a.body, a.isAccepted + doc_text to kg_answers.parquet in row
           groups (never the whole corpus in RAM). doc_text =
           truncate_doc_text(answer_doc_text(title, body), TOKEN_CHUNK_LIMIT),
           the SAME functions C applies to each candidate (llm/common/text.py)
           -- one vector per answer, no multi-chunking. An Answer without a
           parent Question (C can never reach it) or with two parents
           (ambiguous text) aborts the export with a count; the 2026-10-17
           KG has 0 of either.
  encode : all-MiniLM-L6-v2 over doc_text, L2-normalized, into a float32
           memmap + int64 id memmap. RESUMABLE: work proceeds in fixed units
           of --unit-rows rows (keyset over answer_id); after each unit the
           memmaps are flushed, then encode_checkpoint.json is replaced
           atomically. Re-running the same command continues from the
           checkpoint. Batch size / device / unit size are recorded in the
           checkpoint and must match on resume (see DETERMINISM).
  index  : faiss.IndexFlatIP (exact search) over the memmap, plus a meta
           parquet of answer_id, question_id, is_accepted (row i = vector
           i). No text is pickled with the index -- text stays in
           kg_answers.parquet.
  all    : export (skipped if kg_answers.parquet + manifest exist) -> encode -> index.

DETERMINISM (encode)
--------------------
C encodes candidates one text per call (batch_size=1, CPU) because
all-MiniLM-L6-v2's output for a text moves by ~1e-7 with the other texts in
its batch. B's document vectors are never compared with C's, so B encodes in
batches (default --batch-size 64, CPU) for speed; what makes the result
reproducible is that every encode() call sees exactly the same texts:
units are fixed (--unit-rows consecutive answer_ids), each unit is one
encode() call, and a resume starts on a unit boundary. A resumed run is
therefore bit-identical to an uninterrupted run with the same
--unit-rows/--batch-size/--device (tested in
llm/b_rag/tests/test_kg_corpus_build.py). Changing any of them would change
vectors by ~1e-7 (never the ranking in practice), so the checkpoint refuses
a resume with different values.

USAGE
-----
    python llm/b_rag/build_kg_answer_index.py --stage all
    python llm/b_rag/build_kg_answer_index.py --stage encode     # resumes if interrupted

Reads NEO4J_* from the repo-root .env (export stage only).
"""

import argparse
import json
import logging
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from llm.b_rag.kg_corpus import (CORPUS_VERSION, EMBED_DIM, IdsDigest, corpus_paths, ids_sha256,  # noqa: E402
                                 read_json, write_json_atomic)
from llm.common.text import TOKENIZER_NAME, answer_doc_text, get_tokenizer, truncate_doc_text  # noqa: E402

PULL_BATCH_SIZE = 5000
DEFAULT_UNIT_ROWS = 4096
DEFAULT_BATCH_SIZE = 64
DEFAULT_INDEX_CACHE_DIR = Path(__file__).resolve().parent / "index_cache"

log = print


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def peak_rss_mb() -> float:
    # ru_maxrss: bytes on macOS, KiB on Linux
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1024 / 1024 if sys.platform == "darwin" else r / 1024, 1)


# ---------------------------------------------------------------------
# export (Neo4j, read-only)
# ---------------------------------------------------------------------

EXPORT_CYPHER = """
MATCH (a:Answer) WHERE a.id > $last_id
WITH a ORDER BY a.id ASC LIMIT $batch
OPTIONAL MATCH (q:Question)-[r:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a)
RETURN a.id AS answer_id, q.id AS question_id, q.title AS question_title,
       a.body AS answer_body, a.isAccepted AS is_accepted, type(r) AS rel_type
ORDER BY answer_id ASC
"""

PARQUET_SCHEMA_FIELDS = [
    ("answer_id", "int64"), ("question_id", "int64"), ("question_title", "string"), ("answer_body", "string"),
    ("is_accepted", "bool_"), ("rel_type", "string"), ("doc_text", "string"), ("truncated", "bool_"),
]


def _parquet_schema():
    import pyarrow as pa

    return pa.schema([(name, getattr(pa, typ)()) for name, typ in PARQUET_SCHEMA_FIELDS])


def _connect_neo4j():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(REPO_ROOT / ".env")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    password = (os.getenv("NEO4J_PASSWORD") or "").strip().strip('"').strip("'")
    if not password:
        raise SystemExit("NEO4J_PASSWORD tidak ditemukan di .env (root project)")
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j").strip(), password))
    driver.verify_connectivity()
    return driver, os.getenv("NEO4J_DATABASE", "neo4j").strip()


def make_doc_rows(records: list[dict], enc, token_chunk_limit: int, pool=None) -> list[dict]:
    """Add doc_text/truncated to exported records using the shared C
    functions. `pool` (ThreadPoolExecutor) only parallelizes the calls --
    tiktoken releases the GIL; the function applied per text is unchanged."""
    full = [answer_doc_text(r["question_title"], r["answer_body"]) for r in records]
    fn = lambda t: truncate_doc_text(enc, t, token_chunk_limit)  # noqa: E731
    cut = list(pool.map(fn, full)) if pool is not None else [fn(t) for t in full]
    return [{**r, "doc_text": c, "truncated": c != f} for r, f, c in zip(records, full, cut)]


def stage_export(paths, token_chunk_limit: int, limit: int | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    paths.root.mkdir(parents=True, exist_ok=True)
    driver, database = _connect_neo4j()
    enc = get_tokenizer()
    schema = _parquet_schema()
    tmp = paths.parquet.with_name(paths.parquet.name + ".partial")
    digest = IdsDigest()
    t0 = time.time()
    n_rows = n_orphan = n_truncated = 0
    by_rel = {}
    orphan_ids, multi_parent_ids = [], []
    try:
        with driver.session(database=database, default_access_mode="READ") as session:
            n_answer_nodes = session.run("MATCH (a:Answer) RETURN count(a) AS n").single()["n"]
            log(f"[export] {n_answer_nodes:,} Answer node di database '{database}'")
            writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            last_id = -1
            with ThreadPoolExecutor(max_workers=4) as pool:
                while True:
                    batch = [dict(r) for r in session.run(EXPORT_CYPHER, last_id=last_id, batch=PULL_BATCH_SIZE)]
                    if not batch:
                        break
                    last_id = batch[-1]["answer_id"]
                    seen, rows = set(), []
                    for r in batch:
                        aid = r["answer_id"]
                        if aid in seen:
                            multi_parent_ids.append(aid)
                            continue
                        seen.add(aid)
                        if r["question_id"] is None:
                            n_orphan += 1
                            orphan_ids.append(aid)
                            continue
                        rows.append(r)
                    if multi_parent_ids:
                        raise SystemExit(f"[ABORT] Answer(s) with more than one parent Question: {multi_parent_ids[:10]}")
                    if limit is not None:
                        rows = rows[: max(0, limit - n_rows)]
                    rows = make_doc_rows(rows, enc, token_chunk_limit, pool)
                    if rows:
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                        digest.update([r["answer_id"] for r in rows])
                        n_rows += len(rows)
                        n_truncated += sum(r["truncated"] for r in rows)
                        for r in rows:
                            by_rel[r["rel_type"]] = by_rel.get(r["rel_type"], 0) + 1
                    el = time.time() - t0
                    log(f"[export] {n_rows:,}/{n_answer_nodes:,} ({n_rows / el:.0f} baris/detik, "
                        f"peak RSS {peak_rss_mb()} MB)")
                    if limit is not None and n_rows >= limit:
                        break
            writer.close()
    finally:
        driver.close()
    if orphan_ids:
        raise SystemExit(f"[ABORT] {n_orphan} Answer node(s) without a parent Question (C cannot reach "
                         f"them; decide their treatment before exporting): {orphan_ids[:10]}")
    tmp.replace(paths.parquet)
    manifest = {
        "corpus_version": CORPUS_VERSION, "token_chunk_limit": token_chunk_limit, "tokenizer": TOKENIZER_NAME,
        "doc_text_format": "truncate_doc_text(answer_doc_text(q.title, a.body)) -- llm/common/text.py",
        "neo4j_database": database, "n_answer_nodes": n_answer_nodes, "n_rows": n_rows,
        "n_orphan_answers": n_orphan, "n_multi_parent_answers": len(multi_parent_ids),
        "n_by_rel_type": by_rel, "n_truncated": n_truncated, "limit": limit,
        "ids_sha256": digest.hexdigest(), "exported_at_utc": _now(),
        "duration_sec": round(time.time() - t0, 1), "peak_rss_mb": peak_rss_mb(),
        "parquet_bytes": paths.parquet.stat().st_size,
    }
    write_json_atomic(paths.export_manifest, manifest)
    log(f"[export] selesai: {n_rows:,} baris -> {paths.parquet} ({manifest['parquet_bytes'] / 1e6:.0f} MB, "
        f"{manifest['duration_sec']}s)")
    return manifest


# ---------------------------------------------------------------------
# encode (resumable)
# ---------------------------------------------------------------------

def _load_model(embed_model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embed_model_name, device=device)


def encode_texts(model, texts: list[str], batch_size: int, device: str) -> np.ndarray:
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False,
                       device=device).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return emb / norms


def stage_encode(paths, embed_model_name: str, batch_size: int = DEFAULT_BATCH_SIZE, device: str = "cpu",
                 unit_rows: int = DEFAULT_UNIT_ROWS, threads: int | None = None, stop_after_units: int | None = None,
                 model=None) -> dict:
    """Encode kg_answers.parquet's doc_text. Returns the checkpoint dict.
    `stop_after_units` simulates an interruption (tests / manual check)."""
    import duckdb

    export = read_json(paths.export_manifest)
    n_total, sha = export["n_rows"], export["ids_sha256"]
    paths.embed_dir.mkdir(parents=True, exist_ok=True)
    params = {"embed_model": embed_model_name, "batch_size": batch_size, "device": device, "unit_rows": unit_rows,
              "n_total": n_total, "ids_sha256": sha}

    if paths.encode_checkpoint.exists():
        ckpt = read_json(paths.encode_checkpoint)
        mismatch = {k: (ckpt.get(k), v) for k, v in params.items() if ckpt.get(k) != v}
        if mismatch:
            raise SystemExit(f"[ABORT] encode checkpoint was made with different parameters {mismatch} -- "
                             f"delete {paths.embed_dir} to start over, or rerun with the original values.")
        mode = "r+"
        log(f"[encode] resume dari checkpoint: {ckpt['next_row']:,}/{n_total:,} (last_id={ckpt['last_id']})")
    else:
        ckpt = {**params, "next_row": 0, "last_id": -1, "started_at_utc": _now(), "encode_sec": 0.0}
        mode = "w+"
    if ckpt["next_row"] >= n_total:
        log("[encode] sudah lengkap.")
        return ckpt

    emb_mm = np.memmap(paths.embeddings, dtype="float32", mode=mode, shape=(n_total, EMBED_DIM))
    ids_mm = np.memmap(paths.embed_ids, dtype="int64", mode=mode, shape=(n_total,))
    if model is None:
        if threads:
            import torch

            torch.set_num_threads(threads)
        model = _load_model(embed_model_name, device)
    con = duckdb.connect()
    parquet_sql = str(paths.parquet).replace("'", "''")
    row, last_id, units_done = ckpt["next_row"], ckpt["last_id"], 0
    t0 = time.time()
    while row < n_total:
        df = con.execute(
            f"SELECT answer_id, doc_text FROM read_parquet('{parquet_sql}') "
            f"WHERE answer_id > ? ORDER BY answer_id LIMIT ?", [last_id, unit_rows]
        ).df()
        if df.empty:
            break
        n = len(df)
        emb_mm[row:row + n] = encode_texts(model, df["doc_text"].tolist(), batch_size, device)
        ids_mm[row:row + n] = df["answer_id"].to_numpy(dtype="int64")
        emb_mm.flush()
        ids_mm.flush()
        row += n
        last_id = int(df["answer_id"].iloc[-1])
        ckpt.update(next_row=row, last_id=last_id, encode_sec=round(ckpt["encode_sec"] + time.time() - t0, 1),
                    updated_at_utc=_now(), peak_rss_mb=max(ckpt.get("peak_rss_mb", 0), peak_rss_mb()))
        t0 = time.time()
        write_json_atomic(paths.encode_checkpoint, ckpt)
        units_done += 1
        if units_done % 10 == 0 or row >= n_total:
            rate = row / ckpt["encode_sec"] if ckpt["encode_sec"] else 0
            log(f"[encode] {row:,}/{n_total:,} ({rate:.0f} teks/detik kumulatif, "
                f"ETA ~{(n_total - row) / rate / 60 if rate else float('nan'):.1f} menit, peak RSS {peak_rss_mb()} MB)")
        if stop_after_units is not None and units_done >= stop_after_units:
            log(f"[encode] berhenti setelah {units_done} unit (--stop-after-units).")
            break
    del emb_mm, ids_mm
    if row >= n_total:
        ckpt["finished_at_utc"] = _now()
        write_json_atomic(paths.encode_checkpoint, ckpt)
    return ckpt


# ---------------------------------------------------------------------
# index
# ---------------------------------------------------------------------

def stage_index(paths, add_chunk: int = 65536) -> dict:
    import duckdb
    import faiss
    import pyarrow.parquet as pq

    export = read_json(paths.export_manifest)
    ckpt = read_json(paths.encode_checkpoint)
    n_total, sha = export["n_rows"], export["ids_sha256"]
    if ckpt["next_row"] < n_total or ckpt.get("ids_sha256") != sha:
        raise SystemExit(f"[ABORT] encoding incomplete ({ckpt['next_row']:,}/{n_total:,}) or for another export.")
    t0 = time.time()
    ids_mm = np.memmap(paths.embed_ids, dtype="int64", mode="r", shape=(n_total,))
    if ids_sha256(ids_mm) != sha:
        raise SystemExit("[ABORT] embedding id order != kg_answers.parquet id order.")
    emb_mm = np.memmap(paths.embeddings, dtype="float32", mode="r", shape=(n_total, EMBED_DIM))
    index = faiss.IndexFlatIP(EMBED_DIM)
    for i in range(0, n_total, add_chunk):
        index.add(np.ascontiguousarray(emb_mm[i:i + add_chunk]))
    faiss.write_index(index, str(paths.index(sha)))

    con = duckdb.connect()
    meta = con.execute(
        f"SELECT answer_id, question_id, is_accepted FROM read_parquet('{str(paths.parquet).replace(chr(39), chr(39) * 2)}') "
        f"ORDER BY answer_id"
    ).fetch_arrow_table()
    if not np.array_equal(meta.column("answer_id").to_numpy(), np.asarray(ids_mm)):
        raise SystemExit("[ABORT] meta answer_id order != embedding order.")
    pq.write_table(meta, paths.index_meta(sha), compression="zstd")
    manifest = {
        "cache_key": paths.key(sha), "corpus_version": CORPUS_VERSION, "index_type": "IndexFlatIP",
        "metric": "inner product over L2-normalized vectors (= cosine)", "n_vectors": int(index.ntotal),
        "dim": EMBED_DIM, "ids_sha256": sha, "embed_model": ckpt["embed_model"],
        "encode": {k: ckpt[k] for k in ("batch_size", "device", "unit_rows", "encode_sec")},
        "token_chunk_limit": export["token_chunk_limit"], "built_at_utc": _now(),
        "index_sec": round(time.time() - t0, 1), "index_bytes": paths.index(sha).stat().st_size,
        "peak_rss_mb": peak_rss_mb(),
    }
    write_json_atomic(paths.index_manifest(sha), manifest)
    log(f"[index] {index.ntotal:,} vektor -> {paths.index(sha)} ({manifest['index_bytes'] / 1e6:.0f} MB)")
    return manifest


def main():
    global log
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["export", "encode", "index", "all"], default="all")
    parser.add_argument("--index-cache-dir", default=str(DEFAULT_INDEX_CACHE_DIR))
    parser.add_argument("--token-chunk-limit", type=int, default=int(os.getenv("TOKEN_CHUNK_LIMIT", 400)))
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--unit-rows", type=int, default=DEFAULT_UNIT_ROWS)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's own)")
    parser.add_argument("--stop-after-units", type=int, default=None, help="simulate an interruption")
    parser.add_argument("--limit", type=int, default=None, help="export only the first N answers (testing)")
    parser.add_argument("--force-export", action="store_true", help="re-export even if kg_answers.parquet exists")
    args = parser.parse_args()

    paths = corpus_paths(args.index_cache_dir, args.token_chunk_limit, args.embed_model)
    paths.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(paths.root / "build.log", encoding="utf-8")])
    log = logging.getLogger("build_kg_answer_index").info
    log(f"[config] stage={args.stage} root={paths.root} tok={args.token_chunk_limit} model={args.embed_model} "
        f"batch={args.batch_size} device={args.device} unit_rows={args.unit_rows}")

    if args.stage in ("export", "all"):
        if args.stage == "all" and paths.parquet.exists() and paths.export_manifest.exists() and not args.force_export:
            log(f"[export] dilewati -- {paths.parquet} sudah ada (pakai --force-export / --stage export untuk ulang)")
        else:
            if paths.embed_dir.exists() and paths.encode_checkpoint.exists():
                log("[export] PERINGATAN: checkpoint encode lama akan ditolak bila ids_sha256 berubah.")
            stage_export(paths, args.token_chunk_limit, args.limit)
    if args.stage in ("encode", "all"):
        ckpt = stage_encode(paths, args.embed_model, args.batch_size, args.device, args.unit_rows, args.threads,
                            args.stop_after_units)
        if ckpt["next_row"] < ckpt["n_total"]:
            log("[encode] belum lengkap -- jalankan ulang perintah yang sama untuk melanjutkan.")
            return
    if args.stage in ("index", "all"):
        stage_index(paths)


if __name__ == "__main__":
    main()
