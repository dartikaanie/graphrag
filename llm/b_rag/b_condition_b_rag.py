"""
b_condition_b_rag.py
=====================================
Fase 2 — Kondisi B (Conventional RAG) pada protokol evaluasi tiga-kondisi
penelitian: dense retrieval flat-chunk atas korpus komunitas Stack Overflow
(SORD) sebagai pembanding bagi Kondisi A (LLM murni) dan Kondisi C
(GraphRAG Framework berbobot).

METODOLOGI
------------------------------------------------------------
Sama seperti Kondisi A untuk bagian yang overlap (supaya perbandingan adil):
  - Kriteria inklusi/eksklusi pertanyaan evaluasi: identik Kondisi A
    (hanya pertanyaan dg accepted answer, <= TOKEN_LIMIT token).
  - Ukuran sampel evaluasi, seed, provider LLM: identik pola Kondisi A.
  - Model embedding utk similarity scoring akhir: SAMA (all-MiniLM-L6-v2)
    supaya cosine similarity Kondisi A vs B vs C bisa dibandingkan apple-
    to-apple.

Yang BARU dibanding Kondisi A (retrieval-augmented generation):
  1. Korpus retrieval: pasangan (Title pertanyaan + Body jawaban) dari
     Answers x Questions parquet (JOIN via ParentId), di-flat-chunk (tanpa
     struktur graph -- itu yang membedakan dari Kondisi C), di-index pakai
     FAISS (IndexFlatIP atas embedding ternormalisasi = cosine similarity).
  2. Retrieval top-k per pertanyaan evaluasi sebelum prompting.
  3. Prompt memakai struktur 4-turn chat dialog yang SAMA PERSIS dengan
     Kondisi A (system persona umum -> user persona dari tags -> assistant
     priming -> user pertanyaan+deskripsi), DITAMBAH konteks hasil
     retrieval disisipkan di turn user terakhir sebelum pertanyaan. LLM
     dipanggil via factory provider yang SAMA PERSIS dengan Kondisi A
     (get_llm_client).
  4. Index di-cache ke disk (INDEX_CACHE_DIR) supaya tidak perlu di-build
     ulang setiap run -- pakai --rebuild-index untuk paksa rebuild.

KEPUTUSAN METODOLOGIS EKSPLISIT (didokumentasikan supaya bisa dipertanggung-
jawabkan di sidang / bab metodologi)
------------------------------------------------------------
  - Korpus retrieval = Title pertanyaan asal + Body jawaban (BUKAN body
    jawaban saja) -- jawaban SO seringkali ambigu tanpa konteks
    pertanyaannya, jadi title diikutsertakan ke setiap chunk supaya
    retrieval secara semantik lebih bermakna.
  - PENCEGAHAN DATA LEAKAGE (ketat, by design): korpus index yang dibangun
    di Step 5 secara eksplisit MENGECUALIKAN (a) accepted answer dari
    seluruh pertanyaan evaluasi (target yang sedang diprediksi), DAN
    (b) semua jawaban lain (non-accepted) yang ParentId-nya sama dengan
    pertanyaan evaluasi manapun. Artinya retrieval TIDAK PERNAH bisa
    menemukan kembali jawaban dari pertanyaan yang sedang dievaluasi,
    baik langsung maupun tidak langsung -- supaya cosine similarity yang
    diukur benar-benar mencerminkan kemampuan generalisasi RAG, bukan
    kebetulan menemukan jawabannya sendiri di index.
  - Chunking: setiap dokumen (title + body) dipotong per TOKEN_CHUNK_LIMIT
    token (tiktoken cl100k_base, konsisten dgn cara hitung token Kondisi A)
    -- dokumen pendek jadi 1 chunk, dokumen panjang jadi beberapa chunk
    berurutan tanpa overlap (desain sederhana utk Kondisi B; strategi
    chunking lebih canggih/overlap bisa jadi perbaikan Kondisi C kalau perlu).
  - retrieved_context disimpan lengkap di output (bukan cuma teks final
    LLM) supaya nanti bisa dipakai utk metrik RAGAS (context precision/
    recall) di Bab VI tanpa perlu re-run retrieval.

INSTALL DEPENDENCY (tambahan dari Kondisi A)
-------------------
    pip install faiss-cpu
    # dependency lain (duckdb, tiktoken, pandas, python-dotenv,
    # sentence-transformers, + provider LLM) sama seperti Kondisi A.

SETUP .env (WAJIB)
----------------------------------------------------------------------------
    # Path data (relatif terhadap folder b_rag/, sejajar dg 00_datasource/)
    QUESTIONS_PARQUET=../00_datasource/merged/questions_raw_union.parquet
    ANSWERS_PARQUET=../00_datasource/merged/answers_raw_union.parquet
    OUTPUT_DIR=results
    # ^ Nama file JSONL otomatis: condition_b_{provider}_{safe_model}.jsonl
    #   (pola identik Kondisi A). Pakai --output untuk override manual.

    # Parameter sampel evaluasi -- SAMAKAN dengan Kondisi A supaya sample
    # pertanyaan yang dievaluasi identik (fair comparison). Kalau SEED sama
    # dan OVERSAMPLE_POOL/N_SAMPLE sama, sample_questions() akan reproduce
    # sample pertanyaan yang persis sama dengan Kondisi A.
    N_SAMPLE=384
    SEED=42
    OVERSAMPLE_POOL=1200

    # Parameter retrieval (BARU utk Kondisi B)
    INDEX_POOL=8000
    # ^ jumlah kandidat jawaban (sebelum chunking) yang di-sample utk index
    #   FAISS. Mulai kecil (5000-10000) utk dev/pilot run, naikkan (mis.
    #   50000+) utk run resmi supaya korpus lebih representatif.
    TOP_K=5
    TOKEN_CHUNK_LIMIT=400
    INDEX_CACHE_DIR=index_cache
    EMBED_MODEL=all-MiniLM-L6-v2

    # Provider LLM: openai / anthropic / local / ollama -- pola SAMA
    # PERSIS dengan Kondisi A, tinggal ganti flag, tidak ada perubahan
    # struktur kode.
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

CARA PAKAI
----------
    # Run pertama akan build index (lama), run berikutnya pakai cache (cepat)
    python b_condition_b_rag.py

    # Pilot run cepat dgn sample kecil + index kecil + provider dev (ollama)
    python b_condition_b_rag.py --provider ollama --model qwen2.5:1.5b --n-sample 10 --index-pool 2000

    # Paksa rebuild index (mis. setelah ganti INDEX_POOL atau data sumber berubah)
    python b_condition_b_rag.py --rebuild-index

Skrip ini RESUMABLE (pola sama dgn Kondisi A): kalau terhenti di tengah
jalan, jalankan ulang command yang sama -- pertanyaan yang sudah diproses
(tercatat di file output) otomatis dilewati. Index yang sudah di-cache TIDAK
di-rebuild ulang kecuali --rebuild-index dipakai.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from dotenv import load_dotenv

load_dotenv()

# Import modul LLM bersama (llm/) di root repo -- dipakai SEMUA kondisi
# (A/B/C) supaya cara panggil provider LLM & struktur prompt dasar tidak
# terduplikasi/berisiko diam-diam berbeda antar file kondisi.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm.citations import extract_citations
from llm.client_factory import get_llm_client
from llm.prompts import build_rag_messages

TOKEN_LIMIT = 2048  # kriteria eksklusi pertanyaan evaluasi, sama dgn Kondisi A

# Dibind ke logger.info di dalam main() setelah setup_logging() dipanggil.
log = print

LOG_DIR = Path("logs")


def setup_logging(provider: str, model: str) -> tuple[logging.Logger, Path]:
    """Identik pola Kondisi A: 1 file log timestamped per run + tampil di
    console, nama file mengandung provider+model."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
    log_path = LOG_DIR / f"{ts}_{provider}_{safe_model}.log"

    logger = logging.getLogger("condition_b")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger, log_path


def append_run_history(record: dict):
    """Sama pola dgn Kondisi A -- satu baris ringkasan per run di
    logs/run_history.jsonl, dipakai utk tabel perbandingan A vs B vs C."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / "run_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return history_path


# ---------------------------------------------------------------------
# 1-4. Sampel pertanyaan evaluasi + ground truth
# (IDENTIK Kondisi A -- disalin apa adanya supaya sample pertanyaan
#  evaluasi reproducible sama persis kalau SEED/N_SAMPLE/OVERSAMPLE_POOL
#  disamakan, untuk perbandingan yang adil antar kondisi)
# ---------------------------------------------------------------------

def get_candidate_questions(con: duckdb.DuckDBPyConnection, questions_parquet: str,
                             answers_parquet: str, oversample_pool: int, seed: int) -> pd.DataFrame:
    log(f"[1/8] Mengambil oversample pool ({oversample_pool} kandidat pertanyaan "
          f"evaluasi yang SUDAH dipastikan punya accepted answer matching)...")
    query = f"""
        SELECT Id, Title, Body, Tags, AcceptedAnswerId, ViewCount, Score
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
              AND AcceptedAnswerId IN (
                  SELECT DISTINCT Id FROM read_parquet('{answers_parquet}')
              )
        )
        WHERE rn = 1
        USING SAMPLE {oversample_pool} ROWS (reservoir, {seed})
    """
    df = con.execute(query).df()
    log(f"      Oversample pool diambil: {len(df):,} kandidat")
    return df


def filter_by_token_limit(df: pd.DataFrame, limit: int = TOKEN_LIMIT) -> pd.DataFrame:
    import tiktoken

    log(f"[2/8] Menghitung token (title+body+tags) & filter <= {limit} token "
          f"(kriteria sama dgn Kondisi A)...")
    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(row):
        text = f"{row['Title']} {row['Body']} {row['Tags']}"
        return len(enc.encode(str(text)))

    df = df.copy()
    df["n_tokens"] = df.apply(count_tokens, axis=1)
    before = len(df)
    df_filtered = df[df["n_tokens"] <= limit]
    excluded = before - len(df_filtered)
    log(f"      {before:,} kandidat -> {len(df_filtered):,} lolos filter token "
          f"({excluded:,} dikecualikan, {excluded/before*100:.1f}%)")
    return df_filtered


def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """NESTED SAMPLING -- IDENTIK Kondisi A (lihat komentar lengkap di
    a_baseline_replication.py). Permutasi SEKALI dgn seed tetap, ambil N
    pertama sbg PREFIX -- menjamin sample n=30 SUBSET PERSIS dari n=100/n=384,
    SELAMA oversample_pool & seed identik di semua run/kondisi."""
    log(f"[3/8] Mengambil {n} pertanyaan evaluasi PERTAMA dari permutasi tetap "
          f"(seed={seed}) -- prefix ini NESTED thd n_sample lain...")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df_permuted) < n:
        log(f"      [WARN] Populasi hasil filter ({len(df_permuted)}) < target sample ({n}), "
              f"pakai semua yang tersedia.")
        return df_permuted
    return df_permuted.head(n).reset_index(drop=True)


def get_accepted_answers(con: duckdb.DuckDBPyConnection, answers_parquet: str,
                          accepted_answer_ids: list) -> pd.DataFrame:
    log("[4/8] Mengambil teks accepted answer (ground truth) dari Answers parquet...")
    ids_str = ",".join(str(int(i)) for i in accepted_answer_ids)
    query = f"""
        SELECT Id AS AcceptedAnswerId, Body AS AcceptedAnswerBody
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{answers_parquet}')
            WHERE Id IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    log(f"      Ditemukan {len(df):,} accepted answer")
    return df


# ---------------------------------------------------------------------
# 5. Bangun korpus retrieval (Answers JOIN Questions via ParentId),
#    exclude leakage, chunking per TOKEN_CHUNK_LIMIT token
# ---------------------------------------------------------------------

def build_retrieval_corpus(con: duckdb.DuckDBPyConnection, questions_parquet: str,
                            answers_parquet: str, index_pool: int, seed: int,
                            exclude_question_ids: list, exclude_answer_ids: list) -> pd.DataFrame:
    """Ambil pasangan (Title pertanyaan asal + Body jawaban) sebagai dokumen
    korpus retrieval. Flat (tanpa struktur graph) -- ini yang membedakan
    Kondisi B dari Kondisi C nanti.

    PENCEGAHAN LEAKAGE (ketat): exclude_answer_ids (accepted answer dari
    seluruh sample evaluasi) DAN exclude_question_ids (semua jawaban lain,
    termasuk non-accepted, dari pertanyaan mana pun yang sedang dievaluasi)
    dibuang SEBELUM sampling -- supaya index retrieval tidak pernah bisa
    menemukan kembali jawaban dari pertanyaan yang sedang diprediksi.
    """
    log(f"[5/8] Membangun korpus retrieval ({index_pool} kandidat jawaban, "
          f"exclude {len(exclude_answer_ids)} accepted answer + jawaban lain dari "
          f"{len(exclude_question_ids)} pertanyaan evaluasi -- cegah leakage)...")

    excl_q_str = ",".join(str(int(i)) for i in exclude_question_ids) or "-1"
    excl_a_str = ",".join(str(int(i)) for i in exclude_answer_ids) or "-1"

    query = f"""
        SELECT a.Id AS answer_id, a.ParentId AS question_id, a.Score AS answer_score,
               q.Title AS question_title, q.Tags AS question_tags, a.Body AS answer_body
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{answers_parquet}')
            WHERE Id NOT IN ({excl_a_str})
              AND ParentId NOT IN ({excl_q_str})
        ) a
        JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
        ) q
        ON a.ParentId = q.Id
        WHERE a.rn = 1 AND q.rn = 1
        USING SAMPLE {index_pool} ROWS (reservoir, {seed})
    """
    df = con.execute(query).df()
    log(f"      Korpus retrieval (sebelum chunking): {len(df):,} dokumen (title+answer)")
    return df


def chunk_corpus(df: pd.DataFrame, token_limit: int) -> pd.DataFrame:
    """Pecah tiap dokumen (title + body) jadi beberapa chunk berurutan
    <= token_limit token (tiktoken cl100k_base, konsisten dgn Kondisi A).
    Dokumen pendek -> 1 chunk. Tanpa overlap (desain sederhana Kondisi B)."""
    import tiktoken

    log(f"[5/8] Chunking korpus (<= {token_limit} token/chunk, tiktoken cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")

    rows = []
    for _, row in df.iterrows():
        doc_text = f"Q: {row['question_title']}\nA: {row['answer_body']}"
        token_ids = enc.encode(doc_text)
        if len(token_ids) <= token_limit:
            chunks = [doc_text]
        else:
            chunks = [
                enc.decode(token_ids[i:i + token_limit])
                for i in range(0, len(token_ids), token_limit)
            ]
        for c_idx, chunk_text in enumerate(chunks):
            rows.append({
                "answer_id": int(row["answer_id"]),
                "question_id": int(row["question_id"]),
                "question_title": row["question_title"],
                "chunk_idx": c_idx,
                "chunk_text": chunk_text,
            })
    chunk_df = pd.DataFrame(rows)
    log(f"      {len(df):,} dokumen -> {len(chunk_df):,} chunk")
    return chunk_df


# ---------------------------------------------------------------------
# 6. Build/Load FAISS index (IndexFlatIP atas embedding ternormalisasi
#    = cosine similarity) -- di-cache ke disk supaya tidak rebuild tiap run
# ---------------------------------------------------------------------

def build_or_load_faiss_index(chunk_df: pd.DataFrame, embed_model, cache_dir: Path,
                               index_pool: int, token_chunk_limit: int, embed_model_name: str,
                               rebuild: bool):
    import faiss

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"idx{index_pool}_tok{token_chunk_limit}_{embed_model_name.replace('/', '_')}"
    index_path = cache_dir / f"{cache_key}.faiss"
    meta_path = cache_dir / f"{cache_key}_meta.pkl"

    if not rebuild and index_path.exists() and meta_path.exists():
        log(f"[6/8] Memuat index FAISS dari cache -> {index_path}")
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            meta_df = pickle.load(f)
        log(f"      Index dimuat: {index.ntotal:,} vektor "
              f"(pakai --rebuild-index kalau korpus/parameter berubah)")
        return index, meta_df

    log(f"[6/8] Membangun index FAISS baru dari {len(chunk_df):,} chunk "
          f"(embedding model: {embed_model_name})...")
    embeddings = embed_model.encode(
        chunk_df["chunk_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    ).astype("float32")
    faiss.normalize_L2(embeddings)  # normalisasi -> inner product == cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunk_df.reset_index(drop=True), f)

    log(f"      Index dibangun & di-cache -> {index_path} ({index.ntotal:,} vektor)")
    return index, chunk_df.reset_index(drop=True)


def retrieve_context(embed_model, index, meta_df: pd.DataFrame, query_text: str, top_k: int):
    """Retrieve top_k chunk paling mirip (cosine similarity) untuk 1 query."""
    import faiss

    q_emb = embed_model.encode([query_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = meta_df.iloc[idx]
        results.append({
            "answer_id": int(row["answer_id"]),
            "question_id": int(row["question_id"]),
            "question_title": row["question_title"],
            "chunk_text": row["chunk_text"],
            "score": float(score),
        })
    return results


# ---------------------------------------------------------------------
# 7. Prompt retrieval-augmented & pemanggilan model -- lihat
#    llm/prompts.py (build_rag_messages) dan llm/client_factory.py
#    (dipakai bersama semua kondisi eksperimen)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 8. Similarity scoring -- IDENTIK Kondisi A
# ---------------------------------------------------------------------

def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    from sentence_transformers import util
    emb_a = embed_model.encode(str(text_a), convert_to_tensor=True)
    emb_b = embed_model.encode(str(text_b), convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


# ---------------------------------------------------------------------
# Proses per-pertanyaan (dipakai CLI main() DAN dashboard backend/engine_service)
# ---------------------------------------------------------------------

def process_sample(sample_df: pd.DataFrame, llm_client, call_llm_fn, embed_model, index, meta_df: pd.DataFrame,
                    top_k: int, model: str, output_path: Path, already_done: set | None = None,
                    on_progress=None, check_cancel=None, require_citation: bool = True) -> list[dict]:
    """Proses satu-per-satu sample_df: retrieval top-k + prompt RAG + hitung
    cosine similarity, tulis ke output_path (append, resumable). Diekstrak
    dari main() dengan pola SAMA PERSIS dengan process_sample() Kondisi A
    (llm/a_pure_llm/a_baseline_replication.py) supaya CLI dan dashboard
    backend (PLAN_UI_UX.md §5.7) berbagi satu implementasi.

    `on_progress(dict)` / `check_cancel()` -> lihat docstring Kondisi A;
    keduanya opsional supaya CLI (main()) tidak berubah perilaku.

    `require_citation` (default True -- lihat --require-citation di main()):
    kalau True, build_rag_messages() memberi label [SO-<id>] + instruksi
    sitasi (format sama dgn Kondisi C), dan setiap jawaban divalidasi via
    llm/citations.py.extract_citations() sehingga record punya field
    has_citation/cited_source_ids/has_valid_citation/valid_cited_source_ids
    -- dipakai utk membandingkan NF2 (citation compliance) Kondisi B vs C.
    """
    already_done = already_done or set()
    total = len(sample_df)
    results = []
    with open(output_path, "a") as f_out:
        for i, row in sample_df.iterrows():
            if check_cancel is not None and check_cancel():
                log(f"      [cancelled] Dihentikan setelah {len(results)}/{total} pertanyaan.")
                break

            qid = int(row["Id"])
            if qid in already_done:
                continue

            query_text = f"{row['Title']}\n{row['Body']}"
            retrieved = retrieve_context(embed_model, index, meta_df, query_text, top_k)

            messages = build_rag_messages(row["Title"], row["Body"], row["Tags"], retrieved,
                                           require_citation=require_citation)
            try:
                llm_answer = call_llm_fn(llm_client, messages, model)
            except Exception as e:
                log(f"      [{i+1}/{total}] Id={qid} [FAIL] API error: {e}")
                if on_progress:
                    on_progress({"question_id": qid, "index": i, "total": total,
                                 "status": "failed", "error": str(e)})
                continue

            similarity = compute_similarity(embed_model, llm_answer, row["AcceptedAnswerBody"])

            if require_citation:
                has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(llm_answer, retrieved)

            view_count = row.get("ViewCount")
            question_score = row.get("Score")
            record = {
                "question_id": qid,
                "title": row["Title"],
                "tags": row["Tags"],
                "n_tokens": int(row["n_tokens"]),
                # cast eksplisit ke native Python (numpy int64/float64 dari pandas
                # tidak JSON-serializable secara default) -- + default=str di bawah
                # sebagai jaring pengaman kalau ada tipe non-serializable lain lolos.
                "view_count": None if pd.isna(view_count) else int(view_count),
                "question_score": None if pd.isna(question_score) else int(question_score),
                "accepted_answer_id": int(row["AcceptedAnswerId"]),
                "ground_truth_answer": row["AcceptedAnswerBody"],
                "retrieved_context": retrieved,  # utk RAGAS (context precision/recall) nanti
                "prompt_messages": messages,
                "llm_answer": llm_answer,
                "llm_model": model,
                "cosine_similarity": similarity,
                **({
                    "has_citation": has_citation,
                    "cited_source_ids": cited_ids,
                    "has_valid_citation": has_valid_citation,
                    "valid_cited_source_ids": valid_ids,
                } if require_citation else {}),
            }
            try:
                line = json.dumps(record, default=str)
            except Exception as e:
                log(f"      [{i+1}/{total}] Id={qid} [FAIL-WRITE] "
                      f"gagal serialize record ke JSON: {e}")
                if on_progress:
                    on_progress({"question_id": qid, "index": i, "total": total,
                                 "status": "failed", "error": str(e)})
                continue

            n_written = f_out.write(line + "\n")
            f_out.flush()
            os.fsync(f_out.fileno())  # paksa OS tulis ke disk, hindari false-negative di stat() akhir
            results.append(record)
            citation_note = f" citation={record['has_valid_citation']}" if require_citation else ""
            log(f"      [{i+1}/{total}] Id={qid} "
                  f"retrieved={len(retrieved)} similarity={similarity:.3f}{citation_note} "
                  f"(tertulis {n_written} char)")
            if on_progress:
                on_progress({"question_id": qid, "index": i, "total": total,
                             "status": "done", "similarity": similarity, "record": record})
            time.sleep(0.3)
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_output_path(output_dir: str, provider: str, model: str, n_sample: int, seed: int) -> Path:
    """Nama file JSONL otomatis dari provider+model+n_sample+seed -- identik
    pola Kondisi A, format:
    condition_b_{provider}_{safe_model}_n{n_sample}_seed{seed}.jsonl.

    PENTING: n_sample & seed WAJIB di nama file. Kalau tidak, run dengan
    ukuran sample berbeda (mis. pilot n=5 lalu n=30) akan menulis ke file
    yang sama, dan fitur resume di bawah akan mengira sample lama 'sudah
    diproses' lalu menambahkan sample baru ke situ -- file JSONL jadi
    campuran dua sample yang tidak koheren."""
    safe_model = model.replace("/", "-").replace(":", "-").replace(".", "-")
    return Path(output_dir) / f"condition_b_{provider}_{safe_model}_n{n_sample}_seed{seed}.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"),
                         help="Default dari QUESTIONS_PARQUET di .env")
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"),
                         help="Default dari ANSWERS_PARQUET di .env")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "results"),
                         help="Folder output (default dari OUTPUT_DIR di .env). Nama file "
                              "JSONL dibuat OTOMATIS dari provider+model yang dipakai, "
                              "format: condition_b_{provider}_{safe_model}.jsonl")
    parser.add_argument("--output", default=None,
                         help="Override manual path output lengkap (opsional). Kalau diisi, "
                              "auto-naming dari --output-dir/provider/model DIABAIKAN.")
    parser.add_argument("--n-sample", type=int, default=int(os.getenv("N_SAMPLE", 384)),
                         help="Default dari N_SAMPLE di .env. SAMAKAN dgn Kondisi A "
                              "(+ seed + oversample-pool yg sama) utk sample pertanyaan identik.")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)),
                         help="Default dari SEED di .env")
    parser.add_argument("--oversample-pool", type=int, default=int(os.getenv("OVERSAMPLE_POOL", 0)),
                         help="Default: otomatis 3x n_sample kalau tidak diset")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local", "ollama"],
                         help="Default dari LLM_PROVIDER di .env")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                         help="Default dari LLM_MODEL di .env")
    parser.add_argument("--index-pool", type=int, default=int(os.getenv("INDEX_POOL", 8000)),
                         help="Jumlah kandidat jawaban (sebelum chunking) utk index FAISS. "
                              "Default dari INDEX_POOL di .env")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", 5)),
                         help="Jumlah chunk yang diambil per pertanyaan. Default dari TOP_K di .env")
    parser.add_argument("--token-chunk-limit", type=int, default=int(os.getenv("TOKEN_CHUNK_LIMIT", 400)),
                         help="Ukuran maksimal token per chunk korpus. Default dari TOKEN_CHUNK_LIMIT di .env")
    parser.add_argument("--index-cache-dir", default=os.getenv("INDEX_CACHE_DIR", "index_cache"),
                         help="Direktori cache index FAISS. Default dari INDEX_CACHE_DIR di .env")
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"),
                         help="Model sentence-transformers utk indexing/retrieval/scoring. "
                              "HARUS SAMA dgn Kondisi A supaya similarity comparable.")
    parser.add_argument("--rebuild-index", action="store_true",
                         help="Paksa rebuild index FAISS meski cache sudah ada")
    parser.add_argument("--require-citation", action=argparse.BooleanOptionalAction,
                         default=os.getenv("CONDITION_B_REQUIRE_CITATION", "true").strip().lower()
                         in ("1", "true", "yes", "on"),
                         help="Wajibkan model mengutip [SO-<id>] per klaim, format PERSIS sama dgn "
                              "Kondisi C, supaya NF2 (citation compliance) B vs C bisa dibandingkan "
                              "apples-to-apples -- lihat llm/citations.py. Default: AKTIF, dari "
                              "CONDITION_B_REQUIRE_CITATION di .env. Nonaktifkan dgn "
                              "--no-require-citation utk RAG konvensional tanpa instruksi sitasi "
                              "(perilaku Kondisi B sebelum fitur ini ada).")
    args = parser.parse_args()

    global log
    logger, log_path = setup_logging(args.provider, args.model)
    log = logger.info
    run_started_at = datetime.now(timezone.utc)
    log(f"[logging] Log detail run ini -> {log_path}")

    # PENTING utk NESTED SAMPLING -- IDENTIK Kondisi A, lihat komentar
    # lengkap di a_baseline_replication.py. Pool FIXED thd MAX_PLANNED_N_SAMPLE,
    # BUKAN thd args.n_sample lagi -- WAJIB SAMA PERSIS dgn nilai di A & C
    # supaya ketiga kondisi menarik populasi kandidat & sample yang identik
    # (apple-to-apple comparison, lihat catatan metodologi sampling consistency).
    MAX_PLANNED_N_SAMPLE = 384
    oversample_pool = args.oversample_pool if args.oversample_pool > 0 else MAX_PLANNED_N_SAMPLE * 4
    if args.n_sample > MAX_PLANNED_N_SAMPLE:
        log(f"      [WARN] n_sample ({args.n_sample}) > MAX_PLANNED_N_SAMPLE "
            f"({MAX_PLANNED_N_SAMPLE}) -- pool otomatis TIDAK cukup, set "
            f"--oversample-pool manual (mis. {args.n_sample * 4}).")

    # Auto-naming -- pola sama dgn Kondisi A: nama file dibuat dari
    # provider+model yang benar-benar dipakai, kecuali di-override manual.
    if args.output:
        log(f"[config] --output diisi manual -- auto-naming provider/model diabaikan.")
    else:
        args.output = str(build_output_path(args.output_dir, args.provider, args.model,
                                             args.n_sample, args.seed))
        log(f"[config] output auto-generated dari provider='{args.provider}' "
              f"model='{args.model}' -> '{args.output}'")

    if not args.questions_parquet or not args.answers_parquet:
        log("[ERROR] QUESTIONS_PARQUET dan ANSWERS_PARQUET harus diset di .env "
              "(atau lewat --questions-parquet/--answers-parquet)")
        sys.exit(1)

    log(f"[config] questions_parquet = {args.questions_parquet}")
    log(f"[config] answers_parquet   = {args.answers_parquet}")
    log(f"[config] output            = {args.output}")
    log(f"[config] n_sample={args.n_sample} seed={args.seed} "
          f"provider={args.provider} model={args.model}")
    log(f"[config] index_pool={args.index_pool} top_k={args.top_k} "
          f"token_chunk_limit={args.token_chunk_limit} embed_model={args.embed_model}")
    log(f"[config] require_citation={args.require_citation} "
          f"({'[SO-<id>] labels + citation instruction, format sama dgn Kondisi C' if args.require_citation else 'RAG konvensional tanpa instruksi sitasi'})")

    llm_client, call_llm_fn = get_llm_client(args.provider, args.model, log=log)

    con = duckdb.connect()

    # --- 1-4: sample pertanyaan evaluasi + ground truth (identik Kondisi A) ---
    candidates = get_candidate_questions(con, args.questions_parquet, args.answers_parquet,
                                          oversample_pool, args.seed)
    candidates = filter_by_token_limit(candidates)
    sample_df = sample_questions(candidates, args.n_sample, args.seed)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = get_accepted_answers(con, args.answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")

    missing_answer = sample_df["AcceptedAnswerBody"].isna().sum()
    if missing_answer:
        log(f"      [WARN] {missing_answer} pertanyaan tidak ketemu accepted answer-nya "
              f"di Answers parquet -- SEHARUSNYA jarang terjadi setelah semi-join di Step 1.")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

    # --- 5: bangun korpus retrieval (exclude leakage) + chunking ---
    eval_question_ids = sample_df["Id"].astype(int).tolist()
    corpus_df = build_retrieval_corpus(
        con, args.questions_parquet, args.answers_parquet, args.index_pool, args.seed,
        exclude_question_ids=eval_question_ids, exclude_answer_ids=accepted_ids,
    )
    chunk_df = chunk_corpus(corpus_df, args.token_chunk_limit)

    # --- 6: load embedding model + build/load index FAISS ---
    log(f"[6/8] Load model embedding ({args.embed_model})...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model)

    index, meta_df = build_or_load_faiss_index(
        chunk_df, embed_model, Path(args.index_cache_dir),
        args.index_pool, args.token_chunk_limit, args.embed_model, args.rebuild_index,
    )

    # --- Resume support: baca Id yang sudah pernah diproses ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["question_id"])
                except Exception:
                    continue
        log(f"      [resume] {len(already_done)} pertanyaan sudah diproses sebelumnya, akan dilewati")

    # --- 7-8: retrieval + prompting + similarity scoring ---
    log(f"[7/8] Retrieval top-{args.top_k} + prompting {args.model} untuk "
          f"{len(sample_df)} pertanyaan...")

    process_sample(sample_df, llm_client, call_llm_fn, embed_model, index, meta_df, args.top_k,
                    args.model, output_path, already_done, require_citation=args.require_citation)

    log("[8/8] Selesai memproses seluruh sample.")

    # --- Ringkasan akhir (pola sama dgn Kondisi A) ---
    file_exists = output_path.exists()
    file_size = output_path.stat().st_size if file_exists else -1
    log(f"[debug] Cek file output: exists={file_exists} "
          f"absolute_path={output_path.resolve()} size_bytes={file_size}")
    if not file_exists or file_size == 0:
        log("\n[ERROR] Tidak ada hasil tersimpan sama sekali -- cek pesan [FAIL]/[FAIL-WRITE] di atas.")
        history_path = append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "condition": "B",
            "status": "no_results",
            "provider": args.provider,
            "model": args.model,
            "n_sample_target": args.n_sample,
            "seed": args.seed,
            "index_pool": args.index_pool,
            "top_k": args.top_k,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        return

    results_df = pd.read_json(output_path, lines=True)
    if "cosine_similarity" not in results_df.columns or len(results_df) == 0:
        log("\n[ERROR] File hasil ada tapi tidak berisi record valid.")
        history_path = append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "condition": "B",
            "status": "no_valid_records",
            "provider": args.provider,
            "model": args.model,
            "n_sample_target": args.n_sample,
            "seed": args.seed,
            "index_pool": args.index_pool,
            "top_k": args.top_k,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        return

    log("\n" + "=" * 70)
    log("RINGKASAN HASIL KONDISI B (Conventional RAG)")
    log("=" * 70)
    log(f"Total pertanyaan diproses : {len(results_df)}")
    log(f"Cosine similarity rata-rata: {results_df['cosine_similarity'].mean():.4f}")
    log(f"Cosine similarity median   : {results_df['cosine_similarity'].median():.4f}")
    log(f"% similarity > 0.5         : {(results_df['cosine_similarity'] > 0.5).mean()*100:.1f}%")

    citation_summary = {}
    if args.require_citation and "has_valid_citation" in results_df.columns:
        pct_citation = results_df["has_citation"].mean() * 100
        pct_valid_citation = results_df["has_valid_citation"].mean() * 100
        log(f"% jawaban dgn >=1 kutipan format cocok  : {pct_citation:.1f}%")
        log(f"% jawaban dgn >=1 kutipan VALID (NF2)   : {pct_valid_citation:.1f}% "
            f"(pembanding langsung terhadap Kondisi C)")
        citation_summary = {
            "pct_with_citation": round(float(pct_citation), 1),
            "pct_with_valid_citation": round(float(pct_valid_citation), 1),
        }
    log(f"\nHasil lengkap tersimpan -> {output_path}")

    history_path = append_run_history({
        "run_started_at": run_started_at.isoformat(),
        "condition": "B",
        "status": "success",
        "provider": args.provider,
        "model": args.model,
        "n_sample_target": args.n_sample,
        "n_processed": len(results_df),
        "seed": args.seed,
        "index_pool": args.index_pool,
        "top_k": args.top_k,
        "token_chunk_limit": args.token_chunk_limit,
        "require_citation": args.require_citation,
        "cosine_similarity_mean": round(float(results_df["cosine_similarity"].mean()), 4),
        "cosine_similarity_median": round(float(results_df["cosine_similarity"].median()), 4),
        "pct_similarity_above_0_5": round(float((results_df["cosine_similarity"] > 0.5).mean() * 100), 1),
        **citation_summary,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
    })
    log(f"[logging] Ringkasan run dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())