"""
d_lightrag.py
=====================================
Fase 4 -- Kondisi D (Dual-Level Retrieval, ADAPTASI mekanisme LightRAG,
Guo dkk.) pada protokol evaluasi. Berjalan DI ATAS KG Neo4j & FAISS cache
YANG SAMA PERSIS dengan Kondisi C -- TIDAK membangun graf baru, TIDAK
melakukan ekstraksi entitas via LLM.

PENTING -- INI BUKAN LightRAG asli, INI ADAPTASI
------------------------------------------------------------
Node/edge di KG Kondisi C (Question/Answer/Tag/User dengan edge
HAS_ACCEPTED_ANSWER/HAS_ANSWER/TAGGED_WITH/IS_RELATED_TO/TAG_COOCCUR/
EMBED_SIM/AUTHOR_TRUST) dipakai APA ADANYA -- TIDAK melakukan ekstraksi
entitas generik ala LightRAG asli. Ini KEPUTUSAN METODOLOGIS yang
disengaja supaya Kondisi C vs D jadi ablasi terkontrol MURNI pada
mekanisme retrieval (trust-weighted fusion vs dual-level tanpa trust),
bukan ablasi pada data/graf yang dipakai.

METODOLOGI
------------------------------------------------------------
Sama seperti Kondisi A/B/C (supaya perbandingan adil):
  - Kriteria inklusi/eksklusi pertanyaan evaluasi & sampling: IDENTIK
    (get_candidate_questions/filter_by_token_limit/sample_questions/
    get_accepted_answers/get_all_answer_ids_for_questions DISALIN APA
    ADANYA dari c_graphrag.py -- lihat komentar di sana).
  - Model embedding: SAMA (all-MiniLM-L6-v2).
  - Struktur prompt 4-turn dasar: SAMA, lihat llm/prompts.py
    (build_lightrag_messages).
  - LLM dipanggil via factory provider yang SAMA PERSIS (get_llm_client).
  - Neo4j connection & FAISS cache: SAMA PERSIS Kondisi C (connect_neo4j/
    load_faiss_cache DISALIN APA ADANYA) -- Kondisi D HANYA MEMBACA graf &
    index yang sudah ada, TIDAK membangun apa pun baru.

Yang BERBEDA dari Kondisi C -- mekanisme DUAL-LEVEL RETRIEVAL (adaptasi
LightRAG) menggantikan traverse_graph + semantic_expansion + fuse_and_rank:

  1. LOW-LEVEL RETRIEVAL (retrieve_low_level) -- analog "low-level keyword/
     entity retrieval" LightRAG: anchor_via_vector_search() SAMA PERSIS
     Kondisi C -> top N_LOW_LEVEL Question anchor, ambil jawaban 1-hop
     langsung (HAS_ACCEPTED_ANSWER|HAS_ANSWER). Skor low-level = skor
     cosine similarity FAISS anchor-nya, BUKAN edge_weight/trustScore.

  2. HIGH-LEVEL RETRIEVAL (retrieve_high_level) -- analog "high-level
     theme/community retrieval" LightRAG: dari anchor yang sama, telusuri
     edge TAG_COOCCUR & IS_RELATED_TO (2-hop) ke Question lain, ambil
     jawabannya. Skor high-level = skor similarity/Jaccard MENTAH dari edge
     itu sendiri (r2.weight) -- TIDAK dikombinasikan dengan trustScore
     Answer sama sekali.

  3. FUSION (fuse_dual_level) -- dedup by answer_id (simpan skor relevansi
     TERTINGGI, selalu murni skor similarity, TIDAK PERNAH ada trustScore/
     is_accepted yang ikut mempengaruhi skor), label source_stage
     "low_level"/"high_level" (metadata transparansi, BUKAN trust), sort
     descending by skor relevansi, cap top-K, chunking ke
     TOKEN_CHUNK_LIMIT (disalin dari fuse_and_rank() Kondisi C).

  4. GROUNDING ABLATION -- build_lightrag_messages() (llm/prompts.py)
     punya switch `require_grounding` (default True):
       - True : dual-constraint SAMA PERSIS Kondisi C (grounding + citation).
       - False: HANYA instruksi citation (identik Kondisi B
         require_citation=True), TANPA instruksi grounding.
     Switch ini isolasi variabel GROUNDING CONSTRAINT terpisah dari variabel
     mekanisme retrieval (trust-weighted vs dual-level).

PENCEGAHAN LEAKAGE -- SAMA PERSIS Kondisi C
------------------------------------------------------------
  1. Anchor vector search: EXCLUDE eval question's own id.
  2. Low-level 1-hop & high-level 2-hop: EXCLUDE eval_question_id (sbg
     via_question) DAN accepted_answer_id + semua answer_id lain milik
     eval question tsb.

INSTALL DEPENDENCY -- SAMA Kondisi C
-------------------
    pip install neo4j faiss-cpu

SETUP .env (WAJIB, sama dgn Kondisi C, ditambah 3 baris baru)
----------------------------------------------------------------------------
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j
    KG_WORKSPACE_DIR=../../01_data_cleaning/_kg_workspace

    N_LOW_LEVEL=3
    N_HIGH_LEVEL=3
    LIGHTRAG_REQUIRE_GROUNDING=true   # atau "false"

CARA PAKAI
----------
    python d_lightrag.py
    python d_lightrag.py --n-sample 30 --seed 42
    python d_lightrag.py --require-grounding --n-sample 30      # dual-constraint (default)
    python d_lightrag.py --no-require-grounding --n-sample 30   # hanya citation, tanpa grounding

Prasyarat WAJIB: sama dgn Kondisi C -- 11_densify_embedding_similarity.py
--stage all sudah pernah dijalankan sukses (index FAISS + embedding harus
ada di KG_WORKSPACE_DIR). Skrip ini TIDAK membangun ulang KG maupun edge
densifikasi -- murni membaca graf & index yang sudah ada.

Resumable, pola sama dgn Kondisi A/B/C.
"""

import argparse
import json
import logging
import os
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm.client_factory import get_llm_client
from llm.prompts import build_lightrag_messages

TOKEN_LIMIT = 2048  # kriteria eksklusi pertanyaan evaluasi, IDENTIK Kondisi A/B/C
EMBED_DIM = 384  # all-MiniLM-L6-v2, HARUS sama dgn 11_densify_embedding_similarity.py

log = print
LOG_DIR = Path("logs")


def setup_logging(provider: str, model: str) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
    log_path = LOG_DIR / f"{ts}_{provider}_{safe_model}.log"

    logger = logging.getLogger("condition_d")
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


def append_run_history(record: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / "run_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return history_path


# ---------------------------------------------------------------------
# 1-4. Sampel pertanyaan evaluasi + ground truth
# DISALIN APA ADANYA dari c_graphrag.py -- JANGAN diubah logikanya.
# ---------------------------------------------------------------------

def get_candidate_questions(con, questions_parquet: str, answers_parquet: str,
                             oversample_pool: int, seed: int) -> pd.DataFrame:
    log(f"[1/9] Mengambil oversample pool ({oversample_pool} kandidat)...")
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
    log(f"[2/9] Menghitung token & filter <= {limit} token...")
    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(row):
        text = f"{row['Title']} {row['Body']} {row['Tags']}"
        return len(enc.encode(str(text)))

    df = df.copy()
    df["n_tokens"] = df.apply(count_tokens, axis=1)
    before = len(df)
    df_filtered = df[df["n_tokens"] <= limit]
    log(f"      {before:,} -> {len(df_filtered):,} lolos filter token")
    return df_filtered


def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """NESTED SAMPLING -- IDENTIK Kondisi A/B/C."""
    log(f"[3/9] Mengambil {n} pertanyaan evaluasi PERTAMA dari permutasi tetap "
          f"(seed={seed}) -- prefix ini NESTED thd n_sample lain...")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df_permuted) < n:
        log(f"      [WARN] Populasi ({len(df_permuted)}) < target ({n}), pakai semua.")
        return df_permuted
    return df_permuted.head(n).reset_index(drop=True)


def get_accepted_answers(con, answers_parquet: str, accepted_answer_ids: list) -> pd.DataFrame:
    log("[4/9] Mengambil teks accepted answer (ground truth)...")
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


def get_all_answer_ids_for_questions(con, answers_parquet: str, question_ids: list) -> dict:
    """Utk leakage prevention: peta question_id -> list SEMUA answer_id
    miliknya (bukan cuma accepted), sama ketatnya dgn Kondisi B/C."""
    log("[4/9] Memetakan SEMUA answer_id per pertanyaan evaluasi (utk exclusion)...")
    ids_str = ",".join(str(int(i)) for i in question_ids)
    query = f"""
        SELECT ParentId AS question_id, Id AS answer_id
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{answers_parquet}')
            WHERE ParentId IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    mapping = df.groupby("question_id")["answer_id"].apply(list).to_dict()
    return mapping


# ---------------------------------------------------------------------
# Neo4j + FAISS setup -- DISALIN APA ADANYA dari c_graphrag.py
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
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    logger(f"[5/9] Terhubung ke Neo4j ({uri}, database={database})")
    return driver, database


def load_faiss_cache(kg_workspace_dir: Path, logger):
    """Reuse index & embedding cache Kondisi C (11_densify_embedding_similarity.py
    --stage edges). TIDAK membangun ulang di sini -- kalau cache tidak ada,
    ABORT dengan pesan jelas."""
    import faiss

    index_path = kg_workspace_dir / "question_embeddings_ivf.faiss"
    ids_path = kg_workspace_dir / "question_embeddings_ivf_ids.npy"
    emb_path = kg_workspace_dir / "question_embeddings.f32"
    ckpt_path = kg_workspace_dir / "embed_checkpoint.json"

    if not (index_path.exists() and ids_path.exists() and emb_path.exists() and ckpt_path.exists()):
        raise SystemExit(
            f"[ABORT] Cache FAISS/embedding tidak ditemukan di {kg_workspace_dir}. "
            f"Jalankan dulu 11_densify_embedding_similarity.py --stage all "
            f"(di folder 01_data_cleaning/) sebelum menjalankan Kondisi D."
        )

    n_total = json.loads(ckpt_path.read_text())["next_row"]
    ids_arr = np.load(ids_path)
    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_total, EMBED_DIM))
    index = faiss.read_index(str(index_path))
    logger(f"[5/9] FAISS index & embedding cache dimuat: {index.ntotal:,} vektor "
           f"dari {kg_workspace_dir}")

    id_to_row = {int(qid): i for i, qid in enumerate(ids_arr)}
    return index, ids_arr, embeddings, id_to_row


def embed_query(text: str, sbert_model) -> np.ndarray:
    emb = sbert_model.encode([text], convert_to_numpy=True).astype("float32")
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    return emb / norm


# ---------------------------------------------------------------------
# ENTITY ANCHORING -- vector search, DISALIN APA ADANYA dari c_graphrag.py
# (dipakai ulang sbg titik masuk low-level DAN high-level retrieval)
# ---------------------------------------------------------------------

def anchor_via_vector_search(query_emb: np.ndarray, index, ids_arr: np.ndarray,
                              top_n: int, exclude_ids: set) -> list:
    """Cari top_n Question id paling mirip via FAISS, EXCLUDE eval
    question's own id (leakage prevention #1)."""
    search_n = top_n + len(exclude_ids) + 5
    scores, indices = index.search(query_emb, search_n)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        qid = int(ids_arr[idx])
        if qid in exclude_ids:
            continue
        results.append((qid, float(score)))
        if len(results) >= top_n:
            break
    return results  # list of (question_id, cosine_score)


# ---------------------------------------------------------------------
# LOW-LEVEL RETRIEVAL -- jawaban 1-hop langsung dari anchor. Skor = cosine
# similarity anchor-nya (dari FAISS search), BUKAN edge_weight/trustScore.
# ---------------------------------------------------------------------

def retrieve_low_level(driver, database, anchors: list, exclude_answer_ids: set) -> list:
    """`anchors`: list of (question_id, cosine_score) dari anchor_via_vector_search().
    Query 1-hop DISALIN dari traverse_graph() Kondisi C bagian 1-hop, TAPI
    skor kandidat diambil dari anchor_score (similarity FAISS), bukan
    r.weight/trustScore."""
    if not anchors:
        return []
    anchor_score = {qid: score for qid, score in anchors}
    anchor_ids = list(anchor_score.keys())
    exclude_a = list(exclude_answer_ids) or [-1]

    candidates = []
    with driver.session(database=database) as session:
        result = session.run(
            """
            UNWIND $anchor_ids AS anchor_id
            MATCH (anchor:Question {id: anchor_id})-[r:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN anchor_id AS via_question_id, anchor.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, 1 AS hop,
                   type(r) AS rel_type
            """,
            anchor_ids=anchor_ids, exclude_a=exclude_a,
        )
        for record in result:
            c = dict(record)
            c["relevance_score"] = anchor_score.get(c["via_question_id"], 0.0)
            candidates.append(c)

    return candidates


# ---------------------------------------------------------------------
# HIGH-LEVEL RETRIEVAL -- 2-hop via TAG_COOCCUR/IS_RELATED_TO. Skor = skor
# similarity/Jaccard MENTAH dari edge itu sendiri (r2.weight), TIDAK
# dikombinasikan dengan trustScore Answer.
# ---------------------------------------------------------------------

def retrieve_high_level(driver, database, anchors: list, exclude_question_ids: set,
                         exclude_answer_ids: set, n_high_level: int) -> list:
    """Query 2-hop DISALIN POLA-nya dari traverse_graph() Kondisi C bagian
    IS_RELATED_TO|TAG_COOCCUR|EMBED_SIM, TAPI skor kandidat = r2.weight
    MENTAH (edge relevansi/tema Question<->Question), bukan hasil kali
    dengan r3.weight/trustScore Answer seperti di Kondisi C. Dibatasi ke
    N_HIGH_LEVEL Question unik tertinggi skornya."""
    if not anchors:
        return []
    anchor_ids = [qid for qid, _ in anchors]
    exclude_q = list(exclude_question_ids) or [-1]
    exclude_a = list(exclude_answer_ids) or [-1]

    with driver.session(database=database) as session:
        result = session.run(
            """
            UNWIND $anchor_ids AS anchor_id
            MATCH (anchor:Question {id: anchor_id})-[r2:IS_RELATED_TO|TAG_COOCCUR]->(q2:Question)
            WHERE NOT q2.id IN $exclude_q
            WITH DISTINCT q2, r2.weight AS relevance_score
            ORDER BY relevance_score DESC
            LIMIT $n_high_level
            MATCH (q2)-[r3:HAS_ACCEPTED_ANSWER|HAS_ANSWER]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN q2.id AS via_question_id, q2.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, 2 AS hop,
                   'TAG_COOCCUR|IS_RELATED_TO' AS rel_type, relevance_score
            """,
            anchor_ids=anchor_ids, exclude_q=exclude_q, exclude_a=exclude_a,
            n_high_level=n_high_level,
        )
        candidates = [dict(record) for record in result]

    return candidates


# ---------------------------------------------------------------------
# FUSION -- dedup by answer_id (keep HIGHEST relevance score -- selalu
# murni skor similarity, TIDAK PERNAH ada trustScore/is_accepted yang ikut),
# label source_stage, sort desc, cap top-K, chunking.
# ---------------------------------------------------------------------

def fuse_dual_level(low_level_candidates: list, high_level_candidates: list,
                     top_k: int, token_chunk_limit: int) -> list:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    all_candidates = []
    for c in low_level_candidates:
        c = dict(c)
        c["source_stage"] = "low_level"
        all_candidates.append(c)
    for c in high_level_candidates:
        c = dict(c)
        c["source_stage"] = "high_level"
        all_candidates.append(c)

    best_by_answer = {}
    for c in all_candidates:
        aid = c["answer_id"]
        if aid not in best_by_answer or c["relevance_score"] > best_by_answer[aid]["relevance_score"]:
            best_by_answer[aid] = c

    ranked = sorted(best_by_answer.values(), key=lambda c: c["relevance_score"], reverse=True)
    top = ranked[:top_k]

    final = []
    for c in top:
        doc_text = f"Q: {c['via_question_title']}\nA: {c['answer_body']}"
        token_ids = enc.encode(doc_text)
        if len(token_ids) > token_chunk_limit:
            doc_text = enc.decode(token_ids[:token_chunk_limit])
        final.append({
            "question_id": int(c["via_question_id"]),
            "answer_id": int(c["answer_id"]),
            "chunk_text": doc_text,
            "relevance_score": round(float(c["relevance_score"] or 0.0), 4),
            "hop": c["hop"],
            "rel_type": c.get("rel_type"),
            "source_stage": c["source_stage"],
        })
    return final


# ---------------------------------------------------------------------
# Citation check -- SAMA PERSIS Kondisi B/C (llm/citations.py)
# ---------------------------------------------------------------------

from llm.citations import CITATION_PATTERN, extract_citations  # noqa: E402,F401


# ---------------------------------------------------------------------
# Similarity scoring -- IDENTIK Kondisi A/B/C
# ---------------------------------------------------------------------

def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    from sentence_transformers import util
    emb_a = embed_model.encode(str(text_a), convert_to_numpy=True, device="cpu")
    emb_b = embed_model.encode(str(text_b), convert_to_numpy=True, device="cpu")
    return float(util.cos_sim(emb_a, emb_b).item())


# ---------------------------------------------------------------------
# Proses per-pertanyaan (dipakai CLI main() DAN dashboard backend/engine_service)
# ---------------------------------------------------------------------

def process_sample(sample_df: pd.DataFrame, llm_client, call_llm_fn, embed_model, driver, database,
                    faiss_index, faiss_ids, faiss_embeddings, id_to_row: dict, all_answer_ids_map: dict,
                    top_k: int, n_low_level: int, n_high_level: int, token_chunk_limit: int,
                    model: str, output_path: Path, already_done: set | None = None,
                    on_progress=None, check_cancel=None, require_grounding: bool = True,
                    ) -> tuple[list[dict], dict]:
    """Pola SAMA PERSIS process_sample() Kondisi C: retrieval (dual-level, bukan
    trust-weighted) + prompt (grounded/ungrounded sesuai require_grounding) +
    hitung cosine similarity + validasi sitasi, tulis ke output_path (append,
    resumable).

    Return (results, stats). `on_progress(dict)` / `check_cancel()` opsional,
    dipakai dashboard backend (engine_service.run_condition_d).
    """
    already_done = already_done or set()
    total = len(sample_df)
    results = []
    n_no_context = 0
    n_no_citation = 0
    n_no_valid_citation = 0
    interrupted = False

    try:
        with open(output_path, "a") as f_out:
            for i, row in sample_df.iterrows():
                if check_cancel is not None and check_cancel():
                    log(f"      [cancelled] Dihentikan setelah {len(results)}/{total} pertanyaan.")
                    break

                qid = int(row["Id"])
                if qid in already_done:
                    continue

                t_query_start = time.time()

                exclude_question_ids = {qid}
                exclude_answer_ids = set(all_answer_ids_map.get(qid, []))
                exclude_answer_ids.add(int(row["AcceptedAnswerId"]))

                if qid in id_to_row:
                    query_emb = faiss_embeddings[id_to_row[qid]:id_to_row[qid] + 1]
                    query_emb = np.ascontiguousarray(query_emb)
                else:
                    query_text = f"{row['Title']} {str(row['Body'])[:1000]}"
                    query_emb = embed_query(query_text, embed_model)

                # --- Entity Anchoring (dipakai ulang sbg titik masuk keduanya) ---
                anchors = anchor_via_vector_search(
                    query_emb, faiss_index, faiss_ids, max(n_low_level, n_high_level),
                    exclude_question_ids,
                )

                # --- Low-level retrieval ---
                low_level_candidates = retrieve_low_level(
                    driver, database, anchors[:n_low_level], exclude_answer_ids,
                )

                # --- High-level retrieval ---
                high_level_candidates = retrieve_high_level(
                    driver, database, anchors[:n_high_level], exclude_question_ids,
                    exclude_answer_ids, n_high_level,
                )

                # --- Fusi + cap top-K + chunking ---
                retrieved = fuse_dual_level(low_level_candidates, high_level_candidates,
                                            top_k, token_chunk_limit)
                retrieval_latency = time.time() - t_query_start
                if not retrieved:
                    n_no_context += 1

                messages = build_lightrag_messages(row["Title"], row["Body"], row["Tags"], retrieved,
                                                    require_grounding=require_grounding)
                try:
                    llm_answer = call_llm_fn(llm_client, messages, model)
                except Exception as e:
                    log(f"      [{i+1}/{total}] Id={qid} [FAIL] API error: {e}")
                    if on_progress:
                        on_progress({"question_id": qid, "index": i, "total": total,
                                     "status": "failed", "error": str(e)})
                    continue

                has_citation, cited_ids, has_valid_citation, valid_ids = extract_citations(llm_answer, retrieved)
                if not has_citation:
                    n_no_citation += 1
                if not has_valid_citation:
                    n_no_valid_citation += 1

                similarity = compute_similarity(embed_model, llm_answer, row["AcceptedAnswerBody"])

                view_count = row.get("ViewCount")
                question_score = row.get("Score")
                record = {
                    "question_id": qid,
                    "title": row["Title"],
                    "tags": row["Tags"],
                    "n_tokens": int(row["n_tokens"]),
                    "view_count": None if pd.isna(view_count) else int(view_count),
                    "question_score": None if pd.isna(question_score) else int(question_score),
                    "accepted_answer_id": int(row["AcceptedAnswerId"]),
                    "ground_truth_answer": row["AcceptedAnswerBody"],
                    "retrieved_context": retrieved,
                    "anchor_question_ids": [
                        {"question_id": int(aid), "similarity": round(float(score), 4)} for aid, score in anchors
                    ],
                    "prompt_messages": messages,
                    "n_low_level_candidates": len(low_level_candidates),
                    "n_high_level_candidates": len(high_level_candidates),
                    "require_grounding": require_grounding,
                    "retrieval_latency_sec": round(retrieval_latency, 3),
                    "llm_answer": llm_answer,
                    "llm_model": model,
                    "cosine_similarity": similarity,
                    "has_citation": has_citation,
                    "cited_source_ids": cited_ids,
                    "has_valid_citation": has_valid_citation,
                    "valid_cited_source_ids": valid_ids,
                }
                try:
                    line = json.dumps(record, default=str)
                except Exception as e:
                    log(f"      [{i+1}/{total}] Id={qid} [FAIL-WRITE] {e}")
                    if on_progress:
                        on_progress({"question_id": qid, "index": i, "total": total,
                                     "status": "failed", "error": str(e)})
                    continue

                f_out.write(line + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())
                results.append(record)
                log(f"      [{i+1}/{total}] Id={qid} low_level={len(low_level_candidates)} "
                    f"high_level={len(high_level_candidates)} context={len(retrieved)} "
                    f"citation={has_citation} valid_citation={has_valid_citation} "
                    f"similarity={similarity:.3f} retrieval={retrieval_latency:.2f}s")

                if retrieval_latency > 15.0:
                    log(f"      [WARN] Id={qid} retrieval_latency={retrieval_latency:.2f}s "
                        f"MELEBIHI NF3 (<=15 detik)")

                if on_progress:
                    on_progress({"question_id": qid, "index": i, "total": total,
                                 "status": "done", "similarity": similarity, "record": record})

                time.sleep(0.3)
    except KeyboardInterrupt:
        interrupted = True
        log("\n[INTERRUPTED] Proses dihentikan manual (Ctrl+C / kill). Jalankan command "
            "yang SAMA PERSIS lagi kapan saja -- otomatis lanjut (resume).")

    stats = {
        "n_no_context": n_no_context,
        "n_no_citation": n_no_citation,
        "n_no_valid_citation": n_no_valid_citation,
        "interrupted": interrupted,
    }
    return results, stats


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_output_path(output_dir: str, provider: str, model: str, n_sample: int, seed: int,
                       require_grounding: bool = True) -> Path:
    """Nama file menyertakan varian grounding supaya run grounded/ungrounded
    tidak saling menimpa file .jsonl satu sama lain."""
    safe_model = model.replace("/", "-").replace(":", "-").replace(".", "-")
    base = f"condition_d_{provider}_{safe_model}_n{n_sample}_seed{seed}"
    suffix = "grounded" if require_grounding else "ungrounded"
    return Path(output_dir) / f"{base}_{suffix}.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "results"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-sample", type=int, default=int(os.getenv("N_SAMPLE", 384)))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)))
    parser.add_argument("--oversample-pool", type=int, default=int(os.getenv("OVERSAMPLE_POOL", 0)))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local", "ollama"])
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", 5)),
                         help="Jumlah item konteks final (setelah fusi dual-level)")
    parser.add_argument("--token-chunk-limit", type=int, default=int(os.getenv("TOKEN_CHUNK_LIMIT", 400)))
    parser.add_argument("--n-low-level", type=int, default=int(os.getenv("N_LOW_LEVEL", 3)),
                         help="Jumlah Question anchor utk low-level retrieval (1-hop jawaban)")
    parser.add_argument("--n-high-level", type=int, default=int(os.getenv("N_HIGH_LEVEL", 3)),
                         help="Jumlah Question unik tertinggi skornya utk high-level retrieval (2-hop tema/tag)")
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--kg-workspace-dir", default=os.getenv("KG_WORKSPACE_DIR",
                         "../../01_data_cleaning/_kg_workspace"))
    parser.add_argument("--require-grounding", action=argparse.BooleanOptionalAction,
                         default=os.getenv("LIGHTRAG_REQUIRE_GROUNDING", "true").strip().lower() == "true",
                         help="True (default): dual-constraint grounding+citation SAMA PERSIS Kondisi C. "
                              "False: hanya instruksi citation (identik Kondisi B require_citation=True), "
                              "TANPA instruksi grounding.")
    args = parser.parse_args()

    global log
    logger, log_path = setup_logging(args.provider, args.model)
    log = logger.info
    run_started_at = datetime.now(timezone.utc)
    log(f"[logging] Log detail run ini -> {log_path}")

    MAX_PLANNED_N_SAMPLE = 384
    oversample_pool = args.oversample_pool if args.oversample_pool > 0 else MAX_PLANNED_N_SAMPLE * 4
    if args.n_sample > MAX_PLANNED_N_SAMPLE:
        log(f"      [WARN] n_sample ({args.n_sample}) > MAX_PLANNED_N_SAMPLE "
            f"({MAX_PLANNED_N_SAMPLE}) -- pool otomatis TIDAK cukup, set "
            f"--oversample-pool manual (mis. {args.n_sample * 4}).")

    if args.output:
        log("[config] --output diisi manual -- auto-naming diabaikan.")
    else:
        args.output = str(build_output_path(args.output_dir, args.provider, args.model,
                                             args.n_sample, args.seed, args.require_grounding))
        log(f"[config] output auto-generated -> '{args.output}'")

    if not args.questions_parquet or not args.answers_parquet:
        log("[ERROR] QUESTIONS_PARQUET dan ANSWERS_PARQUET harus diset di .env")
        sys.exit(1)

    log(f"[config] n_sample={args.n_sample} seed={args.seed} provider={args.provider} model={args.model}")
    log(f"[config] top_k={args.top_k} n_low_level={args.n_low_level} "
        f"n_high_level={args.n_high_level} token_chunk_limit={args.token_chunk_limit}")
    log(f"[config] require_grounding={args.require_grounding} "
        f"({'dual-constraint grounding+citation' if args.require_grounding else 'citation only, no grounding'})")

    llm_client, call_llm_fn = get_llm_client(args.provider, args.model, log=log)

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    candidates = get_candidate_questions(con, args.questions_parquet, args.answers_parquet,
                                          oversample_pool, args.seed)
    candidates = filter_by_token_limit(candidates)
    sample_df = sample_questions(candidates, args.n_sample, args.seed)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = get_accepted_answers(con, args.answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

    eval_question_ids = sample_df["Id"].astype(int).tolist()
    all_answer_ids_map = get_all_answer_ids_for_questions(con, args.answers_parquet, eval_question_ids)

    driver, database = connect_neo4j(log)
    kg_workspace_dir = Path(args.kg_workspace_dir)
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = load_faiss_cache(kg_workspace_dir, log)

    log(f"[6/9] Load model embedding ({args.embed_model})...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")

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
        log(f"      [resume] {len(already_done)} pertanyaan sudah diproses sebelumnya")

    log(f"[7/9] Dual-level retrieval (low-level+high-level) + prompting {args.model} "
        f"untuk {len(sample_df)} pertanyaan...")

    results, stats = process_sample(
        sample_df, llm_client, call_llm_fn, embed_model, driver, database,
        faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
        args.top_k, args.n_low_level, args.n_high_level, args.token_chunk_limit,
        args.model, output_path, already_done,
        require_grounding=args.require_grounding,
    )
    interrupted = stats["interrupted"]

    driver.close()
    if interrupted:
        log("[8/9] Diinterupsi sebelum semua sample selesai -- lihat catatan di atas.")
    else:
        log("[8/9] Selesai memproses seluruh sample.")

    file_exists = output_path.exists()
    file_size = output_path.stat().st_size if file_exists else -1
    if not file_exists or file_size == 0:
        log("\n[ERROR] Tidak ada hasil tersimpan sama sekali.")
        history_path = append_run_history({
            "run_started_at": run_started_at.isoformat(), "condition": "D", "status": "no_results",
            "provider": args.provider, "model": args.model, "n_sample_target": args.n_sample,
            "seed": args.seed, "output_path": str(output_path), "log_path": str(log_path),
            "require_grounding": args.require_grounding,
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        return

    results_df = pd.read_json(output_path, lines=True)
    log("\n" + "=" * 70)
    log(f"[9/9] RINGKASAN HASIL KONDISI D (Dual-Level Retrieval, adaptasi LightRAG) "
        f"-- varian: {'grounded' if args.require_grounding else 'ungrounded'}")
    log("=" * 70)
    log(f"Total pertanyaan diproses      : {len(results_df)}")
    log(f"Cosine similarity rata-rata    : {results_df['cosine_similarity'].mean():.4f}")
    log(f"Cosine similarity median       : {results_df['cosine_similarity'].median():.4f}")
    log(f"% similarity > 0.5             : {(results_df['cosine_similarity'] > 0.5).mean()*100:.1f}%")
    pct_citation = results_df["has_citation"].mean() * 100
    pct_valid_citation = results_df["has_valid_citation"].mean() * 100
    log(f"% jawaban dgn >=1 kutipan format cocok  : {pct_citation:.1f}%")
    log(f"% jawaban dgn >=1 kutipan VALID (NF2)   : {pct_valid_citation:.1f}%")
    avg_latency = results_df["retrieval_latency_sec"].mean()
    p95_latency = results_df["retrieval_latency_sec"].quantile(0.95)
    log(f"Retrieval latency -- avg: {avg_latency:.2f}s, p95: {p95_latency:.2f}s "
        f"[NF3 target: <=15s] {'PASS' if avg_latency <= 15.0 else 'FAIL'}")
    log(f"\nHasil lengkap tersimpan -> {output_path}")

    history_path = append_run_history({
        "run_started_at": run_started_at.isoformat(), "condition": "D",
        "status": "interrupted" if interrupted else "success",
        "provider": args.provider, "model": args.model, "n_sample_target": args.n_sample,
        "n_processed": len(results_df), "seed": args.seed,
        "top_k": args.top_k, "n_low_level": args.n_low_level,
        "n_high_level": args.n_high_level, "require_grounding": args.require_grounding,
        "cosine_similarity_mean": round(float(results_df["cosine_similarity"].mean()), 4),
        "cosine_similarity_median": round(float(results_df["cosine_similarity"].median()), 4),
        "pct_similarity_above_0_5": round(float((results_df["cosine_similarity"] > 0.5).mean() * 100), 1),
        "pct_with_citation": round(float(pct_citation), 1),
        "pct_with_valid_citation": round(float(pct_valid_citation), 1),
        "avg_retrieval_latency_sec": round(float(avg_latency), 3),
        "p95_retrieval_latency_sec": round(float(p95_latency), 3),
        "output_path": str(output_path), "log_path": str(log_path),
        "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
    })
    log(f"[logging] Ringkasan run dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())
