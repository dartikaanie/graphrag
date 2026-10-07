"""
c_condition_c_graphrag.py
=====================================
Fase 3 -- Kondisi C (GraphRAG Framework, trust-weighted Knowledge Graph)
pada protokol evaluasi tiga-kondisi. Pembanding utama bagi Kondisi A (LLM
murni) dan Kondisi B (RAG konvensional/flat dense retrieval).

METODOLOGI
------------------------------------------------------------
Sama seperti Kondisi A & B (supaya perbandingan adil):
  - Kriteria inklusi/eksklusi pertanyaan evaluasi & sampling: IDENTIK
    (fungsi disalin apa adanya dari b_condition_b_rag.py -- accepted
    answer wajib matched, <= TOKEN_LIMIT token, oversample pool reservoir
    sample seed sama).
  - Model embedding: SAMA (all-MiniLM-L6-v2) supaya cosine similarity
    A vs B vs C comparable.
  - Struktur prompt 4-turn dasar: SAMA, lihat llm/prompts.py
    (build_graphrag_messages).
  - LLM dipanggil via factory provider yang SAMA PERSIS (get_llm_client).

Yang BARU dibanding Kondisi B (hybrid retrieval + grounded generation):
  1. Korpus retrieval BUKAN flat chunk index terpisah, tapi Knowledge
     Graph berbobot trust di Neo4j (dibangun 7_build_knowledge_graph.py
     + didensifikasi 10_densify_tag_cooccurrence.py &
     11_densify_embedding_similarity.py).
  2. Hybrid retrieval TIGA TAHAP SEKUENSIAL (sesuai F4 proposal):
       a. ENTITY ANCHORING   -- vector search (FAISS, index yang sama
          dgn hasil 11_densify_embedding_similarity.py --stage edges)
          atas embedding Title+Body pertanyaan evaluasi -> top-N_ANCHOR
          Question node paling mirip sbg titik masuk ke graf.
       b. GRAPH TRAVERSAL    -- dari tiap anchor, jelajah edge berbobot
          trust (HAS_ACCEPTED_ANSWER/HAS_ANSWER 1-hop; IS_RELATED_TO/
          TAG_COOCCUR/EMBED_SIM -> Answer 2-hop), mengumpulkan Answer
          node beserta bobot jalur (path trust).
       c. SEMANTIC EXPANSION -- simpul Question hasil traversal dipakai
          SEBAGAI ANCHOR BARU utk vector search putaran kedua (menangkap
          konten yang relevan secara semantik tapi TIDAK terhubung
          eksplisit di graf).
     Ketiga tahap difusikan: dedup by answer_id, rerank by combined trust
     score, cap top-K -> unified context window (F4).
  3. Grounded generation DUAL-CONSTRAINT (F5): setiap item konteks diberi
     label sumber [SO-<question_id>] eksplisit; prompt mewajibkan LLM
     mengutip label tsb utk SETIAP klaim (lihat build_graphrag_messages
     di llm/prompts.py). Pasca-generation, jawaban di-scan (regex) utk
     menghitung has_citation & cited_source_ids -- inilah cara NF2 (100%
     jawaban dgn >=1 kutipan sumber) diukur.
  4. ABLATION STUDY -- fusion mode: --fusion-mode {trust_weighted,uniform}
     mengisolasi kontribusi trust-weighting itu sendiri. "uniform" memakai
     STRUKTUR retrieval yang PERSIS SAMA (anchoring/traversal/expansion/
     leakage exclusion tidak berubah sama sekali) tapi ranking akhir tidak
     memprioritaskan trust score -- lihat fuse_and_rank(). Bobot fusi
     (--fusion-w-path-trust, --fusion-w-intrinsic,
     --semantic-expansion-trust-cap) juga bisa diubah lewat CLI/.env utk
     ablasi bobot, bukan cuma on/off.
  5. ABLATION STUDY -- grounding constraint & semantic expansion stage kini
     opsional: --no-require-grounding melepas instruksi grounding (hanya
     instruksi citation yang tersisa, lihat build_graphrag_messages());
     --no-enable-semantic-expansion men-skip tahap semantic expansion
     sepenuhnya (retrieval hanya anchor+traversal). Struktur retrieval lain
     & leakage exclusion TIDAK berubah pada kedua toggle ini.

PENCEGAHAN LEAKAGE -- BEDA DARI KONDISI B, WAJIB DIBACA
------------------------------------------------------------
KG di Neo4j adalah SATU graf untuk SELURUH populasi Question -- 384
pertanyaan evaluasi JUGA ada di dalamnya sbg node biasa, lengkap dgn
edge HAS_ACCEPTED_ANSWER ke jawaban yang sedang diprediksi. Kondisi B
menghindari ini dgn membangun index terpisah yang exclude leakage SEKALI
di awal (build-time); Kondisi C TIDAK BISA begitu karena korpusnya adalah
graf bersama, jadi exclusion dilakukan SECARA DINAMIS DI SETIAP QUERY:
  1. Anchor vector search: EXCLUDE eval question's own id dari hasil top-N.
  2. Graph traversal (kedua Cypher query, 1-hop & 2-hop): EXCLUDE
     eval_question_id (sbg via_question) DAN accepted_answer_id + semua
     answer_id lain milik eval question tsb (WHERE NOT ... IN $exclude_*).
  3. Semantic expansion: exclusion yang sama diterapkan lagi (anchor baru
     bisa saja berupa eval question itu sendiri kalau tidak difilter).
Tanpa exclusion ini, cosine similarity Kondisi C akan bias tinggi secara
artifisial (retrieval "menemukan kembali" jawaban yang seharusnya
diprediksi) -- BUKAN pengukuran generalisasi yang valid.

INSTALL DEPENDENCY (tambahan dari Kondisi B)
-------------------
    pip install neo4j faiss-cpu

SETUP .env (WAJIB, tambahan dari Kondisi A/B)
----------------------------------------------------------------------------
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j
    KG_WORKSPACE_DIR=../../01_data_cleaning/_kg_workspace
    # ^ folder cache FAISS index & embedding hasil
    #   11_densify_embedding_similarity.py --stage edges (relatif thd
    #   folder c_graphrag/ tempat script ini dijalankan)

    N_ANCHOR=3
    N_SEMANTIC_EXPANSION=3
    MAX_HOPS=2   # informational -- traversal saat ini fixed 1-2 hop, lihat catatan di traverse_graph()

    # Ablation study (opsional -- default di bawah = perilaku asli, sebelum
    # fitur ablasi ada). Override sesaat via CLI: --fusion-mode dst.
    FUSION_MODE=trust_weighted   # atau "uniform"
    FUSION_W_PATH_TRUST=0.7
    FUSION_W_ANSWER_INTRINSIC_TRUST=0.3
    SEMANTIC_EXPANSION_TRUST_CAP=0.4

CARA PAKAI
----------
    python c_condition_c_graphrag.py
    python c_condition_c_graphrag.py --provider ollama --model qwen2.5:1.5b --n-sample 10

    # Ablation study -- isolasi kontribusi trust-weighting (struktur retrieval
    # SAMA PERSIS di kedua run, hanya fusi/ranking yang beda; SAMAKAN --seed
    # dan --n-sample di kedua run supaya sample evaluasi identik):
    python c_condition_c_graphrag.py --fusion-mode uniform --n-sample 30
    python c_condition_c_graphrag.py --fusion-mode trust_weighted \
        --fusion-w-path-trust 0.3 --fusion-w-intrinsic 0.7 --n-sample 30

Prasyarat WAJIB sebelum run: 11_densify_embedding_similarity.py --stage all
sudah pernah dijalankan sukses (index FAISS + embedding harus ada di
KG_WORKSPACE_DIR). Skrip ini TIDAK membangun ulang KG maupun edge
densifikasi -- murni membaca graf & index yang sudah ada.

Resumable, pola sama dgn Kondisi A/B.
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
from llm.manifest import read_already_done, sum_usage_tokens, write_manifest
from llm.prompts import PROMPT_VERSION, build_graphrag_messages

TOKEN_LIMIT = 2048  # kriteria eksklusi pertanyaan evaluasi, IDENTIK Kondisi A/B
EMBED_DIM = 384  # all-MiniLM-L6-v2, HARUS sama dgn 11_densify_embedding_similarity.py

# Bobot fusi skor akhir kandidat: 0.7 kontribusi dari trust struktural graf
# (edge_weight / proxy), 0.3 dari trustScore intrinsik node Answer (Tabel
# II.2 -- fungsi dari score/isAccepted/reputation). Konstanta ini KEPUTUSAN
# METODOLOGIS yang bisa didokumentasikan/disesuaikan di Bab III/IV.
#
# Dijadikan DEFAULT (bukan hardcoded) sejak fitur ablation study fusion-mode
# ditambahkan -- lihat --fusion-w-path-trust/--fusion-w-intrinsic/
# --semantic-expansion-trust-cap di main() dan parameter fuse_and_rank().
# Nilai default TIDAK berubah, supaya run tanpa override tetap identik
# dengan perilaku sebelum fitur ini ada.
DEFAULT_FUSION_W_PATH_TRUST = 0.7
DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST = 0.3

# C_RETRIEVAL_VERSION
# ------------------------------------------------------------
# "v1" = traverse_graph()/semantic_expansion() punya ORDER BY tanpa tie-break
#        (1-hop malah TIDAK ADA ORDER BY sama sekali), dan fuse_and_rank()'s
#        Python sort (mode trust_weighted) memakai combined_score SAJA sbg
#        key -- pada combined_score yang SAMA PERSIS (ties memang terjadi,
#        dikonfirmasi lewat perbandingan langsung thd file n=10 pilot: 7/10
#        pertanyaan berubah retrieved_context-nya setelah fix ini, beberapa
#        bukan cuma re-order tapi item BERBEDA yg lolos dedup by-answer_id),
#        hasil akhir bergantung pada urutan return Neo4j yang TIDAK dijamin
#        stabil antar eksekusi query yang identik.
# "v2" (C_RETRIEVAL_VERSION saat ini) = SEMUA query Cypher ber-ORDER BY+LIMIT
#        (2-hop traverse_graph) diberi tie-break sekunder (a.id ASC), 1-hop
#        traverse_graph()/semantic_expansion() diberi ORDER BY eksplisit
#        (a.id ASC, sebelumnya tidak ada sama sekali), dan fuse_and_rank()'s
#        sort trust_weighted memakai (-combined_score, answer_id) sbg key.
#        RANKING/TRUST LOGIC TIDAK DIUBAH -- ini murni membuat urutan akhir
#        deterministik/reproducible, bukan mengubah kriteria pemeringkatan.
#        Direkam sbg field `c_retrieval_version` di run_history.jsonl.
C_RETRIEVAL_VERSION = "v2"
# Proxy trust utk kandidat yang HANYA ditemukan lewat semantic expansion
# (tidak lewat edge graf eksplisit) -- disamakan dgn tier EMBED_SIM
# (trust "terendah" di antara 3 sumber densifikasi, lihat
# 11_densify_embedding_similarity.py), diskalakan oleh cosine score-nya.
DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP = 0.4

# C retrieval v3 (docs/DECISION_C_SCORING.md) -- defaults for every v3-
# only / exploratory parameter, used BOTH as the actual default behavior
# AND as the backward-compatibility threshold in build_config() below (a
# key is included in the hashed dict ONLY when it differs from its
# default, or -- for `alpha` -- when c_retrieval_version is "v3"). This
# guarantees build_config() for a v2 run with no exploratory overrides
# produces a dict BYTE-IDENTICAL to the one every existing run's
# config_hash was computed from, so config_hash for every pre-existing
# run never changes. See test_config_hash_backward_compatibility.py,
# which recomputes all 9 real pilot runs' config_hash and asserts an
# exact match. Defined here (near the top of the module, before
# traverse_graph()/fuse_and_rank()/build_config() all reference them) so
# every function below can use them as default argument values.
DEFAULT_SAMPLE_SPLIT = "test"
DEFAULT_MAX_HOPS = 2
DEFAULT_EDGE_TYPES = ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER", "IS_RELATED_TO", "TAG_COOCCUR", "EMBED_SIM")
DEFAULT_USE_AUTHOR_TRUST = False
DEFAULT_ACCEPTED_ONLY = False
AUTHOR_TRUST_BETA = 0.3  # see docs/DECISION_C_SCORING.md's use_author_trust section

log = print
LOG_DIR = Path("logs")


def setup_logging(provider: str, model: str) -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
    log_path = LOG_DIR / f"{ts}_{provider}_{safe_model}.log"

    logger = logging.getLogger("condition_c")
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
# (IDENTIK Kondisi A/B -- disalin apa adanya. Lihat b_condition_b_rag.py
#  utk komentar lengkap; tidak diulang di sini supaya file ini fokus pada
#  bagian yang BARU utk Kondisi C.)
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
    """NESTED SAMPLING -- IDENTIK Kondisi A/B (lihat komentar lengkap di
    a_baseline_replication.py). Permutasi SEKALI dgn seed tetap, ambil N
    pertama sbg PREFIX -- menjamin sample n=30 SUBSET PERSIS dari n=100/n=384,
    SELAMA oversample_pool & seed identik di semua run/kondisi.

    UNCHANGED since before the dev/test split existed -- byte-identical
    to sample_questions_split(df, n, seed, split="test"), kept as its own
    function (rather than a thin wrapper) so this function's behavior can
    never be affected by a bug in the new split logic. See
    test_sample_questions_split.py's byte-identical-to-test assertion."""
    log(f"[3/9] Mengambil {n} pertanyaan evaluasi PERTAMA dari permutasi tetap "
          f"(seed={seed}) -- prefix ini NESTED thd n_sample lain...")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df_permuted) < n:
        log(f"      [WARN] Populasi ({len(df_permuted)}) < target ({n}), pakai semua.")
        return df_permuted
    return df_permuted.head(n).reset_index(drop=True)


def sample_questions_split(df: pd.DataFrame, n: int, seed: int, split: str = "test",
                            dev_offset: int = 385) -> pd.DataFrame:
    """Same deterministic permutation as sample_questions() (SAME
    `df.sample(frac=1.0, random_state=seed)` call), but returns a
    position SLICE instead of always `head(n)`:
      - split="test" (default): positions 1..n (iloc[0:n]) -- BYTE-
        IDENTICAL to sample_questions(df, n, seed) today. Existing CLI/
        dashboard behavior is unaffected by this function existing.
      - split="dev": positions dev_offset..dev_offset+n-1 (1-indexed in
        docs/DECISION_C_SCORING.md's "positions 385-434" language ==
        iloc[384:434] here, since dev_offset is given 1-indexed and
        converted to a 0-indexed start below).
    Disjointness from the test split (positions 1..384) is a property of
    slicing non-overlapping ranges of the SAME permutation -- guaranteed
    as long as dev_offset > the largest test n_sample ever used (384),
    never re-derived from a different seed or a fresh shuffle.
    """
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if split == "test":
        start = 0
    elif split == "dev":
        start = dev_offset - 1  # 1-indexed position -> 0-indexed start
    else:
        raise ValueError(f"Unknown sample_split '{split}' -- must be 'test' or 'dev'")

    end = start + n
    if end > len(df_permuted):
        log(f"      [WARN] Populasi ({len(df_permuted)}) < requested slice end ({end}) "
            f"for split='{split}' -- returning fewer rows than requested.")
    return df_permuted.iloc[start:end].reset_index(drop=True)


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
    miliknya (bukan cuma accepted), sama ketatnya dgn Kondisi B."""
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
# Neo4j + FAISS setup
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
    """Reuse index & embedding yang di-cache oleh
    11_densify_embedding_similarity.py --stage edges. TIDAK membangun ulang
    di sini -- kalau cache tidak ada, ABORT dengan pesan jelas."""
    import faiss

    index_path = kg_workspace_dir / "question_embeddings_ivf.faiss"
    ids_path = kg_workspace_dir / "question_embeddings_ivf_ids.npy"
    emb_path = kg_workspace_dir / "question_embeddings.f32"
    ckpt_path = kg_workspace_dir / "embed_checkpoint.json"

    if not (index_path.exists() and ids_path.exists() and emb_path.exists() and ckpt_path.exists()):
        raise SystemExit(
            f"[ABORT] Cache FAISS/embedding tidak ditemukan di {kg_workspace_dir}. "
            f"Jalankan dulu 11_densify_embedding_similarity.py --stage all "
            f"(di folder 01_data_cleaning/) sebelum menjalankan Kondisi C."
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
# TAHAP a. ENTITY ANCHORING -- vector search atas embedding cache
# ---------------------------------------------------------------------

def anchor_via_vector_search(query_emb: np.ndarray, index, ids_arr: np.ndarray,
                              top_n: int, exclude_ids: set) -> list:
    """Cari top_n Question id paling mirip via FAISS, EXCLUDE eval
    question's own id (leakage prevention #1, lihat dokumentasi di atas)."""
    search_n = top_n + len(exclude_ids) + 5  # buffer supaya tetap dapat top_n setelah exclude
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
# TAHAP b. GRAPH TRAVERSAL -- weighted, 1-hop (jawaban milik anchor) &
#          2-hop (via IS_RELATED_TO/TAG_COOCCUR/EMBED_SIM -> jawaban)
# ---------------------------------------------------------------------

def traverse_graph(driver, database, anchor_ids: list, exclude_question_ids: set,
                    exclude_answer_ids: set, max_hops: int = DEFAULT_MAX_HOPS,
                    edge_types: tuple | None = None) -> list:
    """CATATAN: traversal saat ini FIXED di 1-hop (jawaban langsung anchor)
    + 2-hop (anchor -> Question terkait -> jawabannya). Tidak digeneralisasi
    ke N-hop sembarang -- kalau nanti butuh >2 hop, tulis query tambahan
    dgn pola yang sama (JANGAN pakai variable-length path [*1..N] tanpa
    exclusion per-hop, karena leakage exclusion HARUS diterapkan di
    SETIAP node yang dilewati, bukan cuma titik akhir).

    `max_hops`/`edge_types` (C retrieval v3 exploratory switches, Step
    4b): default (2, every type below) reproduces the query exactly as
    it always ran. `max_hops=1` skips the 2-hop query entirely.
    `edge_types` filters which relationship types each query's MATCH may
    traverse -- the two 1-hop fetch types (HAS_ACCEPTED_ANSWER,
    HAS_ANSWER) and the three 2-hop relation types (IS_RELATED_TO,
    TAG_COOCCUR, EMBED_SIM) are each intersected with `edge_types`
    independently, so e.g. edge_types=("HAS_ACCEPTED_ANSWER",) alone
    disables HAS_ANSWER's 1-hop fetch AND (having no 2-hop relation
    types left) the entire 2-hop query.
    """
    edge_types = set(edge_types) if edge_types is not None else set(DEFAULT_EDGE_TYPES)
    fetch_types = [t for t in ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER") if t in edge_types]
    relation_types = [t for t in ("IS_RELATED_TO", "TAG_COOCCUR", "EMBED_SIM") if t in edge_types]

    exclude_q = list(exclude_question_ids) or [-1]
    exclude_a = list(exclude_answer_ids) or [-1]

    candidates = []
    if not fetch_types:
        return candidates

    fetch_pattern = "|".join(fetch_types)
    with driver.session(database=database) as session:
        # 1-hop: jawaban langsung milik anchor
        result = session.run(
            f"""
            UNWIND $anchor_ids AS anchor_id
            MATCH (anchor:Question {{id: anchor_id}})-[r:{fetch_pattern}]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN anchor_id AS via_question_id, anchor.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, a.trustScore AS answer_trust_score,
                   a.isAccepted AS is_accepted, r.weight AS edge_weight, 1 AS hop,
                   type(r) AS rel_type
            ORDER BY a.id ASC
            """,
            anchor_ids=anchor_ids, exclude_a=exclude_a,
        )
        candidates.extend(dict(record) for record in result)

        # 2-hop: anchor -> Question terkait (IS_RELATED_TO/TAG_COOCCUR/EMBED_SIM)
        # -> jawaban Question terkait tsb. EXCLUDE eval question sbg Question
        # terkait (leakage prevention #2) DAN exclude jawabannya.
        if max_hops >= 2 and relation_types:
            relation_pattern = "|".join(relation_types)
            result = session.run(
                f"""
                UNWIND $anchor_ids AS anchor_id
                MATCH (anchor:Question {{id: anchor_id}})-[r2:{relation_pattern}]->(q2:Question)
                WHERE NOT q2.id IN $exclude_q
                MATCH (q2)-[r3:{fetch_pattern}]->(a:Answer)
                WHERE NOT a.id IN $exclude_a
                WITH q2, a, r2, r3, (r2.weight * r3.weight) AS edge_weight
                ORDER BY edge_weight DESC, a.id ASC
                LIMIT 50
                RETURN q2.id AS via_question_id, q2.title AS via_question_title,
                       a.id AS answer_id, a.body AS answer_body, a.trustScore AS answer_trust_score,
                       a.isAccepted AS is_accepted, edge_weight, 2 AS hop,
                       (type(r2) + '->' + type(r3)) AS rel_type
                """,
                anchor_ids=anchor_ids, exclude_q=exclude_q, exclude_a=exclude_a,
            )
            candidates.extend(dict(record) for record in result)

    return candidates


def fetch_author_trust(driver, database, answer_ids: list[int]) -> dict[int, float]:
    """Batched AUTHOR_TRUST edge weight lookup, keyed by answer_id.
    Answers with no AUTHOR_TRUST edge (deleted/anonymous author, or the
    edge genuinely absent) are simply missing from the returned dict --
    callers must default to 0.0 for a missing key (see
    docs/DECISION_C_SCORING.md's use_author_trust formula), never treat
    a missing key as an error."""
    if not answer_ids:
        return {}
    with driver.session(database=database) as session:
        result = session.run(
            """
            UNWIND $answer_ids AS aid
            MATCH (a:Answer {id: aid})-[r:AUTHOR_TRUST]->(:User)
            RETURN aid AS answer_id, r.weight AS weight
            """,
            answer_ids=list(answer_ids),
        )
        return {int(record["answer_id"]): float(record["weight"]) for record in result}


# ---------------------------------------------------------------------
# TAHAP c. SEMANTIC EXPANSION -- simpul hasil traversal jadi anchor baru
# ---------------------------------------------------------------------

def semantic_expansion(driver, database, traversal_candidates: list, index, ids_arr: np.ndarray,
                        sbert_model, n_expansion: int, already_seen_qids: set,
                        exclude_question_ids: set, exclude_answer_ids: set,
                        edge_types: tuple | None = None) -> list:
    """Ambil s.d. 3 via_question unik dari hasil traversal sbg query baru
    (F4: "simpul hasil Graph Traversal sebagai anchor pencarian vektor").
    Utk tiap query baru, cari n_expansion Question tetangga (vector search)
    yang BELUM pernah muncul, lalu ambil jawaban 1-hop-nya saja (TIDAK
    traversal lebih jauh lagi -- dibatasi supaya latency tetap terkendali,
    NF3 <=15 detik).

    `edge_types` (C retrieval v3 exploratory switch, Step 4b): filters
    the 1-hop fetch types the same way traverse_graph() does. Default
    (every type) reproduces the query exactly as it always ran. If
    neither HAS_ACCEPTED_ANSWER nor HAS_ANSWER survives the filter,
    expansion can fetch nothing and returns [] immediately."""
    edge_types = set(edge_types) if edge_types is not None else set(DEFAULT_EDGE_TYPES)
    fetch_types = [t for t in ("HAS_ACCEPTED_ANSWER", "HAS_ANSWER") if t in edge_types]
    if not traversal_candidates or not fetch_types:
        return []

    seed_questions = {}
    for c in traversal_candidates:
        qid = c["via_question_id"]
        if qid not in seed_questions:
            seed_questions[qid] = c["via_question_title"] or ""
        if len(seed_questions) >= 3:
            break

    expansion_qids = set()
    for qid, title in seed_questions.items():
        q_emb = embed_query(title, sbert_model)
        found = anchor_via_vector_search(
            q_emb, index, ids_arr, top_n=n_expansion,
            exclude_ids=already_seen_qids | exclude_question_ids | expansion_qids | {qid},
        )
        expansion_qids.update(fid for fid, _ in found)

    if not expansion_qids:
        return []

    exclude_a = list(exclude_answer_ids) or [-1]
    fetch_pattern = "|".join(fetch_types)
    candidates = []
    with driver.session(database=database) as session:
        result = session.run(
            f"""
            UNWIND $qids AS qid
            MATCH (q:Question {{id: qid}})-[r:{fetch_pattern}]->(a:Answer)
            WHERE NOT a.id IN $exclude_a
            RETURN qid AS via_question_id, q.title AS via_question_title,
                   a.id AS answer_id, a.body AS answer_body, a.trustScore AS answer_trust_score,
                   a.isAccepted AS is_accepted, r.weight AS edge_weight, 'semantic' AS hop,
                   type(r) AS rel_type
            ORDER BY a.id ASC
            """,
            qids=list(expansion_qids), exclude_a=exclude_a,
        )
        candidates.extend(dict(record) for record in result)

    return candidates


# ---------------------------------------------------------------------
# FUSI -- dedup, rerank by combined trust score, chunking, cap top-K
# ---------------------------------------------------------------------

def _truncate_doc_text(enc, doc_text: str, token_chunk_limit: int) -> str:
    token_ids = enc.encode(doc_text)
    if len(token_ids) > token_chunk_limit:
        return enc.decode(token_ids[:token_chunk_limit])
    return doc_text


def min_max_normalize(values: list[float]) -> list[float]:
    """Per docs/DECISION_C_SCORING.md's sim_norm definition: min-max over
    the given pool. If all values are equal (including a single-item
    pool), every normalized value is 1 -- NOT 0 or NaN, so a uniformly
    irrelevant-but-tied pool doesn't get final_score=0 by a division
    artifact."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo == 0:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_candidate_similarities(embed_model, query_emb: np.ndarray, texts: list[str],
                                    answer_ids: list[int], cache=None) -> list[float]:
    """Batched cosine similarity of `query_emb` (already L2-normalized,
    see docs/DECISION_C_SCORING.md's embedding-consistency section)
    against each of `texts`, encoded with the SAME all-MiniLM-L6-v2
    instance, explicitly L2-normalized the same way. `cache` (an
    embedding_cache.EmbeddingCache, optional) is checked per (answer_id,
    text) before encoding and populated for any miss -- a batched encode
    covers only the misses, never the whole pool when most of it is
    already cached (e.g. repeated α-sweep runs over the same dev set)."""
    query_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    vectors: list[np.ndarray | None] = [None] * len(texts)
    to_encode_idx = []
    to_encode_text = []
    if cache is not None:
        for i, (aid, text) in enumerate(zip(answer_ids, texts)):
            cached = cache.get(aid, text)
            if cached is not None:
                vectors[i] = cached
            else:
                to_encode_idx.append(i)
                to_encode_text.append(text)
    else:
        to_encode_idx = list(range(len(texts)))
        to_encode_text = texts

    if to_encode_text:
        encoded = embed_model.encode(to_encode_text, convert_to_numpy=True, device="cpu", batch_size=32)
        encoded = np.asarray(encoded, dtype=np.float32)
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        encoded = encoded / norms
        for j, i in enumerate(to_encode_idx):
            vectors[i] = encoded[j]
            if cache is not None:
                cache.put(answer_ids[i], texts[i], encoded[j])

    return [float(np.dot(query_vec, v)) for v in vectors]


def fuse_and_rank(graph_candidates: list, expansion_candidates: list, top_k: int,
                   token_chunk_limit: int, fusion_mode: str = "trust_weighted",
                   w_path_trust: float = DEFAULT_FUSION_W_PATH_TRUST,
                   w_intrinsic: float = DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
                   expansion_trust_cap: float = DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP,
                   c_retrieval_version: str = "v2", alpha: float | None = None,
                   embed_model=None, query_emb: np.ndarray | None = None,
                   embedding_cache=None, accepted_only: bool = DEFAULT_ACCEPTED_ONLY,
                   use_author_trust: bool = DEFAULT_USE_AUTHOR_TRUST) -> tuple[list, dict]:
    """Fusi graph_candidates + expansion_candidates -> top-K unified context.
    Returns (final_context_items, meta) -- `meta` is {} for
    c_retrieval_version="v2" (nothing new to report); for "v3" it carries
    {"candidate_pool_size": N} (Step 3.5's required per-question pool
    size), per docs/DECISION_C_SCORING.md.

    `fusion_mode` (c_retrieval_version="v2" ONLY -- ignored entirely for
    "v3", where `alpha` supersedes it):
      - "trust_weighted" (default): PERSIS perilaku asli -- combined_score
        dari w_path_trust*edge_weight + w_intrinsic*answer_trust_score
        (proxy expansion_trust_cap utk kandidat expansion-only), dedup by
        answer_id keep HIGHEST score, ranked by combined_score desc.
      - "uniform": ablasi utk mengisolasi kontribusi trust-weighting --
        combined_score TIDAK dipakai utk ranking. Urutan final murni dari
        urutan penemuan (graph_traversal dulu baru semantic_expansion, hop 1
        sebelum hop 2 dalam tiap grup, urutan asli traverse_graph()/
        semantic_expansion() dipertahankan di dalam grup itu). Dedup by
        answer_id keep kemunculan PERTAMA (bukan skor tertinggi -- tidak ada
        skor yang relevan dibandingkan pada mode ini). combined_score tetap
        dihitung & diisi di record (skema JSONL tetap konsisten dgn mode
        trust_weighted) tapi diabaikan untuk urutan/seleksi top-K.

    v3 (c_retrieval_version="v3"): final_score = (1-alpha)*sim_norm +
    alpha*trust, where trust is the SAME combined_score computed below
    (optionally author-trust-adjusted, see `use_author_trust`), and
    sim_norm is min_max_normalize() of each candidate's cosine
    similarity to `query_emb`, computed on the FINAL, post-truncation
    chunk_text (the text actually shown to the LLM) -- see
    docs/DECISION_C_SCORING.md's Definitions section for why. alpha=1 is
    mathematically identical to v2 trust_weighted's own ranking (same
    trust, sim term weighted to 0); alpha=0 ranks purely by sim_norm.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    all_candidates = []
    for c in graph_candidates:
        intrinsic = c.get("answer_trust_score") or 0.0
        score = (w_path_trust * min(c["edge_weight"] or 0.0, 1.0)
                 + w_intrinsic * intrinsic)
        c = dict(c)
        c["combined_score"] = score
        c["source_stage"] = "graph_traversal"
        all_candidates.append(c)

    for c in expansion_candidates:
        intrinsic = c.get("answer_trust_score") or 0.0
        proxy_weight = min(c["edge_weight"] or 0.0, 1.0) * expansion_trust_cap
        c = dict(c)
        c["edge_weight"] = proxy_weight
        c["combined_score"] = (w_path_trust * proxy_weight
                                + w_intrinsic * intrinsic)
        c["source_stage"] = "semantic_expansion"
        all_candidates.append(c)

    if c_retrieval_version != "v3":
        if fusion_mode == "uniform":
            # Ablasi: ranking TIDAK memprioritaskan trust. Dedup keep FIRST
            # occurrence (urutan penemuan asli, bukan skor tertinggi), lalu
            # potong ke top_k tanpa sort by combined_score.
            best_by_answer = {}
            for c in all_candidates:
                aid = c["answer_id"]
                if aid not in best_by_answer:
                    best_by_answer[aid] = c
            top = list(best_by_answer.values())[:top_k]
        else:
            # Dedup by answer_id, keep highest combined_score occurrence
            best_by_answer = {}
            for c in all_candidates:
                aid = c["answer_id"]
                if aid not in best_by_answer or c["combined_score"] > best_by_answer[aid]["combined_score"]:
                    best_by_answer[aid] = c

            # Secondary key answer_id ASC breaks ties deterministically (combined_score
            # can tie -- e.g. identical edge_weight/trustScore candidates) -- without it,
            # Python's stable sort falls back to `all_candidates` insertion order, which
            # itself depends on Neo4j's MATCH return order (not guaranteed stable run to
            # run without an explicit ORDER BY upstream).
            ranked = sorted(best_by_answer.values(), key=lambda c: (-c["combined_score"], c["answer_id"]))
            top = ranked[:top_k]

        final = []
        for c in top:
            doc_text = f"Q: {c['via_question_title']}\nA: {c['answer_body']}"
            doc_text = _truncate_doc_text(enc, doc_text, token_chunk_limit)
            final.append({
                "question_id": int(c["via_question_id"]),
                "answer_id": int(c["answer_id"]),
                "chunk_text": doc_text,
                "trust_weight": round(float(c["edge_weight"] or 0.0), 4),
                "combined_score": round(float(c["combined_score"]), 4),
                "hop": c["hop"],
                "rel_type": c.get("rel_type"),
                "source_stage": c["source_stage"],
                "is_accepted": bool(c.get("is_accepted")) if c.get("is_accepted") is not None else None,
            })
        return final, {}

    # --- v3 ---
    if accepted_only:
        all_candidates = [c for c in all_candidates if c.get("is_accepted")]

    # Dedup by answer_id, keep highest combined_score occurrence (same
    # policy as v2 trust_weighted's dedup -- the trust term feeding
    # final_score is this SAME combined_score).
    best_by_answer = {}
    for c in all_candidates:
        aid = c["answer_id"]
        if aid not in best_by_answer or c["combined_score"] > best_by_answer[aid]["combined_score"]:
            best_by_answer[aid] = c
    pool = sorted(best_by_answer.values(), key=lambda c: c["answer_id"])
    candidate_pool_size = len(pool)

    if not pool:
        return [], {"candidate_pool_size": 0}

    doc_texts = []
    for c in pool:
        doc_text = f"Q: {c['via_question_title']}\nA: {c['answer_body']}"
        doc_texts.append(_truncate_doc_text(enc, doc_text, token_chunk_limit))

    answer_ids = [int(c["answer_id"]) for c in pool]
    sims = compute_candidate_similarities(embed_model, query_emb, doc_texts, answer_ids, cache=embedding_cache)
    sim_norms = min_max_normalize(sims)

    alpha_val = 1.0 if alpha is None else alpha
    scored = []
    for c, doc_text, sim, sim_norm in zip(pool, doc_texts, sims, sim_norms):
        trust = c["combined_score"]
        if use_author_trust:
            author_weight = c.get("author_weight") or 0.0
            trust = (1 - AUTHOR_TRUST_BETA) * trust + AUTHOR_TRUST_BETA * author_weight
        final_score = (1 - alpha_val) * sim_norm + alpha_val * trust
        scored.append((c, doc_text, sim, sim_norm, trust, final_score))

    scored.sort(key=lambda t: (-t[5], t[0]["answer_id"]))
    top = scored[:top_k]

    final = []
    for rank, (c, doc_text, sim, sim_norm, trust, final_score) in enumerate(top, start=1):
        final.append({
            "question_id": int(c["via_question_id"]),
            "answer_id": int(c["answer_id"]),
            "chunk_text": doc_text,
            "trust_weight": round(float(c["edge_weight"] or 0.0), 4),
            "combined_score": round(float(c["combined_score"]), 4),
            "hop": c["hop"],
            "rel_type": c.get("rel_type"),
            "source_stage": c["source_stage"],
            "is_accepted": bool(c.get("is_accepted")) if c.get("is_accepted") is not None else None,
            "sim": round(float(sim), 4),
            "sim_norm": round(float(sim_norm), 4),
            "trust": round(float(trust), 4),
            "final_score": round(float(final_score), 4),
            "alpha": alpha_val,
            "rank": rank,
        })
    return final, {"candidate_pool_size": candidate_pool_size}


# ---------------------------------------------------------------------
# Citation check -- NF2 (100% jawaban dgn >=1 kutipan sumber)
# Diekstrak ke llm/citations.py supaya Kondisi B (mode --require-citation
# opsional) bisa pakai validasi PERSIS SAMA -- perbandingan NF2 B vs C jadi
# apples-to-apples, bukan dua implementasi yang bisa diam-diam berbeda.
# ---------------------------------------------------------------------

from llm.citations import CITATION_PATTERN, compute_citation_report, extract_citations  # noqa: E402,F401


# ---------------------------------------------------------------------
# Similarity scoring -- IDENTIK Kondisi A/B
# ---------------------------------------------------------------------

def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    from sentence_transformers import util
    # convert_to_numpy + device="cpu" dipilih sengaja (bukan default MPS):
    # untuk teks pendek & model sekecil all-MiniLM-L6-v2, overhead MPS tidak
    # sepadan, dan cache GPU PyTorch di Apple Silicon cenderung menumpuk
    # perlahan sepanjang loop panjang (384 pertanyaan) kalau pakai tensor+MPS.
    emb_a = embed_model.encode(str(text_a), convert_to_numpy=True, device="cpu")
    emb_b = embed_model.encode(str(text_b), convert_to_numpy=True, device="cpu")
    return float(util.cos_sim(emb_a, emb_b).item())


# ---------------------------------------------------------------------
# Proses per-pertanyaan (dipakai CLI main() DAN dashboard backend/engine_service)
# ---------------------------------------------------------------------

def process_sample(sample_df: pd.DataFrame, llm_client, call_llm_fn, embed_model, driver, database,
                    faiss_index, faiss_ids, faiss_embeddings, id_to_row: dict, all_answer_ids_map: dict,
                    top_k: int, n_anchor: int, n_semantic_expansion: int, token_chunk_limit: int,
                    model: str, output_path: Path, already_done: set | None = None,
                    on_progress=None, check_cancel=None, fusion_mode: str = "trust_weighted",
                    fusion_w_path_trust: float = DEFAULT_FUSION_W_PATH_TRUST,
                    fusion_w_intrinsic: float = DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
                    semantic_expansion_trust_cap: float = DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP,
                    require_grounding: bool = True, enable_semantic_expansion: bool = True,
                    log_full_candidates: bool = False,
                    config: dict | None = None, config_hash: str | None = None,
                    c_retrieval_version: str = "v2", alpha: float | None = None,
                    max_hops: int = DEFAULT_MAX_HOPS, edge_types: tuple | None = None,
                    use_author_trust: bool = DEFAULT_USE_AUTHOR_TRUST,
                    accepted_only: bool = DEFAULT_ACCEPTED_ONLY,
                    embedding_cache=None, retrieval_only: bool = False,
                    ) -> tuple[list[dict], dict]:
    """Proses satu-per-satu sample_df: hybrid retrieval (anchor -> traversal ->
    semantic expansion -> fusion) + prompt grounded + hitung cosine similarity
    + validasi sitasi, tulis ke output_path (append, resumable). Diekstrak
    dari main() dengan pola SAMA dengan process_sample() Kondisi A/B supaya
    CLI dan dashboard backend (PLAN_UI_UX.md §5.7) berbagi satu implementasi.

    Return (results, stats) -- stats berisi n_no_context/n_no_citation/
    n_no_valid_citation/interrupted, dipakai main() untuk ringkasan run.
    `on_progress(dict)` / `check_cancel()` -> lihat docstring Kondisi A;
    keduanya opsional supaya CLI (main()) tidak berubah perilaku.

    `fusion_mode`/`fusion_w_path_trust`/`fusion_w_intrinsic`/
    `semantic_expansion_trust_cap` -> diteruskan apa adanya ke
    fuse_and_rank() (lihat docstring-nya utk arti fusion_mode="uniform" vs
    "trust_weighted"); default-nya PERSIS nilai lama, jadi CLI tanpa
    override tetap berperilaku identik dengan sebelum fitur ablasi ini ada.

    `require_grounding` (default True) -> diteruskan ke build_graphrag_messages()
    (lihat docstring-nya); False = hanya instruksi citation, tanpa grounding.
    `enable_semantic_expansion` (default True) -> kalau False, TAHAP c (semantic
    expansion) di-skip sepenuhnya (expansion_candidates selalu []) -- ablasi utk
    mengisolasi kontribusi tahap semantic expansion itu sendiri terhadap hasil.
    Keduanya opsional & default-nya PERSIS perilaku lama.
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
            log(f"      [debug-init] cwd={os.getcwd()!r} output_path={output_path!r} "
                f"resolved={output_path.resolve()!r} fileno={f_out.fileno()}")
            for i, row in sample_df.iterrows():
                if check_cancel is not None and check_cancel():
                    log(f"      [cancelled] Dihentikan setelah {len(results)}/{total} pertanyaan.")
                    break

                qid = int(row["Id"])
                if qid in already_done:
                    continue

                t_query_start = time.time()

                # leakage exclusion set untuk pertanyaan evaluasi INI SAJA
                exclude_question_ids = {qid}
                exclude_answer_ids = set(all_answer_ids_map.get(qid, []))
                exclude_answer_ids.add(int(row["AcceptedAnswerId"]))

                # kalau embedding pertanyaan ini sendiri sudah ada di cache
                # (harusnya iya, KG dibangun full population), pakai LANGSUNG
                # dari memmap -- lebih cepat & konsisten drpd re-encode; fallback
                # re-encode kalau belum ada di cache (mis. KG dibuild sebelum
                # pertanyaan ini masuk parquet, jarang terjadi).
                if qid in id_to_row:
                    query_emb = faiss_embeddings[id_to_row[qid]:id_to_row[qid] + 1]
                    query_emb = np.ascontiguousarray(query_emb)
                else:
                    query_text = f"{row['Title']} {str(row['Body'])[:1000]}"
                    query_emb = embed_query(query_text, embed_model)

                # --- a. Entity Anchoring ---
                anchors = anchor_via_vector_search(
                    query_emb, faiss_index, faiss_ids, n_anchor, exclude_question_ids,
                )
                anchor_ids = [qid_ for qid_, _ in anchors]

                # --- b. Graph Traversal ---
                graph_candidates = []
                if anchor_ids:
                    graph_candidates = traverse_graph(
                        driver, database, anchor_ids, exclude_question_ids, exclude_answer_ids,
                        max_hops=max_hops, edge_types=edge_types,
                    )

                # --- c. Semantic Expansion (ablation: skippable via enable_semantic_expansion) ---
                if enable_semantic_expansion:
                    seen_qids = exclude_question_ids | set(anchor_ids)
                    expansion_candidates = semantic_expansion(
                        driver, database, graph_candidates, faiss_index, faiss_ids, embed_model,
                        n_semantic_expansion, seen_qids, exclude_question_ids, exclude_answer_ids,
                        edge_types=edge_types,
                    )
                else:
                    expansion_candidates = []

                # --- v3 exploratory: author trust (Step 4b) -- batched lookup over
                # every candidate answer_id found so far, merged in before fuse_and_rank
                # so it can build trust' = (1-β)*trust + β*author_weight per
                # docs/DECISION_C_SCORING.md. Skipped entirely (zero extra Neo4j
                # round-trip) unless the switch is on.
                if c_retrieval_version == "v3" and use_author_trust:
                    all_aids = [c["answer_id"] for c in graph_candidates + expansion_candidates]
                    author_weights = fetch_author_trust(driver, database, all_aids)
                    for c in graph_candidates + expansion_candidates:
                        c["author_weight"] = author_weights.get(c["answer_id"], 0.0)

                # --- Fusi + rerank + chunking + cap top-K ---
                retrieved, fuse_meta = fuse_and_rank(
                    graph_candidates, expansion_candidates,
                    top_k, token_chunk_limit, fusion_mode=fusion_mode,
                    w_path_trust=fusion_w_path_trust, w_intrinsic=fusion_w_intrinsic,
                    expansion_trust_cap=semantic_expansion_trust_cap,
                    c_retrieval_version=c_retrieval_version, alpha=alpha,
                    embed_model=embed_model, query_emb=query_emb, embedding_cache=embedding_cache,
                    accepted_only=accepted_only, use_author_trust=use_author_trust,
                )
                candidate_pool_size = fuse_meta.get("candidate_pool_size")
                retrieval_latency = time.time() - t_query_start
                if not retrieved:
                    n_no_context += 1

                if retrieval_only:
                    record = {
                        "question_id": qid, "title": row["Title"], "tags": row["Tags"],
                        "accepted_answer_id": int(row["AcceptedAnswerId"]),
                        "retrieved_context": retrieved,
                        "candidate_pool_size": candidate_pool_size,
                        "n_anchors": len(anchor_ids), "n_graph_candidates": len(graph_candidates),
                        "n_expansion_candidates": len(expansion_candidates),
                        "retrieval_latency_sec": round(retrieval_latency, 3),
                        "config": config, "config_hash": config_hash,
                    }
                    results.append(record)
                    f_out.write(json.dumps(record, default=str) + "\n")
                    f_out.flush()
                    if on_progress:
                        on_progress({"question_id": qid, "index": i, "total": total, "status": "done"})
                    continue

                all_candidate_question_ids = None
                if log_full_candidates:
                    # Opt-in: daftar LENGKAP via_question_id dari SELURUH kandidat
                    # sebelum fuse_and_rank()/top_k cutoff (graph traversal +
                    # semantic expansion digabung, deduped) -- TIDAK mempengaruhi
                    # `retrieved` yang dipakai prompt, murni metadata tambahan utk
                    # menghitung Recall@k sesungguhnya di run berikutnya (lihat
                    # analyze_retrieval_quality.py).
                    seen = []
                    for c in graph_candidates + expansion_candidates:
                        vqid = c.get("via_question_id")
                        if vqid is not None and vqid not in seen:
                            seen.append(int(vqid))
                    all_candidate_question_ids = seen

                messages = build_graphrag_messages(row["Title"], row["Body"], row["Tags"], retrieved,
                                                    require_grounding=require_grounding)
                try:
                    call_result = call_llm_fn(llm_client, messages, model)
                    llm_answer = call_result["content"]
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
                    "n_anchors": len(anchor_ids),
                    "n_graph_candidates": len(graph_candidates),
                    "n_expansion_candidates": len(expansion_candidates),
                    # Actual number of context items sent to the LLM in this
                    # prompt, AFTER fuse_and_rank()'s top_k cutoff -- always
                    # <= top_k regardless of how large the n_anchor/
                    # n_semantic_expansion candidate pool was. Equivalent to
                    # len(retrieved_context), stored explicitly so a pilot
                    # run can be verified without recomputing it.
                    "n_context_items_used": len(retrieved),
                    # Candidate pool size (post-dedup, pre-top_k) -- only
                    # populated for c_retrieval_version="v3" (None for v2,
                    # which never computes a unified scored pool the same
                    # way); see fuse_and_rank()'s meta return.
                    "candidate_pool_size": candidate_pool_size,
                    "require_grounding": require_grounding,
                    "enable_semantic_expansion": enable_semantic_expansion,
                    **({"all_candidate_question_ids": all_candidate_question_ids} if log_full_candidates else {}),
                    "retrieval_latency_sec": round(retrieval_latency, 3),
                    "llm_answer": llm_answer,
                    "llm_model": model,
                    "cosine_similarity": similarity,
                    "has_citation": has_citation,
                    "cited_source_ids": cited_ids,
                    "has_valid_citation": has_valid_citation,
                    "valid_cited_source_ids": valid_ids,
                    "response_id": call_result.get("response_id"),
                    "response_model": call_result.get("response_model"),
                    "response_created": call_result.get("response_created"),
                    "system_fingerprint": call_result.get("system_fingerprint"),
                    "finish_reason": call_result.get("finish_reason"),
                    "usage_full": call_result.get("usage_full"),
                    "request_params": call_result.get("request_params"),
                    "config": config,
                    "config_hash": config_hash,
                }
                try:
                    line = json.dumps(record, default=str)
                except Exception as e:
                    log(f"      [{i+1}/{total}] Id={qid} [FAIL-WRITE] {e}")
                    if on_progress:
                        on_progress({"question_id": qid, "index": i, "total": total,
                                     "status": "failed", "error": str(e)})
                    continue

                try:
                    n_bytes_written = f_out.write(line + "\n")
                    f_out.flush()
                    os.fsync(f_out.fileno())
                    size_after = os.path.getsize(output_path)
                    log(f"      [debug-write] Id={qid} bytes_written_this_line={n_bytes_written} "
                        f"file_size_now={size_after} f_out.tell()={f_out.tell()}")
                except Exception as e:
                    import traceback
                    log(f"      [debug-write] [EXCEPTION SAAT WRITE] Id={qid}: {e}")
                    log(traceback.format_exc())
                    raise
                results.append(record)
                log(f"      [{i+1}/{total}] Id={qid} anchors={len(anchor_ids)} "
                    f"context={len(retrieved)} citation={has_citation} valid_citation={has_valid_citation} "
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
        log("\n[INTERRUPTED] Proses dihentikan manual (Ctrl+C / kill). Ringkasan di bawah "
            "dihitung dari hasil yang SUDAH tersimpan di file sejauh ini. Jalankan command "
            "yang SAMA PERSIS lagi kapan saja -- otomatis lanjut (resume) dari sisa yang "
            "belum diproses, tidak perlu mengulang dari awal.")

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

def _fmt_weight_for_filename(w: float) -> str:
    """0.7 -> "0-7" (dot isn't valid in a filename component here since the
    rest of the naming scheme already uses "-" as a field separator)."""
    return f"{w}".replace(".", "-")


# Defaults for every v3-only / exploratory parameter -- used BOTH as the
def build_config(provider: str, model: str, n_sample: int, seed: int, oversample_pool: int | None,
                  top_k: int, n_anchor: int, n_semantic_expansion: int, fusion_mode: str,
                  fusion_w_path_trust: float, fusion_w_intrinsic: float,
                  semantic_expansion_trust_cap: float, require_grounding: bool,
                  enable_semantic_expansion: bool, c_retrieval_version: str = "v2",
                  alpha: float | None = None, sample_split: str = DEFAULT_SAMPLE_SPLIT,
                  max_hops: int = DEFAULT_MAX_HOPS, edge_types: tuple | None = None,
                  use_author_trust: bool = DEFAULT_USE_AUTHOR_TRUST,
                  accepted_only: bool = DEFAULT_ACCEPTED_ONLY) -> dict:
    """The full set of parameters that affect Condition C's answers -- see
    llm/a_pure_llm/a_baseline_replication.py::build_config()'s docstring
    for why this exists. Includes c_retrieval_version since a retrieval
    code fix changes answers just as much as a parameter does.

    BACKWARD COMPATIBILITY (required, see docs/DECISION_C_SCORING.md):
    `alpha`/`sample_split`/`max_hops`/`edge_types`/`use_author_trust`/
    `accepted_only` are each included in the returned dict ONLY when they
    differ from their default (alpha is also included whenever
    c_retrieval_version == "v3", since alpha is meaningless -- and was
    never part of any v2 run's hash -- otherwise). A v2 run with no
    exploratory overrides therefore produces EXACTLY the same dict
    compute_config_hash() always hashed for it, so config_hash for every
    existing run is unaffected by this feature existing at all.
    """
    edge_types = tuple(edge_types) if edge_types is not None else DEFAULT_EDGE_TYPES
    config = {
        "condition": "C", "provider": provider, "model": model, "n_sample": n_sample, "seed": seed,
        "oversample_pool": oversample_pool, "top_k": top_k, "n_anchor": n_anchor,
        "n_semantic_expansion": n_semantic_expansion, "fusion_mode": fusion_mode,
        "fusion_w_path_trust": fusion_w_path_trust, "fusion_w_intrinsic": fusion_w_intrinsic,
        "semantic_expansion_trust_cap": semantic_expansion_trust_cap,
        "require_grounding": require_grounding, "enable_semantic_expansion": enable_semantic_expansion,
        "c_retrieval_version": c_retrieval_version,
    }
    if c_retrieval_version == "v3" or alpha is not None:
        config["alpha"] = alpha
    if sample_split != DEFAULT_SAMPLE_SPLIT:
        config["sample_split"] = sample_split
    if max_hops != DEFAULT_MAX_HOPS:
        config["max_hops"] = max_hops
    if tuple(edge_types) != DEFAULT_EDGE_TYPES:
        config["edge_types"] = sorted(edge_types)
    if use_author_trust != DEFAULT_USE_AUTHOR_TRUST:
        config["use_author_trust"] = use_author_trust
    if accepted_only != DEFAULT_ACCEPTED_ONLY:
        config["accepted_only"] = accepted_only
    return config


def build_output_path(output_dir: str, provider: str, model: str, n_sample: int, seed: int,
                       fusion_mode: str = "trust_weighted",
                       fusion_w_path_trust: float = DEFAULT_FUSION_W_PATH_TRUST,
                       fusion_w_intrinsic: float = DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
                       require_grounding: bool = True, enable_semantic_expansion: bool = True,
                       oversample_pool: int | None = None, top_k: int = 5, n_anchor: int = 3,
                       n_semantic_expansion: int = 3,
                       semantic_expansion_trust_cap: float = DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP,
                       c_retrieval_version: str = "v2", alpha: float | None = None,
                       sample_split: str = DEFAULT_SAMPLE_SPLIT, max_hops: int = DEFAULT_MAX_HOPS,
                       edge_types: tuple | None = None, use_author_trust: bool = DEFAULT_USE_AUTHOR_TRUST,
                       accepted_only: bool = DEFAULT_ACCEPTED_ONLY) -> Path:
    """Nama file menyertakan fusion mode/bobot supaya run ablasi (mis.
    --fusion-mode uniform vs --fusion-mode trust_weighted dgn bobot
    berbeda) tidak saling menimpa file .jsonl satu sama lain -- pola sama
    dengan alasan n_sample/seed sudah ada di nama file sejak awal.

    `require_grounding=False`/`enable_semantic_expansion=False` masing-masing
    HANYA menambah suffix kalau non-default (False) -- run default (kedua True,
    perilaku asli sebelum toggle ini ada) tetap menghasilkan nama file yang
    SAMA seperti sebelumnya (minus the new prompt_version+config_hash
    suffix, see below -- that part is NOT back-compat on purpose, it's the
    2026-10-05 pilot incident fix: a stale file from an older
    prompt_version/config must never again share a filename with a new
    run's, see llm.manifest.read_already_done's docstring)."""
    from llm.manifest import compute_config_hash
    safe_model = model.replace("/", "-").replace(":", "-").replace(".", "-")
    base = f"condition_c_{provider}_{safe_model}_n{n_sample}_seed{seed}"
    if fusion_mode == "uniform":
        suffix = "uniform"
    else:
        suffix = f"fw{_fmt_weight_for_filename(fusion_w_path_trust)}-{_fmt_weight_for_filename(fusion_w_intrinsic)}"
    if not require_grounding:
        suffix += "_ungrounded"
    if not enable_semantic_expansion:
        suffix += "_noexp"
    if c_retrieval_version == "v3":
        suffix += f"_v3-alpha{_fmt_weight_for_filename(alpha if alpha is not None else 1.0)}"
    if sample_split != DEFAULT_SAMPLE_SPLIT:
        suffix += f"_{sample_split}"
    config_hash = compute_config_hash(build_config(
        provider, model, n_sample, seed, oversample_pool, top_k, n_anchor, n_semantic_expansion,
        fusion_mode, fusion_w_path_trust, fusion_w_intrinsic, semantic_expansion_trust_cap,
        require_grounding, enable_semantic_expansion, c_retrieval_version=c_retrieval_version,
        alpha=alpha, sample_split=sample_split, max_hops=max_hops, edge_types=edge_types,
        use_author_trust=use_author_trust, accepted_only=accepted_only,
    ))
    return Path(output_dir) / f"{base}_{suffix}_{PROMPT_VERSION}_{config_hash}.jsonl"


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
                         help="Jumlah item konteks final (setelah fusi 3 tahap retrieval)")
    parser.add_argument("--token-chunk-limit", type=int, default=int(os.getenv("TOKEN_CHUNK_LIMIT", 400)))
    parser.add_argument("--n-anchor", type=int, default=int(os.getenv("N_ANCHOR", 3)),
                         help="Jumlah Question anchor dari vector search awal")
    parser.add_argument("--n-semantic-expansion", type=int, default=int(os.getenv("N_SEMANTIC_EXPANSION", 3)),
                         help="Jumlah tetangga baru per seed-question di tahap semantic expansion")
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--kg-workspace-dir", default=os.getenv("KG_WORKSPACE_DIR",
                         "../../01_data_cleaning/_kg_workspace"))
    parser.add_argument("--fusion-mode", choices=["trust_weighted", "uniform"],
                         default=os.getenv("FUSION_MODE", "trust_weighted"),
                         help="Ablasi: 'trust_weighted' (default, perilaku asli) rerank kandidat "
                              "by combined trust score; 'uniform' MENGABAIKAN combined_score untuk "
                              "ranking (urutan final murni urutan penemuan) -- struktur retrieval "
                              "(anchoring/traversal/expansion/leakage exclusion) TETAP SAMA, hanya "
                              "fusi/ranking yang berubah, supaya isolasi kontribusi trust-weighting "
                              "tetap apples-to-apples terhadap mode trust_weighted.")
    parser.add_argument("--fusion-w-path-trust", type=float,
                         default=float(os.getenv("FUSION_W_PATH_TRUST", DEFAULT_FUSION_W_PATH_TRUST)),
                         help=f"Bobot trust struktural graf dalam combined_score (default "
                              f"{DEFAULT_FUSION_W_PATH_TRUST}, dari FUSION_W_PATH_TRUST di .env). "
                              f"Tidak dipakai kalau --fusion-mode uniform.")
    parser.add_argument("--fusion-w-intrinsic", type=float,
                         default=float(os.getenv("FUSION_W_ANSWER_INTRINSIC_TRUST",
                                                  DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST)),
                         help=f"Bobot trustScore intrinsik Answer dalam combined_score (default "
                              f"{DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST}, dari "
                              f"FUSION_W_ANSWER_INTRINSIC_TRUST di .env). Tidak dipakai kalau "
                              f"--fusion-mode uniform.")
    parser.add_argument("--semantic-expansion-trust-cap", type=float,
                         default=float(os.getenv("SEMANTIC_EXPANSION_TRUST_CAP",
                                                  DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP)),
                         help=f"Cap proxy trust utk kandidat semantic-expansion-only (default "
                              f"{DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP}, dari "
                              f"SEMANTIC_EXPANSION_TRUST_CAP di .env). Tidak dipakai kalau "
                              f"--fusion-mode uniform.")
    parser.add_argument("--require-grounding", action=argparse.BooleanOptionalAction,
                         default=os.getenv("GRAPHRAG_REQUIRE_GROUNDING", "true").strip().lower() == "true",
                         help="True (default): dual-constraint grounding+citation (perilaku asli). "
                              "False: hanya instruksi citation, TANPA instruksi grounding -- ablasi "
                              "utk mengisolasi kontribusi constraint grounding itu sendiri.")
    parser.add_argument("--enable-semantic-expansion", action=argparse.BooleanOptionalAction,
                         default=os.getenv("SEMANTIC_EXPANSION_ENABLED", "true").strip().lower() == "true",
                         help="True (default): tahap semantic expansion (c) aktif seperti biasa. "
                              "False: tahap ini di-skip sepenuhnya (retrieval hanya anchor+traversal) "
                              "-- ablasi utk mengisolasi kontribusi tahap semantic expansion.")
    parser.add_argument("--log-full-candidates", action="store_true", default=False,
                         help="Opt-in: simpan field tambahan all_candidate_question_ids (daftar LENGKAP "
                              "via_question_id dari SELURUH kandidat graph traversal + semantic expansion "
                              "SEBELUM top-k cutoff) di setiap record -- TIDAK mempengaruhi retrieved_context "
                              "yang dipakai prompt. Dipakai supaya run BERIKUTNYA bisa dihitung Recall@k "
                              "sesungguhnya di analyze_retrieval_quality.py (versi saat ini belum "
                              "menghitungnya). Default: nonaktif.")
    # --- C retrieval v3 (docs/DECISION_C_SCORING.md) -- all default to
    # v2's exact current behavior; see build_config()'s backward-
    # compatibility docstring for why defaults never change config_hash.
    parser.add_argument("--c-retrieval-version", choices=["v2", "v3"], default="v2",
                         help="'v2' (default): unchanged trust-only ranking. 'v3': adds the "
                              "(1-alpha)*sim_norm + alpha*trust scoring -- see --alpha.")
    parser.add_argument("--alpha", type=float, default=None,
                         help="v3 only: final_score = (1-alpha)*sim_norm + alpha*trust, alpha in [0,1]. "
                              "alpha=1 matches v2 trust_weighted's own ranking; alpha=0 ranks purely by "
                              "similarity. Required for --c-retrieval-version v3 (defaults to 1 if omitted).")
    parser.add_argument("--sample-split", choices=["test", "dev"], default="test",
                         help="'test' (default): positions 1..n, unchanged. 'dev': positions "
                              "385..385+n-1 (see --dev-offset) -- a separate, disjoint development "
                              "set for tuning alpha/judge-v2/prompt wording, never used for official results.")
    parser.add_argument("--dev-offset", type=int, default=385,
                         help="1-indexed start position for --sample-split dev (default 385, i.e. "
                              "positions 385-434 for n=50) -- see docs/DECISION_C_SCORING.md.")
    parser.add_argument("--max-hops", type=int, choices=[1, 2], default=DEFAULT_MAX_HOPS,
                         help="v3 exploratory (Step 4b): limit graph traversal depth. Default 2 "
                              "(unchanged). 1 skips the 2-hop (IS_RELATED_TO/TAG_COOCCUR/EMBED_SIM) query.")
    parser.add_argument("--edge-types", nargs="+", default=None,
                         help="v3 exploratory: subset of " + ", ".join(DEFAULT_EDGE_TYPES) +
                              " traversal may follow. Default: all of them (unchanged).")
    parser.add_argument("--use-author-trust", action="store_true", default=DEFAULT_USE_AUTHOR_TRUST,
                         help="v3 exploratory: include AUTHOR_TRUST edge weight in the trust term "
                              "(trust' = 0.7*trust + 0.3*author_weight). Default off (unchanged).")
    parser.add_argument("--accepted-only", action="store_true", default=DEFAULT_ACCEPTED_ONLY,
                         help="v3 exploratory: keep only accepted answers in the candidate pool. "
                              "Default off (unchanged).")
    parser.add_argument("--retrieval-only", action="store_true", default=False,
                         help="Run retrieval only (no LLM call) -- writes a contexts-only JSONL + "
                              "manifest. For the alpha sweep's stage 1 (see docs/"
                              "agent_prompt_c_retrieval_v3_devset.md Step 4.1).")
    parser.add_argument("--embedding-cache-path", default=os.getenv(
        "C_EMBEDDING_CACHE_PATH", str(Path(__file__).resolve().parent / "embedding_cache.sqlite3")),
        help="v3 only: SQLite cache path for candidate-text embeddings (shared across an alpha sweep).")
    args = parser.parse_args()

    global log
    logger, log_path = setup_logging(args.provider, args.model)
    log = logger.info
    run_started_at = datetime.now(timezone.utc)
    log(f"[logging] Log detail run ini -> {log_path}")

    if args.c_retrieval_version == "v3" and args.alpha is None:
        args.alpha = 1.0
        log("[config] --c-retrieval-version v3 without --alpha -- defaulting to alpha=1.0 "
            "(matches v2 trust_weighted's own ranking).")

    # PENTING utk NESTED SAMPLING -- IDENTIK Kondisi A/B, lihat komentar
    # lengkap di a_baseline_replication.py. Pool FIXED thd MAX_PLANNED_N_SAMPLE,
    # BUKAN thd args.n_sample lagi -- WAJIB SAMA PERSIS dgn nilai di A & B
    # supaya ketiga kondisi menarik populasi kandidat & sample yang identik
    # (apple-to-apple comparison, lihat catatan metodologi sampling consistency).
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
                                             args.n_sample, args.seed, args.fusion_mode,
                                             args.fusion_w_path_trust, args.fusion_w_intrinsic,
                                             args.require_grounding, args.enable_semantic_expansion,
                                             oversample_pool=oversample_pool, top_k=args.top_k,
                                             n_anchor=args.n_anchor, n_semantic_expansion=args.n_semantic_expansion,
                                             semantic_expansion_trust_cap=args.semantic_expansion_trust_cap,
                                             c_retrieval_version=args.c_retrieval_version, alpha=args.alpha,
                                             sample_split=args.sample_split, max_hops=args.max_hops,
                                             edge_types=args.edge_types, use_author_trust=args.use_author_trust,
                                             accepted_only=args.accepted_only))
        log(f"[config] output auto-generated -> '{args.output}'")

    if not args.questions_parquet or not args.answers_parquet:
        log("[ERROR] QUESTIONS_PARQUET dan ANSWERS_PARQUET harus diset di .env")
        sys.exit(1)

    log(f"[config] n_sample={args.n_sample} seed={args.seed} provider={args.provider} model={args.model}")
    log(f"[config] top_k={args.top_k} n_anchor={args.n_anchor} "
        f"n_semantic_expansion={args.n_semantic_expansion} token_chunk_limit={args.token_chunk_limit}")
    log(f"[config] fusion_mode={args.fusion_mode} fusion_w_path_trust={args.fusion_w_path_trust} "
        f"fusion_w_intrinsic={args.fusion_w_intrinsic} "
        f"semantic_expansion_trust_cap={args.semantic_expansion_trust_cap}")
    log(f"[config] require_grounding={args.require_grounding} "
        f"enable_semantic_expansion={args.enable_semantic_expansion}")

    llm_client, call_llm_fn = get_llm_client(args.provider, args.model, log=log, with_meta=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    # --- 1-4: sample pertanyaan evaluasi + ground truth (identik A/B) ---
    candidates = get_candidate_questions(con, args.questions_parquet, args.answers_parquet,
                                          oversample_pool, args.seed)
    candidates = filter_by_token_limit(candidates)
    sample_df = sample_questions_split(candidates, args.n_sample, args.seed,
                                        split=args.sample_split, dev_offset=args.dev_offset)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = get_accepted_answers(con, args.answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

    eval_question_ids = sample_df["Id"].astype(int).tolist()
    all_answer_ids_map = get_all_answer_ids_for_questions(con, args.answers_parquet, eval_question_ids)

    # --- 5: koneksi Neo4j + FAISS cache ---
    driver, database = connect_neo4j(log)
    kg_workspace_dir = Path(args.kg_workspace_dir)
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = load_faiss_cache(kg_workspace_dir, log)

    log(f"[6/9] Load model embedding ({args.embed_model})...")
    from sentence_transformers import SentenceTransformer
    # device="cpu" sengaja dipaksa (bukan auto-detect MPS): model sekecil
    # all-MiniLM-L6-v2 dengan teks pendek tidak butuh GPU, dan ini menghindari
    # potensi memory creep MPS di loop panjang (384 pertanyaan). Perubahan ini
    # berlaku untuk semua pemanggilan embed_model.encode() di script ini
    # (embed_query, compute_similarity).
    embed_model = SentenceTransformer(args.embed_model, device="cpu")

    embedding_cache = None
    if args.c_retrieval_version == "v3":
        from embedding_cache import EmbeddingCache
        embedding_cache = EmbeddingCache(args.embedding_cache_path)
        log(f"[config] v3 embedding cache -> {args.embedding_cache_path}")

    # --- Resume support ---
    from llm.manifest import compute_config_hash
    config = build_config(
        args.provider, args.model, args.n_sample, args.seed, oversample_pool, args.top_k,
        args.n_anchor, args.n_semantic_expansion, args.fusion_mode, args.fusion_w_path_trust,
        args.fusion_w_intrinsic, args.semantic_expansion_trust_cap, args.require_grounding,
        args.enable_semantic_expansion, c_retrieval_version=args.c_retrieval_version, alpha=args.alpha,
        sample_split=args.sample_split, max_hops=args.max_hops, edge_types=args.edge_types,
        use_author_trust=args.use_author_trust, accepted_only=args.accepted_only,
    )
    config_hash = compute_config_hash(config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = read_already_done(output_path, PROMPT_VERSION, config_hash)
    if already_done:
        log(f"      [resume] {len(already_done)} pertanyaan sudah diproses sebelumnya")

    log(f"[7/9] Hybrid retrieval (anchor+traversal+expansion) + prompting {args.model} "
        f"untuk {len(sample_df)} pertanyaan...")

    results, stats = process_sample(
        sample_df, llm_client, call_llm_fn, embed_model, driver, database,
        faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
        args.top_k, args.n_anchor, args.n_semantic_expansion, args.token_chunk_limit,
        args.model, output_path, already_done,
        fusion_mode=args.fusion_mode, fusion_w_path_trust=args.fusion_w_path_trust,
        fusion_w_intrinsic=args.fusion_w_intrinsic,
        semantic_expansion_trust_cap=args.semantic_expansion_trust_cap,
        require_grounding=args.require_grounding,
        enable_semantic_expansion=args.enable_semantic_expansion,
        log_full_candidates=args.log_full_candidates,
        config=config, config_hash=config_hash,
        c_retrieval_version=args.c_retrieval_version, alpha=args.alpha,
        max_hops=args.max_hops, edge_types=args.edge_types,
        use_author_trust=args.use_author_trust, accepted_only=args.accepted_only,
        embedding_cache=embedding_cache, retrieval_only=args.retrieval_only,
    )
    interrupted = stats["interrupted"]
    if embedding_cache is not None:
        embedding_cache.close()

    driver.close()
    if interrupted:
        log("[8/9] Diinterupsi sebelum semua sample selesai -- lihat catatan di atas.")
    else:
        log("[8/9] Selesai memproses seluruh sample.")

    file_exists = output_path.exists()
    file_size = output_path.stat().st_size if file_exists else -1
    log(f"[debug] exists={file_exists} path={output_path.resolve()} size_bytes={file_size}")
    if not file_exists or file_size == 0:
        log("\n[ERROR] Tidak ada hasil tersimpan sama sekali.")
        history_path = append_run_history({
            "prompt_version": PROMPT_VERSION,
            "c_retrieval_version": args.c_retrieval_version,
            "alpha": args.alpha, "sample_split": args.sample_split,
            "max_hops": args.max_hops, "edge_types": list(args.edge_types) if args.edge_types else None,
            "use_author_trust": args.use_author_trust, "accepted_only": args.accepted_only,
            "run_started_at": run_started_at.isoformat(), "condition": "C", "status": "no_results",
            "provider": args.provider, "model": args.model, "n_sample_target": args.n_sample,
            "seed": args.seed, "output_path": str(output_path), "log_path": str(log_path),
            "oversample_pool": oversample_pool,
            "log_full_candidates": args.log_full_candidates,
            "fusion_mode": args.fusion_mode,
            "fusion_w_path_trust": args.fusion_w_path_trust,
            "fusion_w_answer_intrinsic_trust": args.fusion_w_intrinsic,
            "semantic_expansion_trust_cap": args.semantic_expansion_trust_cap,
            "require_grounding": args.require_grounding,
            "grounding": "on" if args.require_grounding else "off",
            "enable_semantic_expansion": args.enable_semantic_expansion,
            "config_hash": config_hash,
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        from llm.evaluation._run_metadata import derive_run_label as _derive_run_label_failed
        run_label = _derive_run_label_failed("C", {
            "fusion_mode": args.fusion_mode, "grounding": "on" if args.require_grounding else "off",
        })["run_label"]
        finished_at = datetime.now(timezone.utc)
        manifest_path = write_manifest(
            output_path, run_label=run_label, status="failed",
            config=config, config_hash=config_hash, log_path=str(log_path),
            started_at_utc=run_started_at.isoformat(), finished_at_utc=finished_at.isoformat(),
            item_counts={"attempted": args.n_sample, "succeeded": 0, "failed": args.n_sample},
            prompt_version=PROMPT_VERSION,
        )
        log(f"[logging] Manifest (gagal) ditulis -> {manifest_path}")
        return

    results_df = pd.read_json(output_path, lines=True)
    log("\n" + "=" * 70)
    log("[9/9] RINGKASAN HASIL KONDISI C (GraphRAG Framework)")
    log("=" * 70)
    log(f"Total pertanyaan diproses      : {len(results_df)}")
    log(f"Cosine similarity rata-rata    : {results_df['cosine_similarity'].mean():.4f}")
    log(f"Cosine similarity median       : {results_df['cosine_similarity'].median():.4f}")
    log(f"% similarity > 0.5             : {(results_df['cosine_similarity'] > 0.5).mean()*100:.1f}%")
    pct_citation = results_df["has_citation"].mean() * 100
    pct_valid_citation = results_df["has_valid_citation"].mean() * 100
    log(f"% jawaban dgn >=1 kutipan format cocok  : {pct_citation:.1f}% (longgar, toleransi variasi format)")
    log(f"% jawaban dgn >=1 kutipan VALID (NF2)   : {pct_valid_citation:.1f}% "
        f"[target: 100%] {'PASS' if pct_valid_citation >= 100 else 'BELUM TERCAPAI'} "
        f"(ID kutipan divalidasi ada di retrieved_context yg sebenarnya)")

    # Reporting TAMBAHAN di luar definisi NF2 (pct_valid_citation di atas
    # TIDAK berubah) -- lihat docs/NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md:
    # 3-way split per jawaban, fabricated-citation-rate (jawaban dgn >=1
    # token TIDAK valid, meski ada yg valid juga), dan citation precision
    # (token valid / semua token kutipan, termasuk placeholder, se-run).
    citation_report = compute_citation_report(results_df.to_dict("records"))
    log(f"  -- 3-way split: valid={citation_report['pct_valid']}% / "
        f"no_citation={citation_report['pct_no_citation']}% / "
        f"invalid_only={citation_report['pct_invalid_only']}%")
    log(f"  -- fabricated citation rate (>=1 invalid token, even if also has a valid one): "
        f"{citation_report['fabricated_citation_rate']}%")
    log(f"  -- citation precision (valid tokens / all citation tokens incl. placeholders): "
        f"{citation_report['citation_precision']}")
    pct_no_context = (results_df["n_anchors"] == 0).mean() * 100
    log(f"% pertanyaan tanpa hasil retrieval sama sekali: {pct_no_context:.1f}%")
    avg_latency = results_df["retrieval_latency_sec"].mean()
    p95_latency = results_df["retrieval_latency_sec"].quantile(0.95)
    log(f"Retrieval latency -- avg: {avg_latency:.2f}s, p95: {p95_latency:.2f}s "
        f"[NF3 target: <=15s] {'PASS' if avg_latency <= 15.0 else 'FAIL'}")
    log(f"\nHasil lengkap tersimpan -> {output_path}")

    history_path = append_run_history({
        "prompt_version": PROMPT_VERSION,
        "c_retrieval_version": args.c_retrieval_version,
        "alpha": args.alpha, "sample_split": args.sample_split,
        "max_hops": args.max_hops, "edge_types": list(args.edge_types) if args.edge_types else None,
        "use_author_trust": args.use_author_trust, "accepted_only": args.accepted_only,
        "run_started_at": run_started_at.isoformat(), "condition": "C",
        "status": "interrupted" if interrupted else "success",
        "provider": args.provider, "model": args.model, "n_sample_target": args.n_sample,
        "n_processed": len(results_df), "seed": args.seed,
        "oversample_pool": oversample_pool,
        "n_candidates_after_token_filter": len(candidates),
        "log_full_candidates": args.log_full_candidates,
        "top_k": args.top_k, "n_anchor": args.n_anchor,
        "n_semantic_expansion": args.n_semantic_expansion,
        "fusion_mode": args.fusion_mode,
        "fusion_w_path_trust": args.fusion_w_path_trust,
        "fusion_w_answer_intrinsic_trust": args.fusion_w_intrinsic,
        "semantic_expansion_trust_cap": args.semantic_expansion_trust_cap,
        "require_grounding": args.require_grounding,
        "grounding": "on" if args.require_grounding else "off",
        "enable_semantic_expansion": args.enable_semantic_expansion,
        "config_hash": config_hash,
        "cosine_similarity_mean": round(float(results_df["cosine_similarity"].mean()), 4),
        "cosine_similarity_median": round(float(results_df["cosine_similarity"].median()), 4),
        "pct_similarity_above_0_5": round(float((results_df["cosine_similarity"] > 0.5).mean() * 100), 1),
        "pct_with_citation": round(float(pct_citation), 1),
        "pct_with_valid_citation": round(float(pct_valid_citation), 1),
        "pct_citation_valid": citation_report["pct_valid"],
        "pct_citation_no_citation": citation_report["pct_no_citation"],
        "pct_citation_invalid_only": citation_report["pct_invalid_only"],
        "fabricated_citation_rate": citation_report["fabricated_citation_rate"],
        "citation_precision": citation_report["citation_precision"],
        "pct_no_retrieval": round(float(pct_no_context), 1),
        "avg_retrieval_latency_sec": round(float(avg_latency), 3),
        "p95_retrieval_latency_sec": round(float(p95_latency), 3),
        "output_path": str(output_path), "log_path": str(log_path),
        "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
    })
    log(f"[logging] Ringkasan run dicatat -> {history_path}")

    from llm.evaluation._run_metadata import derive_run_label
    finished_at = datetime.now(timezone.utc)
    run_label = derive_run_label("C", {
        "fusion_mode": args.fusion_mode, "grounding": "on" if args.require_grounding else "off",
    })["run_label"]
    manifest_path = write_manifest(
        output_path,
        run_label=run_label,
        status="interrupted" if interrupted else "completed",
        config=config,
        config_hash=config_hash,
        log_path=str(log_path),
        started_at_utc=run_started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        item_counts={"attempted": args.n_sample, "succeeded": len(results_df),
                     "failed": args.n_sample - len(results_df)},
        prompt_version=PROMPT_VERSION,
        total_tokens=sum_usage_tokens(results_df.to_dict("records")),
    )
    log(f"[logging] Manifest ditulis -> {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())