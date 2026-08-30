"""
7_build_knowledge_graph.py
============================
Konstruksi Knowledge Graph berbobot (Kondisi C) ke Neo4j, dari data merged
SORD (questions/answers) + question_links.parquet (PostLinks enrichment).
Implementasi skema MVP/Phase 1 sesuai `neo4j_schema_design_kondisi_c.md`:

    Node : Question, Answer, Tag, User
    Edge : HAS_ACCEPTED_ANSWER (w=1.0)
           HAS_ANSWER           (w=normalize(score) per-thread)
           TAGGED_WITH          (w=1.0)
           IS_RELATED_TO        (Duplicate->w=1.0, Linked->w=cosine similarity embedding)
           AUTHOR_TRUST         (w=normalize(log(reputation)), clip P99)

    Concept / MENTIONS (butuh NER berbasis LLM) SENGAJA TIDAK dibangun di
    sini -- Phase 2, ditunda sesuai kesepakatan.

SCOPE DATA -- PENTING
-----------------------
questions_raw_union.parquet SORD bisa berisi ratusan ribu baris. Membangun
KG untuk SELURUH populasi tanpa scoping berisiko berat untuk RAM 8GB & Neo4j
Desktop lokal. Script ini MEWAJIBKAN salah satu dari --tags / --only-accepted
/ --limit sebagai filter scope, KECUALI kalau eksplisit pakai
--confirm-full-build (build seluruh populasi, TIDAK disarankan untuk
percobaan pertama).

INSTALL DEPENDENCY
-------------------
    pip install neo4j duckdb pandas python-dotenv
    # + kalau ada edge IS_RELATED_TO tipe 'Linked' dan TIDAK pakai --skip-embeddings:
    pip install sentence-transformers torch

SETUP .env (root project -- QUESTIONS_PARQUET, ANSWERS_PARQUET, NEO4J_* sudah
ada di sana)
----------------------------------------------------------------------------
    QUESTIONS_PARQUET=../00_datasource/merged/questions_raw_union.parquet
    ANSWERS_PARQUET=../00_datasource/merged/answers_raw_union.parquet
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j

CARA PAKAI
----------
    # Scope ke subtopik Python Framework & Library (sesuai Bab III.4.5.4)
    python 7_build_knowledge_graph.py --tags fastapi,pandas,django,sqlalchemy,scikit-learn --only-accepted

    # Dev/testing cepat, tanpa embedding (Linked edge fallback w=0.5)
    python 7_build_knowledge_graph.py --tags fastapi --limit 200 --skip-embeddings --dry-run

    # Build seluruh populasi (HATI-HATI -- bisa berat)
    python 7_build_knowledge_graph.py --confirm-full-build
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# HARUS di atas sebelum import faiss/torch/sentence-transformers -- pola sama
# dengan 4_test_infra.py (mencegah crash OMP/Rust-tokenizer di macOS).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
FULL_BUILD_WARN_THRESHOLD = 50_000  # jumlah question di atas ini butuh --confirm-full-build
BATCH_SIZE_DEFAULT = 2000


# ---------------------------------------------------------------------
# Logging (file + console), pola sama dengan script Kondisi A & 6_postlinks_to_sord.py
# ---------------------------------------------------------------------

import logging


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_build_knowledge_graph.log"

    logger = logging.getLogger("build_kg")
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
# STEP 1 - Koneksi Neo4j + constraint (pola sama dengan 4_test_infra.py)
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


def create_constraints(driver, database, logger):
    stmts = [
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
        "CREATE CONSTRAINT answer_id   IF NOT EXISTS FOR (a:Answer)   REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT tag_name    IF NOT EXISTS FOR (t:Tag)      REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT user_id     IF NOT EXISTS FOR (u:User)     REQUIRE u.id IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for stmt in stmts:
            session.run(stmt)
    logger.info("Constraint uniqueness (Question/Answer/Tag/User) siap.")


def write_batched(driver, database, cypher: str, df: pd.DataFrame, batch_size: int, logger, label: str):
    """UNWIND batch write -- pola batch (bukan satu-satu) supaya efisien
    untuk skala data SORD.

    PENTING: menerima DataFrame (bukan list-of-dict siap pakai) dan
    memanggil .to_dict("records") PER BATCH di dalam loop -- bukan sekali
    untuk seluruh tabel di awal. Untuk tabel jutaan baris (mis. 2,68 juta
    Question / 8,5 juta TAGGED_WITH), materialize semua dict sekaligus
    SEBELUM loop dimulai membuat proses kelihatan diam lama (tidak ada log
    apa pun) dan menahan seluruh list-of-dict di RAM bersamaan. Dengan
    konversi per-batch, log progress muncul sejak batch pertama & memori
    puncak jauh lebih kecil (cuma 1 batch tertahan di memori kapan saja).

    Log interval disesuaikan otomatis (~100 baris log total) supaya tabel
    kecil tetap log tiap batch, tapi tabel jutaan baris tidak membanjiri
    file log dengan ribuan baris log."""
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
# STEP 2 - Tentukan scope Question ID (filter --tags / --only-accepted / --limit)
# ---------------------------------------------------------------------

def resolve_question_scope(con: duckdb.DuckDBPyConnection, args, logger) -> pd.DataFrame:
    q_parquet = args.questions_parquet
    a_parquet = args.answers_parquet

    where_clauses = []
    if args.tags:
        tag_list = [t.strip().lower() for t in args.tags.split(",") if t.strip()]
        tag_conditions = " OR ".join(f"lower(Tags) LIKE '%<{t}>%'" for t in tag_list)
        where_clauses.append(f"({tag_conditions})")
        logger.info(f"Filter --tags aktif: {tag_list}")

    if args.only_accepted:
        where_clauses.append(
            f"""(AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
                 AND AcceptedAnswerId IN (SELECT Id FROM read_parquet('{a_parquet}')))"""
        )
        logger.info("Filter --only-accepted aktif (accepted answer harus matched ke tabel answers).")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    # ORDER BY hanya dipakai kalau --limit diisi (supaya hasil LIMIT
    # deterministic/representative). Full scope TANPA limit sengaja TIDAK
    # di-sort -- ORDER BY atas seluruh tabel (termasuk kolom Body yang besar)
    # butuh menahan semuanya di memori sekaligus untuk sorting, salah satu
    # penyebab OutOfMemoryException di run sebelumnya.
    order_sql = "ORDER BY Id" if args.limit else ""
    limit_sql = f"LIMIT {args.limit}" if args.limit else ""

    total_all, total_distinct = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT Id) FROM read_parquet('{q_parquet}')"
    ).fetchone()

    # PENTING: questions_raw_union.parquet SENGAJA TIDAK di-dedup penuh di
    # tahap merge (lihat 3_merge_sord_sources.py) -- duplikat Id muncul dari
    # overlap Title-match vs Body-match. Dedup dilakukan DI SINI (tahap
    # load-KG, sama seperti filosofi yang sudah didokumentasikan di script
    # merge), pakai GROUP BY + ANY_VALUE (hash aggregation DuckDB, bisa
    # spill ke disk lewat PRAGMA memory_limit) -- BUKAN load semua duplikat
    # ke pandas dulu baru di-drop_duplicates, supaya tetap aman untuk data
    # jutaan baris dengan kolom Body besar.
    scoped = con.execute(
        f"""
        SELECT
            Id,
            ANY_VALUE(Title) AS Title,
            ANY_VALUE(Body) AS Body,
            ANY_VALUE(Tags) AS Tags,
            ANY_VALUE(Score) AS Score,
            ANY_VALUE(ViewCount) AS ViewCount,
            ANY_VALUE(CreationDate) AS CreationDate,
            ANY_VALUE(AcceptedAnswerId) AS AcceptedAnswerId,
            ANY_VALUE(OwnerUserId) AS OwnerUserId
        FROM read_parquet('{q_parquet}')
        {where_sql}
        GROUP BY Id
        {order_sql}
        {limit_sql}
        """
    ).fetchdf()

    logger.info(f"Total baris di parquet (sebelum dedup Id) : {total_all:,}")
    logger.info(f"Question Id unik (setelah dedup)          : {total_distinct:,}")
    if total_all > total_distinct:
        logger.info(f"  -> {total_all - total_distinct:,} baris duplikat Id dibuang "
                     f"(overlap Title/Body-match dari union SORD)")
    logger.info(f"Questions setelah filter scope             : {len(scoped):,}")

    if not where_clauses and not args.limit:
        if total_distinct > FULL_BUILD_WARN_THRESHOLD and not args.confirm_full_build:
            raise SystemExit(
                f"[ABORT] {total_distinct:,} question unik terdeteksi TANPA filter scope "
                f"(--tags / --only-accepted / --limit). Ini berisiko berat untuk RAM 8GB "
                f"& Neo4j Desktop lokal. Pakai salah satu filter scope, atau eksplisit "
                f"--confirm-full-build kalau memang mau build seluruh populasi."
            )

    return scoped


# ---------------------------------------------------------------------
# STEP 3 - Load & siapkan Answers, Tags, Users untuk scope terpilih
# ---------------------------------------------------------------------

def load_scoped_answers(con, args, logger) -> pd.DataFrame:
    """Pakai JOIN terhadap view 'scope_questions' (sudah diregister di main()),
    BUKAN 'WHERE ParentId IN (...)' -- untuk build skala penuh (ratusan ribu+
    question ID), string SQL IN-list akan jadi sangat besar & lambat di-parse.
    JOIN lewat Arrow-registered view jauh lebih efisien di DuckDB.

    Dedup by Id juga diterapkan (defensif) dengan pola sama seperti Questions
    -- answers_raw_union SEHARUSNYA disjoint by construction (Contain vs
    LikeMinusContain), tapi GROUP BY di sini praktis tanpa biaya tambahan
    berarti dan melindungi dari perubahan data di masa depan."""
    answers = con.execute(
        f"""
        SELECT
            a.Id AS answer_id,
            ANY_VALUE(a.ParentId) AS question_id,
            ANY_VALUE(a.Body) AS Body,
            ANY_VALUE(a.Score) AS Score,
            ANY_VALUE(a.OwnerUserId) AS OwnerUserId
        FROM read_parquet('{args.answers_parquet}') a
        JOIN scope_questions s ON a.ParentId = s.id
        GROUP BY a.Id
        """
    ).fetchdf()
    logger.info(f"Answers dalam scope (setelah dedup Id)    : {len(answers):,}")
    return answers


def compute_answer_weights(questions: pd.DataFrame, answers: pd.DataFrame, logger) -> pd.DataFrame:
    """Tandai is_accepted per answer, hitung per-thread min-max normalize(score)
    untuk bobot HAS_ANSWER (§2.1 desain skema). Sepenuhnya VEKTORISASI (tanpa
    .apply(axis=1)) -- penting untuk skala penuh, .apply row-wise di ratusan
    ribu baris jadi bottleneck signifikan."""
    answers = answers.copy()

    accepted_map = questions.set_index("Id")["AcceptedAnswerId"]
    answers["is_accepted"] = answers["answer_id"] == answers["question_id"].map(accepted_map)

    stats = answers.groupby("question_id")["Score"].agg(["min", "max"]).rename(
        columns={"min": "min_score", "max": "max_score"}
    )
    answers = answers.join(stats, on="question_id")

    denom = (answers["max_score"] - answers["min_score"]).replace(0, np.nan)
    raw_weight = (answers["Score"] - answers["min_score"]) / denom
    answers["has_answer_weight"] = raw_weight.fillna(0.5).clip(0, 1)

    n_accepted = int(answers["is_accepted"].sum())
    logger.info(f"Answer accepted (HAS_ACCEPTED_ANSWER): {n_accepted:,}")
    logger.info(f"Answer non-accepted (HAS_ANSWER)     : {len(answers) - n_accepted:,}")
    return answers


def parse_tags(tags_str) -> list:
    if not tags_str or pd.isna(tags_str):
        return []
    return [t for t in str(tags_str).replace(">", "").split("<") if t]


def build_tag_edges(questions: pd.DataFrame, logger) -> pd.DataFrame:
    """Vektorisasi pakai regex findall + explode (BUKAN loop iterrows())
    -- penting untuk skala penuh, iterrows() di ratusan ribu baris sangat
    lambat dibanding operasi vektor pandas."""
    tags_lists = questions["Tags"].fillna("").astype(str).str.findall(r"<([^<>]+)>")
    df = pd.DataFrame({"question_id": questions["Id"].values, "tag_name": tags_lists})
    df = df.explode("tag_name").dropna(subset=["tag_name"])
    df["question_id"] = df["question_id"].astype(int)
    df = df.reset_index(drop=True)
    logger.info(f"Edge TAGGED_WITH (question-tag pairs): {len(df):,}")
    logger.info(f"Tag unik                              : {df['tag_name'].nunique():,}")
    return df


def load_users_reputation(con, args, user_ids: list, logger) -> pd.DataFrame:
    """Sumber reputation, dicoba berurutan:
    1. users.parquet (hasil 8_convert_users_xml.py dari Users.xml dump PENUH
       StackOverflow -- lebih lengkap/otoritatif daripada FilteredUsers.csv)
    2. FilteredUsers.csv (fallback lama, subset SORD)
    """
    if not user_ids:
        return pd.DataFrame(columns=["user_id", "reputation"])

    unique_ids = pd.unique(pd.Series([int(i) for i in user_ids if pd.notna(i)]))
    # Registered view + JOIN, sama alasannya dengan load_scoped_answers --
    # hindari IN-list raksasa untuk build skala penuh.
    con.register("scope_users", pd.DataFrame({"id": unique_ids}))

    users_parquet = Path(args.users_parquet)
    users_csv = Path(args.raw_dir) / "additional-metadata" / "FilteredUsers.csv"

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
        logger.warning(
            f"Tidak ditemukan {users_parquet} maupun {users_csv} -- AUTHOR_TRUST dilewati. "
            f"Jalankan 8_convert_users_xml.py dulu kalau punya Users.xml."
        )
        con.unregister("scope_users")
        return pd.DataFrame(columns=["user_id", "reputation"])

    con.unregister("scope_users")
    logger.info(f"User dengan reputation ditemukan     : {len(users):,} / {len(unique_ids):,} unik dibutuhkan")
    return users


# ---------------------------------------------------------------------
# STEP 4 - Hitung trustScore node (Question & Answer) -- Tabel II.2
# ---------------------------------------------------------------------

def minmax_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def compute_trust_scores(questions: pd.DataFrame, answers: pd.DataFrame, logger):
    q = questions.copy()
    q["norm_score"] = minmax_normalize(q["Score"].fillna(0))
    q["norm_log_view"] = minmax_normalize(np.log1p(q["ViewCount"].fillna(0)))
    q["trustScore"] = 0.7 * q["norm_score"] + 0.3 * q["norm_log_view"]

    a = answers.copy()
    a["norm_score"] = minmax_normalize(a["Score"].fillna(0))
    reputation_map = a.get("reputation", pd.Series(dtype=float))
    a["norm_reputation"] = (
        minmax_normalize(a["reputation"].fillna(0)) if "reputation" in a.columns else 0.0
    )
    a["trustScore"] = (
        0.5 * a["norm_score"]
        + 0.3 * a["is_accepted"].astype(float)
        + 0.2 * a["norm_reputation"]
    )

    logger.info(
        f"trustScore Question -- min={q['trustScore'].min():.3f} "
        f"max={q['trustScore'].max():.3f} mean={q['trustScore'].mean():.3f}"
    )
    logger.info(
        f"trustScore Answer   -- min={a['trustScore'].min():.3f} "
        f"max={a['trustScore'].max():.3f} mean={a['trustScore'].mean():.3f}"
    )
    return q, a


# ---------------------------------------------------------------------
# STEP 5 - IS_RELATED_TO: Duplicate=1.0, Linked=cosine similarity embedding
# ---------------------------------------------------------------------

def load_question_links(con, args, logger) -> pd.DataFrame:
    """Pakai double-JOIN terhadap view 'scope_questions' (diregister di
    main()) -- bukan 'WHERE post_id IN (...) AND related_post_id IN (...)',
    dengan alasan sama seperti load_scoped_answers (skala penuh)."""
    links_path = Path(args.question_links_parquet)
    if not links_path.exists():
        logger.warning(
            f"question_links.parquet tidak ditemukan di {links_path} -- "
            "IS_RELATED_TO dilewati. Jalankan 6_postlinks_to_sord.py dulu kalau perlu."
        )
        return pd.DataFrame(columns=["post_id", "related_post_id", "link_type_label", "creation_date"])

    links = con.execute(
        f"""
        SELECT l.post_id, l.related_post_id, l.link_type_label, l.creation_date
        FROM read_parquet('{links_path.as_posix()}') l
        JOIN scope_questions s1 ON l.post_id = s1.id
        JOIN scope_questions s2 ON l.related_post_id = s2.id
        """
    ).fetchdf()
    logger.info(f"Edge IS_RELATED_TO dalam scope       : {len(links):,} "
                f"(endpoint di luar scope otomatis dibuang)")
    return links


def compute_related_to_weights(links: pd.DataFrame, questions: pd.DataFrame, args, logger) -> pd.DataFrame:
    if links.empty:
        links["weight"] = []
        return links

    links = links.copy()
    links["weight"] = np.nan
    links.loc[links["link_type_label"] == "Duplicate", "weight"] = 1.0

    linked_mask = links["link_type_label"] == "Linked"
    n_linked = int(linked_mask.sum())

    if n_linked == 0:
        return links

    if args.skip_embeddings:
        logger.info(f"--skip-embeddings aktif: {n_linked:,} edge 'Linked' pakai fallback weight=0.5")
        links.loc[linked_mask, "weight"] = 0.5
        return links

    unique_ids = pd.unique(
        pd.concat([links.loc[linked_mask, "post_id"], links.loc[linked_mask, "related_post_id"]])
    )
    logger.info(f"Menghitung cosine similarity untuk {n_linked:,} edge 'Linked' "
                f"({len(unique_ids):,} question unik perlu di-encode, model: {args.embed_model})...")

    from sentence_transformers import SentenceTransformer
    import torch

    # Override thread-limit global (biasanya di-set 1 di awal script untuk
    # cegah crash OMP saat load FAISS) KHUSUS untuk encoding -- proses
    # embedding murni PyTorch, bukan FAISS, jadi lebih aman multi-thread.
    # Kalau ini bikin crash lagi di mesin kamu, set --embed-threads 1.
    try:
        torch.set_num_threads(args.embed_threads)
        logger.info(f"  torch threads utk embedding: {args.embed_threads}")
    except Exception as e:
        logger.warning(f"  Gagal set torch threads ({e}), lanjut pakai default.")

    # Vektorisasi penyusunan teks (bukan .apply row-wise) -- lebih cepat
    # untuk ratusan ribu baris.
    qsub = questions.set_index("Id").loc[unique_ids]
    texts = (
        qsub["Title"].fillna("").astype(str) + " " + qsub["Body"].fillna("").astype(str).str.slice(0, 1000)
    ).tolist()

    model = SentenceTransformer(args.embed_model)

    # Encode per CHUNK dengan progress log tiap chunk + estimasi ETA --
    # sebelumnya model.encode() dipanggil sekali untuk SEMUA teks tanpa
    # output apa pun selama proses berjalan, jadi kelihatan seperti "stuck"
    # padahal cuma lambat (apalagi dengan --embed-threads rendah).
    chunk_size = args.embed_log_chunk
    all_embeddings = []
    t0 = time.time()
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        emb = model.encode(chunk, show_progress_bar=False, batch_size=args.embed_batch_size, convert_to_numpy=True)
        all_embeddings.append(emb)
        done = min(i + chunk_size, len(texts))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(texts) - done) / rate if rate > 0 else float("nan")
        logger.info(f"  ... embedding {done:,}/{len(texts):,} question unik "
                    f"({rate:.1f} teks/detik, ETA ~{eta / 60:.1f} menit)")

    embeddings = np.vstack(all_embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    # Cosine similarity divektorisasi (numpy einsum), BUKAN .apply(axis=1)
    # dengan dict lookup per baris -- jauh lebih cepat untuk ratusan ribu edge.
    id_to_idx = {qid: i for i, qid in enumerate(qsub.index)}
    post_idx = links.loc[linked_mask, "post_id"].map(id_to_idx)
    related_idx = links.loc[linked_mask, "related_post_id"].map(id_to_idx)
    valid = post_idx.notna() & related_idx.notna()

    sims = np.full(n_linked, 0.5)
    if valid.any():
        pi = post_idx[valid].astype(int).to_numpy()
        ri = related_idx[valid].astype(int).to_numpy()
        dots = np.einsum("ij,ij->i", embeddings[pi], embeddings[ri])
        sims[valid.to_numpy()] = np.clip(dots, 0.0, None)

    links.loc[linked_mask, "weight"] = sims
    logger.info(f"Cosine similarity selesai. Rata-rata weight 'Linked': "
                f"{links.loc[linked_mask, 'weight'].mean():.3f}")
    return links


# ---------------------------------------------------------------------
# STEP 6 - AUTHOR_TRUST weight: normalize(log(reputation)), clip P99
# ---------------------------------------------------------------------

def compute_author_trust_weight(users: pd.DataFrame, logger) -> pd.DataFrame:
    if users.empty:
        users["weight"] = []
        return users
    users = users.copy()
    users["reputation"] = users["reputation"].fillna(0).clip(lower=0)
    p99 = users["reputation"].quantile(0.99)
    if p99 <= 0:
        p99 = users["reputation"].max() or 1
    users["weight"] = (np.log1p(users["reputation"]) / np.log1p(p99)).clip(0, 1)
    logger.info(f"P99 reputation (basis normalisasi)  : {p99:,.0f}")
    return users


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--question-links-parquet", default="../00_datasource/merged/question_links.parquet")
    parser.add_argument("--raw-dir", default="../00_datasource/raw")
    parser.add_argument("--users-parquet", default="../00_datasource/merged/users.parquet",
                         help="Hasil 8_convert_users_xml.py (dump Users.xml penuh). "
                              "Kalau tidak ada, fallback ke FilteredUsers.csv di --raw-dir.")
    parser.add_argument("--tags", default=None, help="Filter tag, comma-separated (mis. fastapi,pandas,django)")
    parser.add_argument("--only-accepted", action="store_true", help="Hanya question dgn accepted answer matched")
    parser.add_argument("--limit", type=int, default=None, help="Batas jumlah question (dev/testing)")
    parser.add_argument("--confirm-full-build", action="store_true",
                         help="Wajib diisi kalau TIDAK pakai filter scope apa pun")
    parser.add_argument("--skip-embeddings", action="store_true",
                         help="Skip cosine similarity utk edge Linked (fallback weight=0.5, lebih cepat)")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embed-threads", type=int, default=4,
                         help="Thread PyTorch khusus utk embedding (default 4; turunkan ke 1 kalau crash)")
    parser.add_argument("--embed-batch-size", type=int, default=64,
                         help="Batch size encode SentenceTransformer (default 64, boleh dinaikkan kalau RAM cukup)")
    parser.add_argument("--embed-log-chunk", type=int, default=2000,
                         help="Jumlah teks per chunk sebelum progress di-log (default 2000)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--memory-limit", default="2GB",
                         help="Batas memori DuckDB. Turunkan kalau masih OOM (mis. '1GB').")
    parser.add_argument("--threads", type=int, default=2,
                         help="Jumlah thread DuckDB. Turunkan ke 1 kalau masih OOM.")
    parser.add_argument("--dry-run", action="store_true", help="Hitung semua node/edge, TIDAK menulis ke Neo4j")
    args = parser.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET / ANSWERS_PARQUET harus diset di .env atau lewat CLI")
        sys.exit(1)

    logger, log_path = setup_logging()
    logger.info("KONSTRUKSI KNOWLEDGE GRAPH -- Kondisi C (GraphRAG)")
    logger.info(f"Log file: {log_path}")
    t_start = time.time()

    # PENTING: pakai database on-disk (bukan ':memory:') + PRAGMA memory_limit
    # eksplisit, supaya DuckDB spill hasil intermediate ke disk saat RAM
    # tidak cukup, alih-alih crash OutOfMemoryException -- pola sama dengan
    # 3_merge_sord_sources.py. Workspace file & temp dir TIDAK dihapus
    # otomatis (boleh dihapus manual / masuk .gitignore setelah run selesai).
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
    logger.info(f"[config] memory_limit={args.memory_limit} threads={args.threads} "
                f"temp_directory={tmp_dir} workspace_db={db_path}")

    step(logger, 1, "Tentukan scope Question (filter --tags / --only-accepted / --limit)")
    questions = resolve_question_scope(con, args, logger)
    if questions.empty:
        logger.warning("Tidak ada question yang cocok dengan filter scope. Berhenti.")
        return
    question_ids = questions["Id"].tolist()

    # Registrasi SEKALI di sini, dipakai oleh load_scoped_answers() &
    # load_question_links() lewat JOIN (bukan IN-list) -- penting untuk
    # build skala penuh (ratusan ribu+ question ID).
    con.register("scope_questions", pd.DataFrame({"id": question_ids}))

    step(logger, 2, "Load Answers dalam scope + hitung bobot HAS_ANSWER/HAS_ACCEPTED_ANSWER")
    answers = load_scoped_answers(con, args, logger)
    answers = compute_answer_weights(questions, answers, logger)

    step(logger, 3, "Load User (reputation) untuk AUTHOR_TRUST")
    all_owner_ids = pd.concat([questions["OwnerUserId"], answers["OwnerUserId"]]).dropna().unique().tolist()
    users = load_users_reputation(con, args, all_owner_ids, logger)
    users = compute_author_trust_weight(users, logger)
    reputation_map = users.set_index("user_id")["reputation"].to_dict()
    answers["reputation"] = answers["OwnerUserId"].map(reputation_map)

    step(logger, 4, "Hitung trustScore node (Question & Answer) -- Tabel II.2")
    questions, answers = compute_trust_scores(questions, answers, logger)

    step(logger, 5, "Bangun edge TAGGED_WITH (parsing kolom Tags)")
    tag_edges = build_tag_edges(questions, logger)

    step(logger, 6, "Load & hitung bobot IS_RELATED_TO (question_links -- Duplicate/Linked)")
    links = load_question_links(con, args, logger)
    links = compute_related_to_weights(links, questions, args, logger)

    step(logger, 7, "Ringkasan sebelum tulis ke Neo4j")
    logger.info(f"Question node   : {len(questions):,}")
    logger.info(f"Answer node     : {len(answers):,}")
    logger.info(f"Tag node (unik) : {tag_edges['tag_name'].nunique() if not tag_edges.empty else 0:,}")
    logger.info(f"User node       : {len(users):,}")
    logger.info(f"Edge HAS_ACCEPTED_ANSWER : {int(answers['is_accepted'].sum()):,}")
    logger.info(f"Edge HAS_ANSWER          : {int((~answers['is_accepted']).sum()):,}")
    logger.info(f"Edge TAGGED_WITH         : {len(tag_edges):,}")
    logger.info(f"Edge IS_RELATED_TO       : {len(links):,}")
    logger.info(f"Edge AUTHOR_TRUST        : {len(answers.dropna(subset=['reputation'])):,}")

    if args.dry_run:
        logger.info("--dry-run aktif: TIDAK menulis ke Neo4j. Selesai.")
        return

    step(logger, 8, "Koneksi Neo4j + setup constraint")
    driver, database = connect_neo4j(logger)
    create_constraints(driver, database, logger)

    try:
        step(logger, 9, "Tulis node: Question, Answer, Tag, User")

        # domainTag divektorisasi pakai regex extract (ambil tag pertama),
        # BUKAN .apply(parse_tags) row-wise -- untuk 2,68 juta baris ini
        # signifikan lebih cepat.
        domain_tag = questions["Tags"].fillna("").astype(str).str.extract(r"<([^<>]+)>")[0]

        q_df = questions.assign(
            creationDate=questions["CreationDate"].astype(str),
            domainTag=domain_tag,
        )[["Id", "Title", "Body", "Score", "ViewCount", "creationDate", "domainTag", "trustScore"]].rename(
            columns={"Id": "id", "Title": "title", "Body": "body", "Score": "score", "ViewCount": "viewCount"}
        )
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MERGE (q:Question {id: row.id})
            SET q.title = row.title, q.body = row.body, q.score = row.score,
                q.viewCount = row.viewCount, q.creationDate = row.creationDate,
                q.domainTag = row.domainTag, q.trustScore = row.trustScore
            """,
            q_df, args.batch_size, logger, "Question",
        )

        a_df = answers.assign(
            authorReputation=answers["reputation"].fillna(0),
        )[["answer_id", "Body", "Score", "is_accepted", "authorReputation", "trustScore"]].rename(
            columns={"answer_id": "id", "Body": "body", "Score": "score", "is_accepted": "isAccepted"}
        )
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MERGE (a:Answer {id: row.id})
            SET a.body = row.body, a.score = row.score, a.isAccepted = row.isAccepted,
                a.authorReputation = row.authorReputation, a.trustScore = row.trustScore
            """,
            a_df, args.batch_size, logger, "Answer",
        )

        tag_counts = tag_edges["tag_name"].value_counts() if not tag_edges.empty else pd.Series(dtype=int)
        tag_df = tag_counts.rename_axis("name").reset_index(name="questionCount")
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MERGE (t:Tag {name: row.name})
            SET t.questionCount = row.questionCount
            """,
            tag_df, args.batch_size, logger, "Tag",
        )

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

        step(logger, 10, "Tulis edge: HAS_ACCEPTED_ANSWER, HAS_ANSWER")
        accepted_df = answers[answers["is_accepted"]][["question_id", "answer_id"]]
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (q:Question {id: row.question_id}), (a:Answer {id: row.answer_id})
            MERGE (q)-[r:HAS_ACCEPTED_ANSWER]->(a)
            SET r.weight = 1.0
            """,
            accepted_df, args.batch_size, logger, "HAS_ACCEPTED_ANSWER",
        )

        other_df = answers[~answers["is_accepted"]][
            ["question_id", "answer_id", "has_answer_weight"]
        ].rename(columns={"has_answer_weight": "weight"})
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (q:Question {id: row.question_id}), (a:Answer {id: row.answer_id})
            MERGE (q)-[r:HAS_ANSWER]->(a)
            SET r.weight = row.weight
            """,
            other_df, args.batch_size, logger, "HAS_ANSWER",
        )

        step(logger, 11, "Tulis edge: TAGGED_WITH")
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (q:Question {id: row.question_id}), (t:Tag {name: row.tag_name})
            MERGE (q)-[r:TAGGED_WITH]->(t)
            SET r.weight = 1.0
            """,
            tag_edges, args.batch_size, logger, "TAGGED_WITH",
        )

        step(logger, 12, "Tulis edge: IS_RELATED_TO")
        related_df = links.rename(
            columns={"post_id": "q1", "related_post_id": "q2", "link_type_label": "link_type"}
        )[["q1", "q2", "weight", "link_type", "creation_date"]] if not links.empty else pd.DataFrame(
            columns=["q1", "q2", "weight", "link_type", "creation_date"]
        )
        write_batched(
            driver, database,
            """
            UNWIND $rows AS row
            MATCH (q1:Question {id: row.q1}), (q2:Question {id: row.q2})
            MERGE (q1)-[r:IS_RELATED_TO]->(q2)
            SET r.weight = row.weight, r.link_type = row.link_type, r.creation_date = row.creation_date
            """,
            related_df, args.batch_size, logger, "IS_RELATED_TO",
        )

        step(logger, 13, "Tulis edge: AUTHOR_TRUST")
        trust_df = answers.dropna(subset=["reputation"])[["answer_id", "OwnerUserId"]].copy()
        trust_df["weight"] = trust_df["OwnerUserId"].map(users.set_index("user_id")["weight"].to_dict())
        trust_df = trust_df.rename(columns={"OwnerUserId": "user_id"}).dropna(subset=["weight"])
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

        step(logger, 14, "Verifikasi -- hitung ulang node & edge langsung dari Neo4j")
        with driver.session(database=database) as session:
            counts = session.run(
                """
                CALL {
                    MATCH (n:Question) RETURN 'Question' AS label, count(n) AS n
                    UNION ALL MATCH (n:Answer) RETURN 'Answer' AS label, count(n) AS n
                    UNION ALL MATCH (n:Tag) RETURN 'Tag' AS label, count(n) AS n
                    UNION ALL MATCH (n:User) RETURN 'User' AS label, count(n) AS n
                }
                RETURN label, n ORDER BY label
                """
            ).data()
            for row in counts:
                logger.info(f"  Neo4j {row['label']:<10}: {row['n']:,}")

            edge_counts = session.run(
                """
                CALL {
                    MATCH ()-[r:HAS_ACCEPTED_ANSWER]->() RETURN 'HAS_ACCEPTED_ANSWER' AS rel, count(r) AS n
                    UNION ALL MATCH ()-[r:HAS_ANSWER]->() RETURN 'HAS_ANSWER' AS rel, count(r) AS n
                    UNION ALL MATCH ()-[r:TAGGED_WITH]->() RETURN 'TAGGED_WITH' AS rel, count(r) AS n
                    UNION ALL MATCH ()-[r:IS_RELATED_TO]->() RETURN 'IS_RELATED_TO' AS rel, count(r) AS n
                    UNION ALL MATCH ()-[r:AUTHOR_TRUST]->() RETURN 'AUTHOR_TRUST' AS rel, count(r) AS n
                }
                RETURN rel, n ORDER BY rel
                """
            ).data()
            for row in edge_counts:
                logger.info(f"  Neo4j {row['rel']:<22}: {row['n']:,}")

    finally:
        driver.close()

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info(f"SELESAI dalam {elapsed:.1f}s")
    logger.info("=" * 78)


if __name__ == "__main__":
    sys.exit(main())