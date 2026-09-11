"""
condition_a_baseline_replication.py
=====================================
Fase 1 — Replikasi metodologi baseline Da Silva, Samhi, & Khomh (2025)
"LLMs and Stack Overflow discussions: Reliability, impact, and challenges"
menggunakan dataset SORD dan GPT-4o-mini sebagai Kondisi A (LLM murni,
tanpa RAG/GraphRAG).

METODOLOGI YANG DIREPLIKASI (persis mengikuti paper asli)
------------------------------------------------------------
1. Kriteria inklusi: hanya pertanyaan dengan accepted answer.
2. Kriteria eksklusi: title+description+tags > 2048 token (dihitung pakai
   tiktoken, sama seperti paper asli -- lihat footnote 6 paper).
3. Ukuran sampel: 384 pertanyaan (justifikasi statistik paper asli: 95%
   confidence level, 5% margin of error).
4. Struktur prompt 4-turn chat dialog, PERSIS mereplikasi
   `get_base_message()` di ll_model.py (repo asli leusonmario/chat-stack):
   system (persona umum software engineering) -> user (persona spesifik
   dibentuk dari tags) -> assistant (priming/konfirmasi persona) -> user
   (pertanyaan + deskripsi). Wording tiap turn dipertahankan sama dengan
   repo asli supaya struktur percakapan benar-benar tereplikasi, bukan
   hanya diparafrasekan seperti versi sebelumnya.
5. Konfigurasi model: DEFAULT (tidak override temperature/parameter lain)
   -- meniru pola pemakaian LLM oleh user biasa. CATATAN (open issue,
   belum final): klaim "default settings" ini mengacu ke teks paper asli;
   kode publik chat-stack sendiri sebenarnya SET temperature eksplisit
   untuk sebagian model (chatgpt.py: temperature=0.6, max_tokens=2048;
   llama.py: temperature=1.0, top_p=1.0, top_k=50, do_sample=True) --
   perlu diverifikasi ulang terhadap teks paper & didokumentasikan
   sebagai known discrepancy di Bab III/IV sebelum finalisasi klaim.
6. Satu kali panggilan API per pertanyaan (tidak ada retry/resampling).
7. Metrik: cosine similarity via sentence-transformers all-MiniLM-L6-v2
   (model yang sama dipakai paper asli).

PERBEDAAN DISENGAJA DARI PAPER ASLI
-------------------------------------
- Model: konfigurabel via --provider/--model (openai/anthropic/local) --
  desain pluggable supaya bisa ganti model kapan saja tanpa ubah kode,
  yang penting KONSISTEN dipakai di ketiga Kondisi A/B/C untuk
  perbandingan yang adil.
- Sumber pertanyaan: SORD (bukan SO dump generik) -- sesuai desain
  penelitian yang berfokus pada domain rekomendasi software.

INSTALL DEPENDENCY
-------------------
    pip install duckdb tiktoken pandas python-dotenv sentence-transformers
    # + salah satu provider LLM:
    pip install openai            # untuk --provider openai
    pip install anthropic         # untuk --provider anthropic (Claude)
    pip install transformers torch  # untuk --provider local (mis. LLaMA)

SETUP .env (WAJIB — semua konfigurasi dibaca dari sini, CLI argument bersifat
opsional untuk override sesaat kalau perlu)
----------------------------------------------------------------------------
    # Path data (relatif terhadap folder a_pure_llm/, sejajar
    # dengan 00_datasource/ di root project)
    QUESTIONS_PARQUET=../00_datasource/merged/questions_raw_union.parquet
    ANSWERS_PARQUET=../00_datasource/merged/answers_raw_union.parquet
    OUTPUT_DIR=results
    # ^ Nama file JSONL TIDAK perlu diset manual lagi -- otomatis dibuat
    #   dari provider+model yang benar-benar dipakai run ini, format:
    #   condition_a_{provider}_{safe_model}.jsonl, dengan safe_model =
    #   model.replace("/", "-").replace(":", "-").replace(".", "-")
    #   Contoh: provider=anthropic, model=claude-sonnet-4-5
    #           -> results/condition_a_anthropic_claude-sonnet-4-5.jsonl
    #   Ini menggantikan mekanisme prefix "dev_ollama_" sebelumnya --
    #   nama file sekarang otomatis unik per provider/model sehingga tidak
    #   mungkin bentrok/menimpa hasil provider lain. Kalau butuh path
    #   custom (mis. nama khusus untuk arsip), pakai --output untuk
    #   override manual (mengabaikan auto-naming).

    # Parameter eksperimen
    N_SAMPLE=384
    SEED=42
    OVERSAMPLE_POOL=1200
    # ^ pool acak awal sebelum filter token (default otomatis 3x N_SAMPLE
    #   kalau tidak diset). Optimasi performa: token counting (Step 2) cuma
    #   dijalankan pada pool ini, BUKAN seluruh populasi kandidat (~1.27 juta
    #   pertanyaan) -- jauh lebih cepat, tanpa mengubah validitas statistik
    #   sampling akhir (tetap random sample dari populasi yang sudah difilter).

    # Provider LLM: openai / anthropic / local
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini

    # API key sesuai provider yang dipakai
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    # (ANTHROPIC_API_KEY didapat dari console.anthropic.com, ini BEDA
    #  dari langganan Claude.ai/Claude Pro biasa -- billing terpisah)

CARA PAKAI
----------
    # Semua konfigurasi diambil dari .env, tinggal jalankan:
    python condition_a_baseline_replication.py

    # CLI argument tetap bisa dipakai untuk override sesaat tanpa edit .env,
    # mis. pilot run cepat dengan sample lebih kecil:
    python condition_a_baseline_replication.py --n-sample 10

Skrip ini RESUMABLE: kalau terhenti di tengah jalan (API error, koneksi
putus, dll.), jalankan ulang command yang sama -- pertanyaan yang sudah
diproses (tercatat di file output) otomatis dilewati.
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
from llm.client_factory import get_llm_client
from llm.prompts import build_base_messages

TOKEN_LIMIT = 2048  # sesuai batasan paper asli

# Dibind ke logger.info di dalam main() setelah setup_logging() dipanggil.
# Default ke print supaya modul ini tetap bisa di-import/dites terpisah
# tanpa perlu setup logging dulu.
log = print

LOG_DIR = Path("logs")


def setup_logging(provider: str, model: str) -> tuple[logging.Logger, Path]:
    """Setiap run baseline dapat file log sendiri (timestamped) + tetap
    tampil di console seperti biasa. Nama file mengandung provider+model
    supaya gampang dibedakan run dev (ollama) vs run resmi (openai/anthropic)
    tanpa perlu buka isinya dulu."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model.replace("/", "_").replace(":", "_").replace(".", "_")
    log_path = LOG_DIR / f"{ts}_{provider}_{safe_model}.log"

    logger = logging.getLogger("baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # cegah duplikat handler kalau main() sempat dipanggil >1x

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))  # console tetap rapi tanpa timestamp
    logger.addHandler(console_handler)

    return logger, log_path


def append_run_history(record: dict):
    """Satu baris ringkasan per run, ditambahkan ke logs/run_history.jsonl.
    Ini yang dipakai nanti untuk bikin tabel perbandingan run dev (Qwen)
    vs run resmi (GPT-4o-mini/Claude) di Bab IV/VI -- tidak perlu buka
    file .log detail satu-satu."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    history_path = LOG_DIR / "run_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return history_path


# ---------------------------------------------------------------------
# 1. Ambil kandidat pertanyaan dari Parquet (dedup + filter accepted answer)
# ---------------------------------------------------------------------

def get_candidate_questions(con: duckdb.DuckDBPyConnection, questions_parquet: str,
                             answers_parquet: str, oversample_pool: int, seed: int) -> pd.DataFrame:
    """Dedup by Id (pilih 1 baris representatif) + filter hanya yang punya
    accepted answer, LALU langsung ambil oversample pool kecil secara acak
    (bukan seluruh populasi) menggunakan `USING SAMPLE` DuckDB.

    PENTING (fix): SORD memfilter Questions dan Answers secara independen
    berdasarkan keyword di teks masing-masing, sehingga ~79% AcceptedAnswerId
    di Questions TIDAK punya baris yang cocok di Answers parquet (lihat
    check_accepted_answer_match.py). Kalau filter "harus match ke Answers"
    baru dicek belakangan (di get_accepted_answers), sample n yang sudah
    diambil di sini akan banyak terbuang percuma (mis. dari 30 sample cuma
    ~5-6 yang valid, sesuai match rate populasi ~21%). Makanya filter match
    ini dilakukan di sini, SEBELUM sampling -- via semi-join ke Answers.Id --
    supaya oversample_pool yang di-sample sudah pasti bisa dipakai semua
    (hanya tereliminasi oleh filter token di Step 2, bukan oleh missing
    accepted answer lagi).

    PENTING (optimasi): kita TIDAK perlu materialize/hitung token untuk
    seluruh ~270rb kandidat matched kalau target akhir cuma 384 pertanyaan.
    DuckDB melakukan sampling di level SQL sebelum data ditransfer ke
    pandas, jauh lebih cepat daripada sample belakangan pakai df.sample()
    setelah semua baris (termasuk kolom Body yang berat) sudah termuat ke
    memori Python. oversample_pool diambil sedikit lebih besar dari n_sample
    akhir untuk mengantisipasi baris yang nanti tereliminasi oleh filter
    token limit (Step 2) -- bukan lagi untuk mengantisipasi missing answer."""
    log(f"[1/6] Mengambil oversample pool ({oversample_pool} kandidat yang SUDAH "
          f"dipastikan punya accepted answer matching di Answers parquet)...")
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
    log(f"      Oversample pool diambil: {len(df):,} kandidat (semua sudah "
          f"punya accepted answer matching; hanya subset ini yang akan "
          f"dihitung token-nya di Step 2)")
    return df


# ---------------------------------------------------------------------
# 2. Filter token limit (persis kriteria eksklusi paper asli)
# ---------------------------------------------------------------------

def filter_by_token_limit(df: pd.DataFrame, limit: int = TOKEN_LIMIT) -> pd.DataFrame:
    import tiktoken

    log(f"[2/6] Menghitung token (title+body+tags) & filter <= {limit} token "
          f"(kriteria eksklusi persis paper asli)...")
    enc = tiktoken.get_encoding("cl100k_base")  # sesuai footnote paper (pakai library tiktoken)

    def count_tokens(row):
        text = f"{row['Title']} {row['Body']} {row['Tags']}"
        return len(enc.encode(str(text)))

    df = df.copy()
    df["n_tokens"] = df.apply(count_tokens, axis=1)
    before = len(df)
    df_filtered = df[df["n_tokens"] <= limit]
    excluded = before - len(df_filtered)
    log(f"      {before:,} kandidat -> {len(df_filtered):,} lolos filter token "
          f"({excluded:,} dikecualikan karena > {limit} token, "
          f"{excluded/before*100:.1f}%)")
    return df_filtered


# ---------------------------------------------------------------------
# 3. Sampling acak (384, seed untuk reprodusibilitas)
# ---------------------------------------------------------------------

def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """NESTED SAMPLING (PENTING): populasi kandidat (`df`, hasil filter token)
    di-permutasi SEKALI dgn seed tetap, lalu N pertama diambil sbg PREFIX.
    Ini menjamin sample n=30 adalah SUBSET PERSIS dari sample n=100, yang
    adalah subset dari n=384 -- SELAMA oversample_pool & seed IDENTIK di
    semua run (lihat --oversample-pool / MAX_PLANNED_N_SAMPLE di main()).

    Kenapa TIDAK cukup pakai df.sample(n=n, random_state=seed) langsung
    (versi lama): sampling tanpa-pengembalian pandas/numpy dgn seed sama
    tapi n BERBEDA tidak dijamin nested -- draw n=30 dan draw n=100 dari
    populasi sama bisa menghasilkan himpunan yang TIDAK bertumpuk penuh,
    sehingga run n=100 dianggap "melanjutkan" n=30 padahal sebenarnya draw
    independen baru (biaya API terbuang, dan n=30/n=100 tidak bisa dianalisis
    sbg subset kumulatif). Solusi: permutasi SEKALI di awal, lalu prefix
    n pertama -- prefix dari permutasi yang sama pasti nested berapa pun n."""
    log(f"[3/6] Mengambil {n} pertanyaan PERTAMA dari permutasi tetap (seed={seed}) "
          f"-- prefix ini NESTED thd n_sample lain yg pakai oversample_pool & seed sama...")
    df_permuted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(df_permuted) < n:
        log(f"      [WARN] Populasi hasil filter ({len(df_permuted)}) < target sample ({n}), "
              f"pakai semua yang tersedia. Kalau target 384 tidak tercapai, jalankan "
              f"ulang dengan --oversample-pool lebih besar (mis. {n*5}).")
        return df_permuted
    return df_permuted.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------
# 4. Ambil accepted answer dari Answers parquet
# ---------------------------------------------------------------------

def get_accepted_answers(con: duckdb.DuckDBPyConnection, answers_parquet: str,
                          accepted_answer_ids: list) -> pd.DataFrame:
    log("[4/6] Mengambil teks accepted answer dari Answers parquet...")
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
# 5. Prompt LLM & pemanggilan model -- lihat llm/prompts.py dan
#    llm/client_factory.py (dipakai bersama semua kondisi eksperimen)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 6. Similarity scoring
# ---------------------------------------------------------------------

def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    from sentence_transformers import util
    emb_a = embed_model.encode(str(text_a), convert_to_tensor=True)
    emb_b = embed_model.encode(str(text_b), convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


# ---------------------------------------------------------------------
# Proses per-pertanyaan (dipakai CLI main() DAN dashboard backend/engine_service)
# ---------------------------------------------------------------------

def process_sample(sample_df: pd.DataFrame, llm_client, call_llm_fn, embed_model, model: str,
                    output_path: Path, already_done: set | None = None, on_progress=None,
                    check_cancel=None) -> list[dict]:
    """Proses satu-per-satu sample_df: panggil LLM, hitung cosine similarity,
    tulis ke output_path (append, resumable via `already_done`). Diekstrak
    dari main() supaya SATU-SATUNYA implementasi loop generasi dipakai baik
    oleh CLI maupun oleh dashboard backend (lihat PLAN_UI_UX.md §5.7) --
    tidak ada risiko keduanya diam-diam berbeda perilaku.

    `on_progress(dict)` dipanggil setelah setiap pertanyaan selesai (sukses
    maupun gagal) -- dipakai backend untuk push event SSE progress ke UI.
    `check_cancel()` -> bool, dicek di awal tiap iterasi -- dipakai backend
    untuk POST /api/runs/{id}/cancel (§5.4). Kedua parameter opsional supaya
    CLI (main()) tidak perlu berubah perilaku.
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

            messages = build_base_messages(row["Title"], row["Body"], row["Tags"])
            try:
                llm_answer = call_llm_fn(llm_client, messages, model)
            except Exception as e:
                log(f"      [{i+1}/{total}] Id={qid} [FAIL] API error: {e}")
                if on_progress:
                    on_progress({"question_id": qid, "index": i, "total": total,
                                 "status": "failed", "error": str(e)})
                continue

            similarity = compute_similarity(embed_model, llm_answer, row["AcceptedAnswerBody"])

            record = {
                "question_id": qid,
                "title": row["Title"],
                "tags": row["Tags"],
                "n_tokens": int(row["n_tokens"]),
                "view_count": row.get("ViewCount"),
                "question_score": row.get("Score"),
                "accepted_answer_id": int(row["AcceptedAnswerId"]),
                "ground_truth_answer": row["AcceptedAnswerBody"],
                "prompt_messages": messages,
                "llm_answer": llm_answer,
                "llm_model": model,
                "cosine_similarity": similarity,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            results.append(record)
            log(f"      [{i+1}/{total}] Id={qid} similarity={similarity:.3f}")
            if on_progress:
                on_progress({"question_id": qid, "index": i, "total": total,
                             "status": "done", "similarity": similarity, "record": record})
            time.sleep(0.3)  # jaga-jaga rate limit
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def build_output_path(output_dir: str, provider: str, model: str, n_sample: int, seed: int) -> Path:
    """Nama file JSONL otomatis dari provider+model+n_sample+seed -- format:
    condition_a_{provider}_{safe_model}_n{n_sample}_seed{seed}.jsonl.

    PENTING: n_sample & seed WAJIB ada di nama file (bukan cuma provider+
    model). Kalau tidak, dua run dengan ukuran sample BERBEDA (mis. pilot
    n=5 lalu n=30) akan menulis ke file yang SAMA, dan fitur resume di
    bawah akan mengira sample lama sebagai 'sudah diproses' lalu
    menambahkan sample baru ke situ -- hasilnya file JSONL berisi
    campuran dua sample yang tidak koheren (pernah kejadian: n=5 + n=30
    tercampur jadi 35 baris yang tidak merepresentasikan sample manapun
    secara valid)."""
    safe_model = model.replace("/", "-").replace(":", "-").replace(".", "-")
    return Path(output_dir) / f"condition_a_{provider}_{safe_model}_n{n_sample}_seed{seed}.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Semua argumen defaultnya diambil dari .env -- CLI hanya untuk override
    # sesaat (mis. pilot run cepat) tanpa perlu edit file .env.
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"),
                         help="Default dari QUESTIONS_PARQUET di .env")
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"),
                         help="Default dari ANSWERS_PARQUET di .env")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "results"),
                         help="Folder output (default dari OUTPUT_DIR di .env). Nama file "
                              "JSONL dibuat OTOMATIS dari provider+model yang dipakai, "
                              "format: condition_a_{provider}_{safe_model}.jsonl")
    parser.add_argument("--output", default=None,
                         help="Override manual path output lengkap (opsional). Kalau diisi, "
                              "auto-naming dari --output-dir/provider/model DIABAIKAN.")
    parser.add_argument("--n-sample", type=int, default=int(os.getenv("N_SAMPLE", 384)),
                         help="Default dari N_SAMPLE di .env")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)),
                         help="Default dari SEED di .env")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local", "ollama"],
                         help="Default dari LLM_PROVIDER di .env. 'ollama' = dev/testing "
                              "cepat pakai model kecil lokal, BUKAN untuk evaluasi final.")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                         help="Default dari LLM_MODEL di .env")
    parser.add_argument("--oversample-pool", type=int, default=int(os.getenv("OVERSAMPLE_POOL", 0)),
                         help="Ukuran pool acak awal sebelum filter token (default: "
                              "otomatis 3x n_sample kalau tidak diset)")
    args = parser.parse_args()

    global log
    logger, log_path = setup_logging(args.provider, args.model)
    log = logger.info
    run_started_at = datetime.now(timezone.utc)
    log(f"[logging] Log detail run ini -> {log_path}")

    # PENTING utk NESTED SAMPLING (lihat sample_questions()): pool ini SENGAJA
    # TIDAK bergantung pada args.n_sample lagi (dulu: args.n_sample * 3).
    # Kalau bergantung, tiap n_sample beda akan menarik reservoir pool
    # DuckDB berukuran BEDA, sehingga sample yg dihasilkan TIDAK NESTED
    # (n=30 bukan subset dari n=100) walau seed identik. Pool sekarang FIXED
    # merujuk ke MAX_PLANNED_N_SAMPLE (rencana evaluasi resmi terbesar),
    # supaya SEMUA pilot run (n=10, 30, 50, 100, ..., 384) berbagi populasi
    # kandidat & urutan permutasi yang SAMA PERSIS -- sample kecil selalu
    # PREFIX dari sample besar, TIDAK ADA biaya API yang terbuang saat naik
    # ukuran sample bertahap. --oversample-pool manual tetap bisa override.
    MAX_PLANNED_N_SAMPLE = 384
    oversample_pool = args.oversample_pool if args.oversample_pool > 0 else MAX_PLANNED_N_SAMPLE * 4
    if args.n_sample > MAX_PLANNED_N_SAMPLE:
        log(f"      [WARN] n_sample ({args.n_sample}) > MAX_PLANNED_N_SAMPLE "
            f"({MAX_PLANNED_N_SAMPLE}) -- pool otomatis TIDAK cukup, set "
            f"--oversample-pool manual (mis. {args.n_sample * 4}).")

    # Auto-naming: nama file output dibuat dari provider+model yang benar-benar
    # dipakai run ini, kecuali user eksplisit override lewat --output.
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

    llm_client, call_llm_fn = get_llm_client(args.provider, args.model, log=log)

    con = duckdb.connect()

    candidates = get_candidate_questions(con, args.questions_parquet, args.answers_parquet,
                                          oversample_pool, args.seed)
    candidates = filter_by_token_limit(candidates)
    sample_df = sample_questions(candidates, args.n_sample, args.seed)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = get_accepted_answers(con, args.answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")

    missing_answer = sample_df["AcceptedAnswerBody"].isna().sum()
    if missing_answer:
        # Seharusnya jarang/tidak pernah terjadi lagi sejak get_candidate_questions
        # sudah semi-join ke Answers.Id sebelum sampling -- kalau ini muncul lagi
        # dalam jumlah besar, kemungkinan ada duplikat Id ganjil di Answers parquet
        # (dedup rn=1 di get_accepted_answers gagal match), bukan lagi soal SORD
        # yang independen filter Questions/Answers.
        log(f"      [WARN] {missing_answer} pertanyaan tidak ketemu accepted answer-nya "
              f"di Answers parquet -- SEHARUSNYA jarang terjadi setelah fix semi-join "
              f"di Step 1; kalau jumlahnya besar, cek duplikat Id di Answers parquet.")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

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

    log(f"[5/6] Load model embedding untuk similarity scoring...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    log(f"[6/6] Prompting {args.model} untuk {len(sample_df)} pertanyaan "
          f"(1 panggilan per pertanyaan, konfigurasi default)...")

    process_sample(sample_df, llm_client, call_llm_fn, embed_model, args.model, output_path, already_done)

    # --- Ringkasan akhir ---
    if not output_path.exists() or output_path.stat().st_size == 0:
        log("\n[ERROR] Tidak ada hasil tersimpan sama sekali -- kemungkinan semua "
              "panggilan API gagal (cek pesan [FAIL] di atas, sering karena API key "
              "tidak valid/kehabisan credit/rate limit). Tidak ada ringkasan untuk ditampilkan.")
        history_path = append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "status": "no_results",
            "provider": args.provider,
            "model": args.model,
            "n_sample_target": args.n_sample,
            "seed": args.seed,
            "oversample_pool": oversample_pool,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        return

    results_df = pd.read_json(output_path, lines=True)
    if "cosine_similarity" not in results_df.columns or len(results_df) == 0:
        log("\n[ERROR] File hasil ada tapi tidak berisi record valid -- cek pesan "
              "[FAIL] di atas untuk penyebabnya.")
        history_path = append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "status": "no_valid_records",
            "provider": args.provider,
            "model": args.model,
            "n_sample_target": args.n_sample,
            "seed": args.seed,
            "oversample_pool": oversample_pool,
            "output_path": str(output_path),
            "log_path": str(log_path),
            "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
        })
        log(f"[logging] Ringkasan run (gagal) dicatat -> {history_path}")
        return

    log("\n" + "=" * 70)
    log("RINGKASAN HASIL KONDISI A (Replikasi Baseline)")
    log("=" * 70)
    log(f"Total pertanyaan diproses : {len(results_df)}")
    log(f"Cosine similarity rata-rata: {results_df['cosine_similarity'].mean():.4f}")
    log(f"Cosine similarity median   : {results_df['cosine_similarity'].median():.4f}")
    log(f"% similarity > 0.5         : {(results_df['cosine_similarity'] > 0.5).mean()*100:.1f}% "
          f"(pembanding paper asli: ~85%)")
    log(f"\nHasil lengkap tersimpan -> {output_path}")

    history_path = append_run_history({
        "run_started_at": run_started_at.isoformat(),
        "status": "success",
        "provider": args.provider,
        "model": args.model,
        "n_sample_target": args.n_sample,
        "n_processed": len(results_df),
        "seed": args.seed,
        "oversample_pool": oversample_pool,
        "cosine_similarity_mean": round(float(results_df["cosine_similarity"].mean()), 4),
        "cosine_similarity_median": round(float(results_df["cosine_similarity"].median()), 4),
        "pct_similarity_above_0_5": round(float((results_df["cosine_similarity"] > 0.5).mean() * 100), 1),
        "output_path": str(output_path),
        "log_path": str(log_path),
        "duration_sec": round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1),
    })
    log(f"[logging] Ringkasan run dicatat -> {history_path}")


if __name__ == "__main__":
    sys.exit(main())