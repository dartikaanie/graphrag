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
4. Struktur prompt 3 lapis: persona "ahli" dari tags -> title sebagai
   instruksi inti -> body sebagai konteks tambahan. (Struktur ini
   diparafrasekan dari deskripsi metodologi paper, BUKAN kutipan verbatim
   dari Figure 2 paper -- redaksi kalimat prompt di bawah adalah karya
   sendiri mengikuti pola yang dideskripsikan.)
5. Konfigurasi model: DEFAULT (tidak override temperature/parameter lain)
   -- meniru pola pemakaian LLM oleh user biasa, sesuai keputusan
   metodologis paper asli.
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
    # Path data (relatif terhadap folder 02_baseline_replication/, sejajar
    # dengan 00_datasource/ di root project)
    QUESTIONS_PARQUET=../00_datasource/merged/questions_raw_union.parquet
    ANSWERS_PARQUET=../00_datasource/merged/answers_raw_union.parquet
    OUTPUT_PATH=results/condition_a_gpt4o.jsonl

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
import os
import sys
import time
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

TOKEN_LIMIT = 2048  # sesuai batasan paper asli


# ---------------------------------------------------------------------
# 1. Ambil kandidat pertanyaan dari Parquet (dedup + filter accepted answer)
# ---------------------------------------------------------------------

def get_candidate_questions(con: duckdb.DuckDBPyConnection, questions_parquet: str,
                             oversample_pool: int, seed: int) -> pd.DataFrame:
    """Dedup by Id (pilih 1 baris representatif) + filter hanya yang punya
    accepted answer, LALU langsung ambil oversample pool kecil secara acak
    (bukan seluruh populasi) menggunakan `USING SAMPLE` DuckDB.

    PENTING (optimasi): kita TIDAK perlu materialize/hitung token untuk
    seluruh ~1.27 juta kandidat kalau target akhir cuma 384 pertanyaan.
    DuckDB melakukan sampling di level SQL sebelum data ditransfer ke
    pandas, jauh lebih cepat daripada sample belakangan pakai df.sample()
    setelah semua baris (termasuk kolom Body yang berat) sudah termuat ke
    memori Python. oversample_pool diambil lebih besar dari n_sample akhir
    untuk mengantisipasi baris yang nanti tereliminasi oleh filter token
    limit (Step 2)."""
    print(f"[1/6] Mengambil oversample pool ({oversample_pool} kandidat, "
          f"bukan seluruh populasi) dari kandidat accepted-answer...")
    query = f"""
        SELECT Id, Title, Body, Tags, AcceptedAnswerId, ViewCount, Score
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
        )
        WHERE rn = 1
        USING SAMPLE {oversample_pool} ROWS (reservoir, {seed})
    """
    df = con.execute(query).df()
    print(f"      Oversample pool diambil: {len(df):,} kandidat "
          f"(hanya subset ini yang akan dihitung token-nya di Step 2)")
    return df


# ---------------------------------------------------------------------
# 2. Filter token limit (persis kriteria eksklusi paper asli)
# ---------------------------------------------------------------------

def filter_by_token_limit(df: pd.DataFrame, limit: int = TOKEN_LIMIT) -> pd.DataFrame:
    import tiktoken

    print(f"[2/6] Menghitung token (title+body+tags) & filter <= {limit} token "
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
    print(f"      {before:,} kandidat -> {len(df_filtered):,} lolos filter token "
          f"({excluded:,} dikecualikan karena > {limit} token, "
          f"{excluded/before*100:.1f}%)")
    return df_filtered


# ---------------------------------------------------------------------
# 3. Sampling acak (384, seed untuk reprodusibilitas)
# ---------------------------------------------------------------------

def sample_questions(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    print(f"[3/6] Sampling acak {n} pertanyaan (seed={seed}, "
          f"justifikasi statistik: 95% CI, 5% margin of error -- sama seperti paper asli)...")
    if len(df) < n:
        print(f"      [WARN] Populasi hasil filter ({len(df)}) < target sample ({n}), "
              f"pakai semua yang tersedia. Kalau target 384 tidak tercapai, jalankan "
              f"ulang dengan --oversample-pool lebih besar (mis. {n*5}).")
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------
# 4. Ambil accepted answer dari Answers parquet
# ---------------------------------------------------------------------

def get_accepted_answers(con: duckdb.DuckDBPyConnection, answers_parquet: str,
                          accepted_answer_ids: list) -> pd.DataFrame:
    print("[4/6] Mengambil teks accepted answer dari Answers parquet...")
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
    print(f"      Ditemukan {len(df):,} accepted answer")
    return df


# ---------------------------------------------------------------------
# 5. Prompt LLM (struktur diparafrasekan dari deskripsi metodologi paper)
# ---------------------------------------------------------------------

def build_prompt(title: str, body: str, tags: str) -> str:
    """Struktur 3 lapis mengikuti deskripsi metodologi paper asli:
    (1) persona ahli dibentuk dari tags, (2) title sebagai instruksi inti
    untuk menjelaskan solusi, (3) body sebagai konteks tambahan.
    Redaksi kalimat adalah tulisan sendiri, bukan kutipan dari paper."""
    tag_list = tags.replace("<", "").replace(">", " ").strip() if tags else "software development"
    prompt = (
        f"You are an expert with extensive knowledge in {tag_list}. "
        f"A developer needs help with the following problem. "
        f"Please explain how to fix or address it.\n\n"
        f"Title: {title}\n\n"
        f"Description: {body}"
    )
    return prompt


def call_llm_openai(client, prompt: str, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_llm_anthropic(client, prompt: str, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_llm_local(pipe, prompt: str, model: str) -> str:
    """Untuk model lokal via Hugging Face (mis. LLaMA), pola yang sama
    dipakai paper baseline asli (load Llama-2-7b-chat-hf secara lokal)."""
    output = pipe(prompt, max_new_tokens=512, do_sample=False)
    return output[0]["generated_text"][len(prompt):].strip()


def get_llm_client(provider: str, model: str):
    """Factory: siapkan client sesuai provider. Menambah provider baru
    di masa depan cukup tambah 1 cabang di sini + 1 fungsi call_llm_*,
    tidak perlu ubah bagian lain skrip."""
    if provider == "openai":
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            print("[ERROR] OPENAI_API_KEY tidak ditemukan di .env")
            sys.exit(1)
        return OpenAI(), call_llm_openai

    if provider == "anthropic":
        import anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("[ERROR] ANTHROPIC_API_KEY tidak ditemukan di .env "
                  "(catatan: ini API key dari console.anthropic.com, "
                  "BEDA dari langganan Claude.ai)")
            sys.exit(1)
        return anthropic.Anthropic(), call_llm_anthropic

    if provider == "local":
        from transformers import pipeline
        print(f"      Loading model lokal '{model}' (bisa lama untuk pertama kali)...")
        pipe = pipeline("text-generation", model=model, device_map="auto")
        return pipe, call_llm_local

    raise ValueError(f"Provider '{provider}' tidak dikenal. Pilihan: openai, anthropic, local")


# ---------------------------------------------------------------------
# 6. Similarity scoring
# ---------------------------------------------------------------------

def compute_similarity(embed_model, text_a: str, text_b: str) -> float:
    from sentence_transformers import util
    emb_a = embed_model.encode(str(text_a), convert_to_tensor=True)
    emb_b = embed_model.encode(str(text_b), convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Semua argumen defaultnya diambil dari .env -- CLI hanya untuk override
    # sesaat (mis. pilot run cepat) tanpa perlu edit file .env.
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"),
                         help="Default dari QUESTIONS_PARQUET di .env")
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"),
                         help="Default dari ANSWERS_PARQUET di .env")
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", "results/condition_a_results.jsonl"),
                         help="Default dari OUTPUT_PATH di .env")
    parser.add_argument("--n-sample", type=int, default=int(os.getenv("N_SAMPLE", 384)),
                         help="Default dari N_SAMPLE di .env")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)),
                         help="Default dari SEED di .env")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"),
                         choices=["openai", "anthropic", "local"],
                         help="Default dari LLM_PROVIDER di .env")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                         help="Default dari LLM_MODEL di .env")
    parser.add_argument("--oversample-pool", type=int, default=int(os.getenv("OVERSAMPLE_POOL", 0)),
                         help="Ukuran pool acak awal sebelum filter token (default: "
                              "otomatis 3x n_sample kalau tidak diset)")
    args = parser.parse_args()

    oversample_pool = args.oversample_pool if args.oversample_pool > 0 else args.n_sample * 3

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET dan ANSWERS_PARQUET harus diset di .env "
              "(atau lewat --questions-parquet/--answers-parquet)")
        sys.exit(1)

    print(f"[config] questions_parquet = {args.questions_parquet}")
    print(f"[config] answers_parquet   = {args.answers_parquet}")
    print(f"[config] output            = {args.output}")
    print(f"[config] n_sample={args.n_sample} seed={args.seed} "
          f"provider={args.provider} model={args.model}")

    llm_client, call_llm_fn = get_llm_client(args.provider, args.model)

    con = duckdb.connect()

    candidates = get_candidate_questions(con, args.questions_parquet, oversample_pool, args.seed)
    candidates = filter_by_token_limit(candidates)
    sample_df = sample_questions(candidates, args.n_sample, args.seed)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = get_accepted_answers(con, args.answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")

    missing_answer = sample_df["AcceptedAnswerBody"].isna().sum()
    if missing_answer:
        print(f"      [WARN] {missing_answer} pertanyaan tidak ketemu accepted answer-nya "
              f"di Answers parquet (kemungkinan AcceptedAnswerId tidak konsisten di SORD) "
              f"-- akan dilewati saat prompting.")
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
        print(f"      [resume] {len(already_done)} pertanyaan sudah diproses sebelumnya, akan dilewati")

    print(f"[5/6] Load model embedding untuk similarity scoring...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"[6/6] Prompting {args.model} untuk {len(sample_df)} pertanyaan "
          f"(1 panggilan per pertanyaan, konfigurasi default)...")

    with open(output_path, "a") as f_out:
        for i, row in sample_df.iterrows():
            qid = int(row["Id"])
            if qid in already_done:
                continue

            prompt = build_prompt(row["Title"], row["Body"], row["Tags"])
            try:
                llm_answer = call_llm_fn(llm_client, prompt, args.model)
            except Exception as e:
                print(f"      [{i+1}/{len(sample_df)}] Id={qid} [FAIL] API error: {e}")
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
                "llm_answer": llm_answer,
                "llm_model": args.model,
                "cosine_similarity": similarity,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            print(f"      [{i+1}/{len(sample_df)}] Id={qid} similarity={similarity:.3f}")
            time.sleep(0.3)  # jaga-jaga rate limit

    # --- Ringkasan akhir ---
    if not output_path.exists() or output_path.stat().st_size == 0:
        print("\n[ERROR] Tidak ada hasil tersimpan sama sekali -- kemungkinan semua "
              "panggilan API gagal (cek pesan [FAIL] di atas, sering karena API key "
              "tidak valid/kehabisan credit/rate limit). Tidak ada ringkasan untuk ditampilkan.")
        return

    results_df = pd.read_json(output_path, lines=True)
    if "cosine_similarity" not in results_df.columns or len(results_df) == 0:
        print("\n[ERROR] File hasil ada tapi tidak berisi record valid -- cek pesan "
              "[FAIL] di atas untuk penyebabnya.")
        return

    print("\n" + "=" * 70)
    print("RINGKASAN HASIL KONDISI A (Replikasi Baseline)")
    print("=" * 70)
    print(f"Total pertanyaan diproses : {len(results_df)}")
    print(f"Cosine similarity rata-rata: {results_df['cosine_similarity'].mean():.4f}")
    print(f"Cosine similarity median   : {results_df['cosine_similarity'].median():.4f}")
    print(f"% similarity > 0.5         : {(results_df['cosine_similarity'] > 0.5).mean()*100:.1f}% "
          f"(pembanding paper asli: ~85%)")
    print(f"\nHasil lengkap tersimpan -> {output_path}")


if __name__ == "__main__":
    sys.exit(main())