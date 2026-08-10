"""
stratified_sampling.py
=======================
Fase 0.4 — Sampling strategi benchmark uji dari dataset SORD.

Mereplikasi pola stratifikasi Kabir dkk. (2024) dan Da Silva, Samhi, & Khomh
(2025): pertanyaan distratifikasi berdasarkan (1) popularitas (ViewCount),
(2) tipe pertanyaan (conceptual / how-to / debugging), dan opsional
(3) recency (sebelum/sesudah rilis ChatGPT, 30 Nov 2022) — relevan jika
ingin membandingkan dengan hasil baseline Da Silva dkk.

INPUT YANG DIHARAPKAN
----------------------
File CSV Questions dari SORD (mis. QuestionsBody_Contain.csv atau hasil
union dengan QuestionsBody_LikeMinusContain.csv), dengan kolom minimal:
    Id, Title, Body, Tags, Score, ViewCount, AcceptedAnswerId, CreationDate

CARA PAKAI
----------
    python stratified_sampling.py \
        --input data/sord/questions/QuestionsBody_Contain.csv \
        --extra data/sord/questions/QuestionsBody_LikeMinusContain.csv \
        --n-total 200 \
        --output results/benchmark_sample.csv \
        --seed 42

Jika hanya punya satu file, --extra bisa dihilangkan.

CATATAN PENTING
----------------
- Klasifikasi tipe pertanyaan di sini pakai HEURISTIK KEYWORD (bukan SVM
  terlatih seperti Kabir dkk.). Ini pilihan sadar untuk mempercepat fase
  awal — WAJIB divalidasi manual sebelum dipakai sebagai benchmark final
  (lihat kolom `needs_manual_check` di output, dan fungsi
  `validate_sample_manually` di bagian bawah file).
- Popularitas dihitung dari kuantil ViewCount pada data SETELAH filtering
  (hanya pertanyaan dengan AcceptedAnswerId terisi), bukan dari seluruh
  dataset mentah SORD.
- Skrip ini idempotent: dengan --seed yang sama, hasil sampling selalu
  sama persis (penting untuk reprodusibilitas tesis).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. Loading & deduplication
# ---------------------------------------------------------------------

def load_questions(input_path: str, extra_path: str | None = None) -> pd.DataFrame:
    """Load file Questions SORD, gabungkan dengan varian _LikeMinusContain
    jika disediakan, lalu deduplikasi berdasarkan Id."""
    df = pd.read_csv(input_path, low_memory=False)
    if extra_path:
        df_extra = pd.read_csv(extra_path, low_memory=False)
        df = pd.concat([df, df_extra], ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset="Id", keep="first")
    print(f"[load] {before} baris dimuat, {len(df)} unik setelah dedup by Id")

    required_cols = {"Id", "Title", "Body", "ViewCount", "AcceptedAnswerId"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib hilang dari file input: {missing}. "
            f"Cek nama kolom SORD Anda (lihat Tabel 1 pada paper SORD)."
        )
    return df


def filter_valid_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Hanya pertanyaan yang punya accepted answer yang dipakai sebagai
    kandidat benchmark, karena accepted answer jadi ground truth."""
    before = len(df)
    df = df[df["AcceptedAnswerId"].notna()].copy()
    df = df[df["Body"].notna() & df["Title"].notna()]
    print(f"[filter] {before} -> {len(df)} pertanyaan dengan accepted answer")
    return df


# ---------------------------------------------------------------------
# 2. Stratifikasi: popularitas
# ---------------------------------------------------------------------

def assign_popularity_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Bagi jadi 3 tier berdasarkan kuantil ViewCount:
    Highly Popular (top 10%), Average Popular (tengah), Unpopular (bottom 10%).
    Mengikuti pola Kabir dkk. (2024)."""
    q90 = df["ViewCount"].quantile(0.90)
    q10 = df["ViewCount"].quantile(0.10)

    def tier(v):
        if v >= q90:
            return "Highly Popular"
        elif v <= q10:
            return "Unpopular"
        else:
            return "Average Popular"

    df["popularity_tier"] = df["ViewCount"].apply(tier)
    print(f"[popularity] batas: q10={q10:.0f} views, q90={q90:.0f} views")
    print(df["popularity_tier"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------
# 3. Stratifikasi: tipe pertanyaan (heuristik, perlu validasi manual)
# ---------------------------------------------------------------------

DEBUG_PATTERNS = re.compile(
    r"\b(error|exception|bug|fix|fails?|failing|crash|traceback|"
    r"not working|doesn't work|why (is|does|isn't)|issue|problem)\b",
    re.IGNORECASE,
)
HOWTO_PATTERNS = re.compile(
    r"^\s*(how (do|can|to|should)|what('s| is) the best way to)",
    re.IGNORECASE,
)


def classify_question_type(title: str, body: str) -> str:
    """Heuristik sederhana berbasis keyword untuk klasifikasi awal.
    Prioritas: Debugging > How-to > Conceptual (default).

    PENTING: ini bukan pengganti classifier terlatih (mis. SVM di Kabir
    dkk. dengan akurasi ~78%). Gunakan hanya untuk stratifikasi awal, lalu
    validasi manual pada sample akhir sebelum dipakai sebagai benchmark.
    """
    text = f"{title} {body}"
    if DEBUG_PATTERNS.search(text):
        return "Debugging"
    if HOWTO_PATTERNS.search(title):
        return "How-to"
    return "Conceptual"


def assign_question_type(df: pd.DataFrame) -> pd.DataFrame:
    df["question_type"] = df.apply(
        lambda r: classify_question_type(str(r["Title"]), str(r["Body"])), axis=1
    )
    print("[type] distribusi tipe pertanyaan (heuristik):")
    print(df["question_type"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------
# 4. Stratifikasi opsional: recency (relatif rilis ChatGPT)
# ---------------------------------------------------------------------

CHATGPT_RELEASE = pd.Timestamp("2022-11-30")


def assign_recency(df: pd.DataFrame) -> pd.DataFrame:
    if "CreationDate" not in df.columns:
        df["recency"] = "Unknown"
        return df
    dates = pd.to_datetime(df["CreationDate"], errors="coerce")
    df["recency"] = np.where(dates < CHATGPT_RELEASE, "Old", "New")
    print("[recency] distribusi:")
    print(df["recency"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------
# 5. Stratified sampling
# ---------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    n_total: int,
    strata_cols: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Sampling proporsional-terbatas: tiap kombinasi strata mendapat
    alokasi kuota rata (bukan proporsional terhadap ukuran populasi),
    supaya stratum kecil (mis. Unpopular x Debugging) tetap terwakili
    memadai untuk analisis, sesuai semangat desain Kabir dkk. Tabel 1.
    """
    groups = df.groupby(strata_cols, dropna=False)
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("Tidak ada strata terbentuk — cek data input.")

    quota_per_group = max(1, n_total // n_groups)
    print(f"[sample] {n_groups} kombinasi strata, target ~{quota_per_group} per strata")

    sampled_parts = []
    shortfall = 0
    for keys, g in groups:
        take = min(len(g), quota_per_group)
        shortfall += max(0, quota_per_group - len(g))
        sampled_parts.append(g.sample(n=take, random_state=seed))

    sample_df = pd.concat(sampled_parts, ignore_index=True)

    # Isi kekurangan kuota (stratum yang under-populated) dari sisa pool
    if len(sample_df) < n_total:
        remaining_pool = df.drop(sample_df.index, errors="ignore")
        need = n_total - len(sample_df)
        if len(remaining_pool) > 0:
            extra = remaining_pool.sample(
                n=min(need, len(remaining_pool)), random_state=seed
            )
            sample_df = pd.concat([sample_df, extra], ignore_index=True)

    print(f"[sample] total terkumpul: {len(sample_df)} (target {n_total})")
    if shortfall:
        print(
            f"[sample][warn] {shortfall} kuota tidak terpenuhi di beberapa "
            f"strata kecil — ditambal dari sisa pool acak."
        )
    return sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------
# 6. Validasi manual (placeholder wajib diisi sebelum benchmark final)
# ---------------------------------------------------------------------

def add_manual_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom kosong untuk proses validasi manual tipe pertanyaan,
    mengikuti praktik Kabir dkk. yang memvalidasi manual hasil classifier
    otomatis sebelum dipakai. WAJIB diisi manual (Anda + idealnya 1 rater
    kedua untuk Cohen's kappa) sebelum sample ini dipakai sebagai benchmark
    final di Fase 1 dst."""
    df["needs_manual_check"] = True
    df["manual_type_label"] = ""  # diisi manual: Conceptual/How-to/Debugging
    df["manual_notes"] = ""
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path ke file Questions CSV utama")
    parser.add_argument("--extra", default=None, help="Path ke file Questions varian tambahan (opsional)")
    parser.add_argument("--n-total", type=int, default=200, help="Target jumlah sample benchmark")
    parser.add_argument("--output", required=True, help="Path output CSV hasil sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk reprodusibilitas")
    parser.add_argument(
        "--strata",
        nargs="+",
        default=["popularity_tier", "question_type"],
        help="Kolom yang dipakai sebagai strata (default: popularity_tier question_type)",
    )
    args = parser.parse_args()

    df = load_questions(args.input, args.extra)
    df = filter_valid_questions(df)
    df = assign_popularity_tier(df)
    df = assign_question_type(df)
    df = assign_recency(df)

    sample_df = stratified_sample(df, args.n_total, args.strata, seed=args.seed)
    sample_df = add_manual_validation_columns(sample_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out_path, index=False)

    print(f"\n[done] Sample tersimpan di: {out_path}")
    print("\nRingkasan strata pada sample akhir:")
    print(sample_df.groupby(args.strata).size().to_string())
    print(
        "\n[next step] Buka file output, isi kolom `manual_type_label` untuk "
        "tiap baris (validasi manual tipe pertanyaan), lalu bandingkan "
        "dengan `question_type` (hasil heuristik) untuk cek akurasi awal."
    )


if __name__ == "__main__":
    sys.exit(main())
