"""
analyze_kg_feasibility.py
===========================
Analisis struktural terhadap subset SORD yang Question-nya PUNYA accepted
answer yang match di Answers parquet (~271K pairs) -- untuk menilai apakah
subset ini cukup layak jadi basis Knowledge Graph (trust-weighted, entity
Tag/Error/Solution/CodePattern) sebelum keputusan final soal dataset.

YANG DIHITUNG
-------------
1. Ukuran subset matched vs total populasi
2. Distribusi tag (top-N) -- proxy untuk keragaman domain/topik dalam KG
3. Rata-rata jumlah Comment per Question & per Answer (untuk edge density
   -- Comment biasanya jadi salah satu sumber relasi tambahan di GraphRAG)
4. Distribusi Score (votes) pada matched subset -- untuk cek apakah trust
   signal (upvotes) masih bervariasi & tidak terlalu skewed ke 0
5. Estimasi rata-rata jumlah node per "cluster" pertanyaan (Question ->
   Answer -> Comment) sebagai proxy kepadatan graf

CARA PAKAI
----------
    python analyze_kg_feasibility.py

Jalankan dari folder 02_baseline_replication/ (path .env sama).
"""

import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

QUESTIONS_PARQUET = os.getenv("QUESTIONS_PARQUET", "../00_datasource/merged/questions_raw_union.parquet")
ANSWERS_PARQUET = os.getenv("ANSWERS_PARQUET", "../00_datasource/merged/answers_raw_union.parquet")
COMMENTS_PARQUET = os.getenv("COMMENTS_PARQUET", "../00_datasource/merged/comments_raw_union.parquet")


def main():
    print("=" * 70)
    print("ANALISIS KELAYAKAN KG DARI SUBSET SORD YANG MATCH")
    print("=" * 70)

    con = duckdb.connect()

    # --- 1. Bangun subset matched (Question dedup + AcceptedAnswerId match) ---
    print("\n[1/5] Membangun subset matched (Question x Accepted Answer)...")
    con.execute(f"""
        CREATE TEMP TABLE matched AS
        SELECT q.Id AS QuestionId, q.Tags, q.Score AS QuestionScore,
               q.ViewCount, q.AcceptedAnswerId,
               a.Id AS AnswerId, a.Score AS AnswerScore
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{QUESTIONS_PARQUET}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
        ) q
        JOIN (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{ANSWERS_PARQUET}')
        ) a
        ON q.AcceptedAnswerId = a.Id AND a.rn = 1
        WHERE q.rn = 1
    """)
    total_matched = con.execute("SELECT COUNT(*) FROM matched").fetchone()[0]
    print(f"      Total pasangan Question-AcceptedAnswer matched: {total_matched:,}")

    # --- 2. Distribusi tag (top 20) ---
    print("\n[2/5] Distribusi tag (top 20) pada subset matched...")
    tags_df = con.execute("SELECT Tags FROM matched WHERE Tags IS NOT NULL").df()
    tag_counts = {}
    for tags in tags_df["Tags"]:
        for t in str(tags).replace("<", " ").replace(">", " ").split():
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
    print(f"      Total tag unik dalam subset: {len(tag_counts):,}")
    for tag, cnt in top_tags:
        print(f"        {tag:<20s} {cnt:,}")

    # --- 3. Rata-rata Comment per Question & per Answer ---
    print("\n[3/5] Rata-rata Comment per Question & per Answer (subset matched)...")
    try:
        comment_q = con.execute(f"""
            SELECT AVG(cnt) FROM (
                SELECT m.QuestionId, COUNT(c.Id) AS cnt
                FROM matched m
                LEFT JOIN read_parquet('{COMMENTS_PARQUET}') c ON c.PostId = m.QuestionId
                GROUP BY m.QuestionId
            )
        """).fetchone()[0]
        comment_a = con.execute(f"""
            SELECT AVG(cnt) FROM (
                SELECT m.AnswerId, COUNT(c.Id) AS cnt
                FROM matched m
                LEFT JOIN read_parquet('{COMMENTS_PARQUET}') c ON c.PostId = m.AnswerId
                GROUP BY m.AnswerId
            )
        """).fetchone()[0]
        print(f"      Rata-rata Comment per Question : {comment_q:.2f}")
        print(f"      Rata-rata Comment per Answer    : {comment_a:.2f}")
    except Exception as e:
        print(f"      [SKIP] Tidak bisa hitung comment (cek kolom PostId ada di comments parquet): {e}")

    # --- 4. Distribusi Score (trust signal) ---
    print("\n[4/5] Distribusi Score/votes pada subset matched...")
    score_df = con.execute("SELECT QuestionScore, AnswerScore FROM matched").df()
    print("      Question Score  :", score_df["QuestionScore"].describe().to_dict())
    print("      Answer Score    :", score_df["AnswerScore"].describe().to_dict())
    zero_q = (score_df["QuestionScore"] <= 0).mean() * 100
    zero_a = (score_df["AnswerScore"] <= 0).mean() * 100
    print(f"      % Question dengan Score <= 0 : {zero_q:.1f}%")
    print(f"      % Answer dengan Score <= 0   : {zero_a:.1f}%")

    # --- 5. Kesimpulan otomatis sederhana ---
    print("\n" + "=" * 70)
    print("RINGKASAN UNTUK KEPUTUSAN DATASET")
    print("=" * 70)
    print(f"- Ukuran subset matched      : {total_matched:,} pairs "
          f"({'CUKUP BESAR' if total_matched > 50000 else 'KECIL, perlu waspada'} untuk KG)")
    print(f"- Keragaman tag              : {len(tag_counts):,} tag unik "
          f"({'CUKUP BERAGAM' if len(tag_counts) > 500 else 'TERBATAS, domain mungkin sempit'})")
    print(f"- Trust signal (votes)       : {'BERVARIASI' if zero_q < 50 else 'BANYAK BERNILAI 0/NEGATIF, trust-weighting mungkin kurang efektif'}")
    print("\nCatatan: angka ini indikatif, bukan keputusan final -- pertimbangkan "
          "juga alignment scope 'recommendation-oriented' dengan RQ/judul thesis Anda.")


if __name__ == "__main__":
    main()