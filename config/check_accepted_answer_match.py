"""
check_accepted_answer_match.py
================================
Cek seberapa besar rasio pertanyaan yang punya AcceptedAnswerId tapi
jawabannya TIDAK ketemu di Answers parquet -- untuk tahu apakah masalah
"3 dari 5 tidak ketemu" di pilot run kecil tadi cuma kebetulan sample kecil,
atau memang masalah sistemik di seluruh populasi SORD hasil merge.

CARA PAKAI
----------
    python check_accepted_answer_match.py

Path parquet diambil dari .env yang sama dengan a_baseline_replecation.py
(QUESTIONS_PARQUET, ANSWERS_PARQUET) -- jalankan dari folder
llm/a_pure_llm/ supaya path relatif konsisten.
"""

import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

QUESTIONS_PARQUET = os.getenv("QUESTIONS_PARQUET", "../00_datasource/merged/questions_raw_union.parquet")
ANSWERS_PARQUET = os.getenv("ANSWERS_PARQUET", "../00_datasource/merged/answers_raw_union.parquet")


def main():
    print("=" * 70)
    print("CEK RASIO MATCH: AcceptedAnswerId (Questions) <-> Id (Answers)")
    print("=" * 70)
    print(f"Questions parquet : {QUESTIONS_PARQUET}")
    print(f"Answers parquet   : {ANSWERS_PARQUET}\n")

    con = duckdb.connect()

    # Total pertanyaan unik (setelah dedup by Id, sama seperti logic di
    # a_baseline_replecation.py) yang punya AcceptedAnswerId terisi.
    total_q = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT Id, AcceptedAnswerId,
                   ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{QUESTIONS_PARQUET}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
        ) WHERE rn = 1
    """).fetchone()[0]

    # Berapa dari AcceptedAnswerId itu yang benar-benar ketemu Id-nya
    # di Answers parquet (pakai logic dedup yang sama juga di sisi Answers).
    matched = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT Id, AcceptedAnswerId,
                   ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{QUESTIONS_PARQUET}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
        ) q
        WHERE rn = 1
        AND q.AcceptedAnswerId IN (
            SELECT DISTINCT Id FROM read_parquet('{ANSWERS_PARQUET}')
        )
    """).fetchone()[0]

    missing = total_q - matched
    match_pct = matched / total_q * 100 if total_q else 0
    missing_pct = missing / total_q * 100 if total_q else 0

    print(f"Total pertanyaan dengan AcceptedAnswerId terisi : {total_q:,}")
    print(f"  -> Ketemu jawabannya di Answers parquet        : {matched:,} ({match_pct:.1f}%)")
    print(f"  -> TIDAK ketemu (hilang saat merge/dedup?)      : {missing:,} ({missing_pct:.1f}%)")

    print("\n" + "-" * 70)
    if missing_pct > 15:
        print(f"[PERHATIAN] {missing_pct:.1f}% missing tergolong besar -- ini kemungkinan\n"
              f"masalah sistemik di proses merge SORD (bukan cuma kebetulan sample\n"
              f"kecil), perlu ditelusuri sebelum lanjut ke run 384 sample resmi.\n"
              f"Kemungkinan penyebab: dedup di 3_merge_sord_sources.py membuang\n"
              f"row Answers yang jadi accepted-answer bagi Question lain, atau\n"
              f"tipe data AcceptedAnswerId (float vs int) tidak match dengan Id.")
    else:
        print(f"[OK] {missing_pct:.1f}% missing masih dalam rentang wajar untuk data\n"
              f"komunitas SO (jawaban terhapus/pertanyaan lintas SO-network, dll.)\n"
              f"-- aman untuk lanjut ke sampling 384 seperti biasa; jumlah yang\n"
              f"dilewati saat prompting nanti proporsional kecil.")

    # Cek cepat: apakah AcceptedAnswerId bertipe konsisten dengan Id di Answers
    print("\n" + "-" * 70)
    print("Cek tipe data (kemungkinan penyebab mismatch teknis):")
    dtype_q = con.execute(f"DESCRIBE SELECT AcceptedAnswerId FROM read_parquet('{QUESTIONS_PARQUET}')").fetchone()
    dtype_a = con.execute(f"DESCRIBE SELECT Id FROM read_parquet('{ANSWERS_PARQUET}')").fetchone()
    print(f"  Questions.AcceptedAnswerId dtype : {dtype_q[1]}")
    print(f"  Answers.Id dtype                 : {dtype_a[1]}")


if __name__ == "__main__":
    main()