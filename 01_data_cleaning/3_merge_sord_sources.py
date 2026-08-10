"""
merge_sord_sources.py
======================
Fase 0.2 — Gabungkan Contain + LikeMinusContain per kategori (questions,
answers, comments) dengan flag provenance (`match_source`), lalu simpan
sebagai Parquet untuk pemrosesan lanjutan yang jauh lebih cepat & hemat
memori dibanding CSV mentah (relevan karena skala data SORD sampai
puluhan juta baris).

KEPUTUSAN DESAIN
-----------------
Kita TIDAK memilih Contain atau LikeMinusContain saja — keduanya digabung
(union) dengan kolom `match_source` yang menyimpan asal tiap baris:
    questions -> title_contain | title_like | body_contain | body_like
    answers   -> contain | like
    comments  -> contain | like

PENTING -- strategi dedup DITUNDA (bukan dihilangkan): union mentah ditulis
langsung ke `<category>_raw_union.parquet` TANPA dedup penuh di tahap ini,
supaya prosesnya ringan (streaming, tidak menahan seluruh dataset termasuk
kolom Body/Text yang besar di memori sekaligus -- ini yang bikin versi
sebelumnya OOM di laptop dengan RAM terbatas). LikeMinusContain per definisi
DISJOINT dari Contain, jadi duplikat Id hanya mungkin muncul dari overlap
Title-match vs Body-match (untuk questions) -- overlap ini dihitung terpisah
di `<category>_provenance.parquet` (Id -> daftar match_source, ringan karena
cuma 2 kolom sempit, tanpa Body/Text). Dedup baris penuh (pilih satu record
representatif per Id) baru dilakukan nanti saat load ke Neo4j, di mana
`MERGE (n {id: ...})` sudah otomatis idempotent -- jadi tidak perlu proses
dedup berat terpisah di Python/DuckDB.

INSTALL DEPENDENCY
-------------------
    pip install duckdb

CARA PAKAI
----------
    python merge_sord_sources.py --data-dir ./data --output-dir ./data/merged

OUTPUT
------
    data/merged/questions_raw_union.parquet    (union mentah, mungkin ada duplikat Id)
    data/merged/questions_provenance.parquet   (Id -> match_source, untuk cek overlap)
    data/merged/answers_raw_union.parquet
    data/merged/answers_provenance.parquet
    data/merged/comments_raw_union.parquet
    data/merged/comments_provenance.parquet
    + laporan ringkas jumlah baris & persentase overlap per kategori
"""

import argparse
import sys
from pathlib import Path

import duckdb


# ---------------------------------------------------------------------
# Definisi sumber per kategori
# ---------------------------------------------------------------------

CATEGORY_SOURCES = {
    "questions": [
        # (path relatif, label match_source)
        ("questions/QuestionsTitle_Contain.csv", "title_contain"),
        ("questions/QuestionsTitle_LikeMinusContain.csv", "title_like"),
        ("questions/QuestionsBody_Contain.csv", "body_contain"),
        ("questions/QuestionsBody_LikeMinusContain.csv", "body_like"),
    ],
    "answers": [
        ("answers/Answers_Contain.csv", "contain"),
        ("answers/Answers_LikeMinusContain.csv", "like"),
    ],
    "comments": [
        ("comments/Comments_Contain.csv", "contain"),
        ("comments/Comments_LikeMinusContain.csv", "like"),
    ],
}


def merge_category(con: duckdb.DuckDBPyConnection, data_dir: Path,
                    category: str, sources: list, output_dir: Path):
    print(f"\n=== Menggabungkan kategori: {category} ===")

    union_parts = []
    for rel_path, label in sources:
        full_path = data_dir / rel_path
        if not full_path.exists():
            print(f"  [WARN] {rel_path} tidak ditemukan, dilewati")
            continue
        union_parts.append(
            f"SELECT *, '{label}' AS match_source FROM read_csv_auto("
            f"'{full_path.as_posix()}', ignore_errors=true)"
        )

    if not union_parts:
        print(f"  [SKIP] tidak ada sumber ditemukan untuk kategori {category}")
        return

    union_sql = " UNION ALL BY NAME ".join(union_parts)
    con.execute(f"CREATE OR REPLACE VIEW raw_union AS {union_sql}")

    # --- Langkah 1: tulis union MENTAH langsung ke Parquet (streaming, ringan) ---
    # TIDAK di-dedup di sini secara sengaja. LikeMinusContain per definisi
    # disjoint dari Contain (LikeMinusContain = LIKE minus CONTAINS), jadi
    # duplikat Id HANYA mungkin terjadi antara set Title-match vs Body-match
    # (untuk questions) -- overlap yang relatif kecil (lihat laporan overlap
    # di bawah). Dedup penuh sengaja DITUNDA ke tahap load-KG (Neo4j MERGE
    # secara alami idempotent by Id), supaya langkah ini tidak perlu menahan
    # seluruh dataset (termasuk kolom Body yang besar) di memori sekaligus.
    raw_out_path = output_dir / f"{category}_raw_union.parquet"
    con.execute(f"COPY raw_union TO '{raw_out_path.as_posix()}' (FORMAT PARQUET)")
    raw_count = con.execute(f"SELECT COUNT(*) FROM raw_union").fetchone()[0]
    print(f"  [OK] Union mentah ({raw_count:,} baris, termasuk kemungkinan duplikat Id) "
          f"tersimpan -> {raw_out_path}")

    # --- Langkah 2: laporan overlap Id (RINGAN -- cuma 2 kolom sempit, tanpa Body/Text) ---
    # ORDER BY match_source di dalam STRING_AGG penting supaya kombinasi
    # 'body_contain,title_contain' dan 'title_contain,body_contain' (kombinasi
    # yang SAMA, cuma beda urutan penggabungan) tidak terhitung sebagai dua
    # kategori terpisah di laporan distribusi.
    overlap_sql = f"""
        SELECT Id, string_agg(DISTINCT match_source, ',' ORDER BY match_source) AS match_source,
               COUNT(DISTINCT match_source) AS n_sources
        FROM raw_union
        GROUP BY Id
    """
    con.execute(f"CREATE OR REPLACE TABLE {category}_provenance AS {overlap_sql}")

    total_unique = con.execute(f"SELECT COUNT(*) FROM {category}_provenance").fetchone()[0]
    n_overlap = con.execute(
        f"SELECT COUNT(*) FROM {category}_provenance WHERE n_sources > 1"
    ).fetchone()[0]
    print(f"  [OK] Id unik: {total_unique:,} | Id yang muncul di >1 sumber (duplikat): "
          f"{n_overlap:,} ({n_overlap/total_unique*100:.2f}%)")

    prov_out_path = output_dir / f"{category}_provenance.parquet"
    con.execute(
        f"COPY {category}_provenance TO '{prov_out_path.as_posix()}' (FORMAT PARQUET)"
    )
    print(f"  [OK] Tabel provenance (Id -> match_source) tersimpan -> {prov_out_path}")

    print(f"  Distribusi match_source (top 10 kombinasi):")
    dist = con.execute(f"""
        SELECT match_source, COUNT(*) AS n
        FROM {category}_provenance
        GROUP BY match_source ORDER BY n DESC LIMIT 10
    """).fetchall()
    for src, n in dist:
        print(f"    {src}: {n:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../00_datasource/raw",
                         help="Root folder data SORD mentah (default: ../00_datasource/raw, "
                              "relatif terhadap folder 01_data_cleaning/)")
    parser.add_argument("--output-dir", default="../00_datasource/merged",
                         help="Folder output Parquet gabungan (default: ../00_datasource/merged)")
    parser.add_argument("--categories", nargs="+",
                         default=["questions", "answers", "comments"],
                         choices=["questions", "answers", "comments"])
    parser.add_argument("--memory-limit", default="2GB",
                         help="Batas memori DuckDB. Turunkan kalau masih OOM (mis. '1GB').")
    parser.add_argument("--threads", type=int, default=2,
                         help="Jumlah thread DuckDB. Turunkan ke 1 kalau masih OOM.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # PENTING: pakai database on-disk (bukan ':memory:') supaya DuckDB bisa
    # spill hasil intermediate ke disk saat RAM tidak cukup, alih-alih error
    # OOM. Kombinasi ini yang memperbaiki error "Out of Memory" sebelumnya.
    db_path = output_dir / "_workspace.duckdb"
    con = duckdb.connect(database=str(db_path))
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    print(f"[config] memory_limit={args.memory_limit} threads={args.threads} "
          f"temp_directory={tmp_dir}")

    for category in args.categories:
        merge_category(con, data_dir, category, CATEGORY_SOURCES[category], output_dir)

    print("\n" + "=" * 78)
    print("SELESAI. File Parquet (raw union + provenance) siap dipakai untuk "
          "Fase 0.4 (sampling) dan Fase 2 (KG construction). Dedup penuh by Id "
          "akan terjadi otomatis saat load ke Neo4j (MERGE idempotent).")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())