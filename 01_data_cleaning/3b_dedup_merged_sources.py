"""
3b_dedup_merged_sources.py
============================
Dedup `{category}_raw_union.parquet` (questions/answers/comments) by `Id`,
menimpa file sumber di `00_datasource/merged/` -- supaya SEMUA script
konsumen (Kondisi A/B/C, KG construction, SORD Explorer, dst.) otomatis
dapat data bersih tanpa perlu dedup ulang masing-masing.

LATAR BELAKANG
--------------
`3_merge_sord_sources.py` SENGAJA menulis union MENTAH (bisa ada duplikat
`Id`) -- didesain supaya penulisan Parquet tetap ringan/streaming, dengan
asumsi dedup akan terjadi otomatis nanti saat load ke Neo4j (`MERGE`
idempotent by Id). Asumsi itu ternyata tidak berlaku untuk tahap
KOMPUTASI DI PANDAS SEBELUM sampai ke Neo4j (mis. `Series.map()` butuh
index unik) -- lihat error `InvalidIndexError: Reindexing only valid with
uniquely valued Index objects` saat build_knowledge_graph.py jalan di data
questions penuh (2,811,534 baris, sebagian Id duplikat karena overlap
Title-match vs Body-match).

Alih-alih dedup berulang di tiap script konsumen, script ini dedup SEKALI
di sumbernya.

METODE DEDUP
------------
`QUALIFY ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) = 1`
-- pilih SATU baris representatif per Id (baris pertama secara alfabetis
berdasarkan match_source, deterministik & reproducible). Generik terhadap
skema kolom apa pun (tidak hardcode nama kolom selain Id/match_source),
jadi jalan untuk questions/answers/comments meski skema aslinya beda-beda.

`questions_provenance.parquet` / `answers_provenance.parquet` /
`comments_provenance.parquet` (Id -> daftar match_source, dipakai untuk
laporan overlap di 5_eda_trust_signals.py) TIDAK diubah/dihapus -- tabel
itu justru MEMANG dirancang untuk merekam overlap tersebut, jadi tetap
valid dan konsisten walau raw_union sekarang sudah dedup.

SAFETY
------
- File asli di-backup dulu (`{category}_raw_union.parquet.predup_backup`)
  sebelum ditimpa, kecuali --no-backup.
- Penulisan ke file temp dulu, baru os.replace() (atomic) ke nama asli --
  supaya kalau proses terhenti di tengah jalan, file asli tidak korup.

INSTALL DEPENDENCY
-------------------
    pip install duckdb pandas

CARA PAKAI
----------
    python 3b_dedup_merged_sources.py                     # semua kategori
    python 3b_dedup_merged_sources.py --categories questions
    python 3b_dedup_merged_sources.py --dry-run            # cuma laporan, tidak menulis apa pun
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb


def dedup_category(con: duckdb.DuckDBPyConnection, merged_dir: Path, category: str,
                    make_backup: bool, dry_run: bool) -> None:
    src_path = merged_dir / f"{category}_raw_union.parquet"
    if not src_path.exists():
        print(f"  [SKIP] {src_path.name} tidak ditemukan.")
        return

    print(f"\n=== Dedup kategori: {category} ===")

    total_all, total_distinct = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT Id) FROM read_parquet('{src_path.as_posix()}')"
    ).fetchone()

    n_dup = total_all - total_distinct
    print(f"  Total baris (sebelum dedup) : {total_all:,}")
    print(f"  Id unik                     : {total_distinct:,}")
    print(f"  Baris duplikat akan dibuang : {n_dup:,} "
          f"({n_dup / total_all * 100:.2f}% dari total)" if total_all else "")

    if n_dup == 0:
        print(f"  [OK] Tidak ada duplikat Id -- {src_path.name} sudah bersih, dilewati.")
        return

    if dry_run:
        print("  [DRY-RUN] Tidak menulis apa pun.")
        return

    tmp_path = merged_dir / f"{category}_raw_union.dedup_tmp.parquet"
    backup_path = merged_dir / f"{category}_raw_union.parquet.predup_backup"

    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
                FROM read_parquet('{src_path.as_posix()}')
            )
            WHERE rn = 1
        ) TO '{tmp_path.as_posix()}' (FORMAT PARQUET)
        """
    )
    elapsed = time.time() - t0

    # Verifikasi cepat sebelum menimpa file asli -- kalau row count hasil
    # dedup tidak cocok dengan total_distinct yang dihitung di awal, JANGAN
    # ganti file asli (lebih aman gagal dengan jelas daripada diam-diam
    # menulis data yang salah).
    verify_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{tmp_path.as_posix()}')"
    ).fetchone()[0]
    if verify_count != total_distinct:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"[ABORT] Verifikasi gagal untuk {category}: hasil dedup {verify_count:,} baris, "
            f"diharapkan {total_distinct:,}. File asli TIDAK diubah. Coba lagi atau laporkan bug."
        )

    if make_backup:
        shutil.copy2(src_path, backup_path)
        print(f"  [OK] Backup file asli -> {backup_path.name}")

    os.replace(tmp_path, src_path)  # atomic di filesystem yang sama
    print(f"  [OK] {src_path.name} ditimpa dengan versi dedup "
          f"({verify_count:,} baris, {elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", default="../00_datasource/merged",
                         help="Folder berisi *_raw_union.parquet (default: ../00_datasource/merged)")
    parser.add_argument("--categories", nargs="+", default=["questions", "answers", "comments"],
                         choices=["questions", "answers", "comments"])
    parser.add_argument("--no-backup", action="store_true",
                         help="Skip backup file asli sebelum ditimpa (hemat disk, tapi TIDAK bisa di-undo)")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Cuma laporan duplikat, tidak menulis apa pun")
    args = parser.parse_args()

    merged_dir = Path(args.merged_dir)
    if not merged_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan: {merged_dir}")
        sys.exit(1)

    tmp_dir = merged_dir / "_duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = merged_dir / "_dedup_workspace.duckdb"

    con = duckdb.connect(database=str(db_path))
    con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA threads={args.threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    print(f"[config] memory_limit={args.memory_limit} threads={args.threads} "
          f"merged_dir={merged_dir} dry_run={args.dry_run}")

    for category in args.categories:
        dedup_category(con, merged_dir, category, make_backup=not args.no_backup, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("SELESAI. questions_provenance.parquet / answers_provenance.parquet / "
          "comments_provenance.parquet TIDAK diubah -- tetap valid untuk laporan "
          "overlap Title/Body-match (5_eda_trust_signals.py).")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())