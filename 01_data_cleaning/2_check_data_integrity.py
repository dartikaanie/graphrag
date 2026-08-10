"""
check_data_integrity.py
========================
Fase 0.1 — Cek integritas data SORD setelah diunduh/diekstrak.

Skrip ini melakukan:
  1. Ekstraksi otomatis file .7z yang belum diekstrak (pakai py7zr).
  2. Validasi ukuran file dibanding Tabel 11 (paper SORD).
  3. Validasi JUMLAH BARIS PERSIS dibanding Tabel 7 & Tabel 8 (paper SORD)
     — ini bukan estimasi, tapi angka yang benar-benar dilaporkan penulis.
  4. Cek header kolom minimal & duplikat Id.
  5. Laporan ringkas PASS/WARN/FAIL per file.

INSTALL DEPENDENCY
-------------------
    pip install py7zr pandas

CARA PAKAI
----------
    python check_data_integrity.py --data-dir ./data

    # kalau file besar (Answers_Contain 3.27GB dst) mau dilewati row-count
    # exact karena terlalu lama, cek ukuran & header saja:
    python check_data_integrity.py --data-dir ./data --quick

CATATAN PENTING SOAL ANGKA REFERENSI
-------------------------------------
Jumlah baris di bawah ini dihitung ulang dari persentase pada Tabel 7 paper
SORD (mis. Question Title: Exact Matching (CONTAINS) = 0.31% dari 24,198,178
total = 73,901 baris). Karena persentase di paper dibulatkan, ada toleransi
kecil (default 1%) yang dipakai di sini — bukan exact-match 100%, tapi cukup
ketat untuk menangkap file yang corrupt/truncated saat unduh atau ekstraksi.
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

# Body/Text field pada SO bisa sangat panjang; default limit csv module
# (128KB-an) bisa terlalu kecil dan memicu error parsing di file besar.
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------
# Referensi resmi dari paper SORD (Fatima & Maqbool, 2026)
# Tabel 7 (questions/answers/comments) dan Tabel 8 (additional-metadata)
# ---------------------------------------------------------------------

FILE_SPECS = {
    "questions/QuestionsTitle_Contain.csv": {
        "size_bytes": 118_000_000, "rows": 73_901,
    },
    "questions/QuestionsBody_Contain.csv": {
        "size_bytes": 3_260_000_000, "rows": 1_135_148,
    },
    "questions/QuestionsTitle_LikeMinusContain.csv": {
        "size_bytes": 178_000_000, "rows": 90_685,
    },
    "questions/QuestionsBody_LikeMinusContain.csv": {
        "size_bytes": 3_890_000_000, "rows": 1_511_800,
    },
    "answers/Answers_Contain.csv": {
        "size_bytes": 3_270_000_000, "rows": 2_228_118,
    },
    "answers/Answers_LikeMinusContain.csv": {
        "size_bytes": 2_100_000_000, "rows": 1_427_770,
    },
    "comments/Comments_Contain.csv": {
        "size_bytes": 586_000_000, "rows": 1_900_242,
    },
    "comments/Comments_LikeMinusContain.csv": {
        "size_bytes": 634_000_000, "rows": 2_114_468,
    },
    "additional-metadata/FilteredBadges.csv": {
        "size_bytes": 2_150_000_000, "rows": 34_830_450,
    },
    "additional-metadata/FilteredTags.csv": {
        "size_bytes": 982_000, "rows": 61_672,
    },
    "additional-metadata/FilteredUsers.csv": {
        "size_bytes": 305_000_000, "rows": 1_993_211,
    },
    "additional-metadata/FilteredVotes.csv": {
        "size_bytes": 2_770_000_000, "rows": 50_725_814,
    },
}

# Kolom minimal yang wajib ada per kategori file (bukan skema lengkap,
# hanya penanda bahwa file tidak corrupt/salah-potong).
MIN_COLUMNS = {
    "questions": {"Id", "CreationDate", "Score", "Body", "OwnerUserId"},
    "answers": {"Id", "CreationDate", "Score", "Body", "OwnerUserId"},
    # Comments punya skema BEDA dari Questions/Answers (bukan turunan Posts):
    # field teksnya bernama "Text", bukan "Body" -- dikonfirmasi dari
    # AnalysisofInsightStimulatingExample.ipynb (query SELECT ... [Text] ...
    # FROM Comments_Contain / Comments_LikeMinusContain).
    "comments": {"Id", "PostId", "CreationDate", "Score", "Text"},
    "additional-metadata": set(),  # skema beda-beda, dicek longgar saja
}

SIZE_TOLERANCE = 0.15   # 15% -> karena ambiguitas MB desimal vs biner
ROW_TOLERANCE = 0.01    # 1% -> karena pembulatan persentase di paper


# ---------------------------------------------------------------------
# 1. Ekstraksi .7z
# ---------------------------------------------------------------------

def extract_archives(data_dir: Path):
    try:
        import py7zr
    except ImportError:
        print("[extract] py7zr tidak terinstall. Jalankan: pip install py7zr")
        print("[extract] Melewati langkah ekstraksi otomatis — pastikan Anda "
              "sudah extract .7z secara manual sebelum lanjut.")
        return

    archives = sorted(data_dir.glob("*.7z"))
    if not archives:
        print("[extract] Tidak ada file .7z ditemukan di", data_dir)
        return

    for archive in archives:
        target_folder = data_dir / archive.stem
        if target_folder.exists() and any(target_folder.glob("*.csv")):
            print(f"[extract] {archive.name} -> sudah diekstrak, dilewati")
            continue
        print(f"[extract] mengekstrak {archive.name} ...")
        try:
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(path=data_dir)
            print(f"[extract] selesai: {archive.name}")
        except Exception as e:
            print(f"[extract][ERROR] gagal ekstrak {archive.name}: {e}")


# ---------------------------------------------------------------------
# 2. Cek ukuran file
# ---------------------------------------------------------------------

def check_size(path: Path, expected_bytes: int) -> tuple[bool, str]:
    actual = path.stat().st_size
    diff_ratio = abs(actual - expected_bytes) / expected_bytes
    ok = diff_ratio <= SIZE_TOLERANCE
    msg = (
        f"size actual={actual/1e6:,.1f}MB expected={expected_bytes/1e6:,.1f}MB "
        f"(selisih {diff_ratio*100:.1f}%)"
    )
    return ok, msg


# ---------------------------------------------------------------------
# 3. Hitung baris persis (menghormati quoting CSV multi-baris)
# ---------------------------------------------------------------------

def count_rows_and_columns(path: Path, chunksize: int = 200_000):
    """Hitung total baris data (tanpa header) dan ambil set nama kolom.
    Pakai pandas chunked read supaya file multi-GB tidak perlu masuk
    memori sekaligus. Body/Text field yang mengandung newline di dalam
    tanda kutip ditangani otomatis oleh parser C pandas selama file
    diekspor dengan quoting standar."""
    total_rows = 0
    columns = None
    bad_chunks = 0
    try:
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="warn",
            engine="c",
        )
        for i, chunk in enumerate(reader):
            if columns is None:
                columns = set(chunk.columns)
            total_rows += len(chunk)
    except Exception as e:
        print(f"    [warn] parser C gagal ({e}); mencoba engine python (lebih lambat, "
              f"baris rusak/quoting tidak standar akan di-skip)...")
        total_rows = 0
        columns = None
        try:
            # NOTE: 'low_memory' tidak didukung oleh engine='python', jangan diteruskan.
            reader = pd.read_csv(
                path, chunksize=chunksize,
                on_bad_lines="skip", engine="python",
            )
            for chunk in reader:
                if columns is None:
                    columns = set(chunk.columns)
                total_rows += len(chunk)
            bad_chunks = 1  # tandai bahwa ada baris yang di-skip -> dilaporkan sebagai WARN
        except Exception as e2:
            print(f"    [ERROR] gagal total membaca {path.name}: {e2}")
            return None, None, True

    return total_rows, columns, bool(bad_chunks)


# ---------------------------------------------------------------------
# 4. Cek duplikat Id (sample-based untuk file besar, biar cepat)
# ---------------------------------------------------------------------

def check_duplicate_ids(path: Path, sample_rows: int = 500_000):
    if "Id" not in pd.read_csv(path, nrows=1).columns:
        return None
    try:
        df_sample = pd.read_csv(path, usecols=["Id"], nrows=sample_rows)
        n_dup = df_sample["Id"].duplicated().sum()
        return int(n_dup)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../00_datasource/raw",
                         help="Root folder data SORD mentah (default: ../00_datasource/raw, "
                              "relatif terhadap folder 01_data_cleaning/)")
    parser.add_argument("--quick", action="store_true",
                         help="Lewati exact row-count (hanya cek ukuran & header, jauh lebih cepat)")
    parser.add_argument("--skip-extract", action="store_true",
                         help="Jangan coba ekstrak .7z otomatis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Folder {data_dir} tidak ditemukan.")
        sys.exit(1)

    if not args.skip_extract:
        extract_archives(data_dir)

    print("\n" + "=" * 78)
    print("LAPORAN INTEGRITAS DATA SORD")
    print("=" * 78)

    results = []
    for rel_path, spec in FILE_SPECS.items():
        full_path = data_dir / rel_path
        category = rel_path.split("/")[0]
        print(f"\n--- {rel_path} ---")

        try:
            exists = full_path.exists()
        except PermissionError:
            print("  [FAIL] permission denied saat mengakses file — cek izin baca "
                  "(coba: chmod -R u+rw ./data , dan di macOS: xattr -cr ./data "
                  "untuk hapus flag quarantine hasil download/ekstraksi)")
            results.append((rel_path, "FAIL", "permission denied"))
            continue

        if not exists:
            print("  [FAIL] file tidak ditemukan")
            results.append((rel_path, "FAIL", "file tidak ditemukan"))
            continue

        # 1. Ukuran
        try:
            size_ok, size_msg = check_size(full_path, spec["size_bytes"])
        except PermissionError:
            print("  [FAIL] permission denied saat membaca ukuran file")
            results.append((rel_path, "FAIL", "permission denied"))
            continue
        print(f"  [{'OK' if size_ok else 'WARN'}] {size_msg}")

        # 2. Row count + kolom (kecuali --quick)
        row_ok = None
        col_ok = None
        if not args.quick:
            print("  menghitung baris (bisa lama untuk file >1GB)...")
            try:
                n_rows, columns, had_bad_lines = count_rows_and_columns(full_path)
            except PermissionError:
                print("  [FAIL] permission denied saat membaca isi file")
                results.append((rel_path, "FAIL", "permission denied"))
                continue
            if n_rows is None:
                results.append((rel_path, "FAIL", "gagal parse file"))
                continue
            diff_ratio = abs(n_rows - spec["rows"]) / spec["rows"]
            row_ok = diff_ratio <= ROW_TOLERANCE
            print(f"  [{'OK' if row_ok else 'WARN'}] rows actual={n_rows:,} "
                  f"expected={spec['rows']:,} (selisih {diff_ratio*100:.2f}%)")
            if had_bad_lines:
                print("  [WARN] ada baris bermasalah yang di-skip parser python "
                      "— kemungkinan ada quoting/newline tidak standar di Body/Text")

            expected_min_cols = MIN_COLUMNS.get(category, set())
            if expected_min_cols:
                missing = expected_min_cols - (columns or set())
                col_ok = not missing
                if not col_ok:
                    print(f"  [WARN] kolom minimal hilang: {missing}")
                else:
                    print(f"  [OK] kolom minimal lengkap")

        # 3. Duplikat Id (sample)
        try:
            n_dup = check_duplicate_ids(full_path)
        except PermissionError:
            print("  [WARN] permission denied saat cek duplikat Id — dilewati")
            n_dup = None
        if n_dup is not None:
            status = "OK" if n_dup == 0 else "WARN"
            print(f"  [{status}] duplikat Id pada {min(500_000, spec['rows']):,} baris pertama: {n_dup}")

        overall = "PASS"
        if not size_ok:
            overall = "WARN"
        if row_ok is False:
            overall = "WARN"
        if col_ok is False:
            overall = "WARN"
        results.append((rel_path, overall, ""))

    print("\n" + "=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    for rel_path, status, note in results:
        print(f"  [{status:4s}] {rel_path} {note}")

    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    print(f"\nTotal: {len(results)} file dicek, {n_fail} FAIL, {n_warn} WARN, "
          f"{len(results)-n_fail-n_warn} PASS bersih.")

    if n_fail:
        print("\n[ACTION] Ada file FAIL (tidak ditemukan/tidak bisa diparse) — "
              "unduh ulang atau ekstrak ulang file tersebut sebelum lanjut ke Fase 0.2.")
    elif n_warn:
        print("\n[ACTION] Ada WARN — biasanya masih aman dipakai (selisih kecil "
              "karena pembulatan), tapi cek manual dulu terutama kalau selisih "
              ">5% atau ada baris bermasalah (quoting/newline).")
    else:
        print("\n[OK] Semua file lolos cek integritas dasar. Lanjut ke Fase 0.2 "
              "(keputusan Contain vs LikeMinusContain, sampling).")


if __name__ == "__main__":
    sys.exit(main())