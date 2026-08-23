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
  5. Laporan ringkas PASS/WARN/FAIL per file + chart perbandingan actual vs
     expected (ukuran & jumlah baris) dan ringkasan status.

LOKASI OUTPUT
-------------
Default laporan & chart SELALU dihitung relatif terhadap ROOT REPO (bukan
cwd), sama seperti 1_preview_files.py:
    <repo_root>/report/2_integrity_report.md
    <repo_root>/report/img/integrity_*.png

KALAU LAPORAN SUDAH ADA
-------------------------
Row-count exact untuk file multi-GB bisa memakan waktu lama. Supaya tidak
mengulang proses berat itu setiap kali script dijalankan, default-nya:
kalau laporan sudah ada, LANGSUNG TAMPILKAN isi laporan lama (tanpa proses
ulang apa pun). Pakai --force kalau memang mau regenerate (mis. setelah
data diunduh ulang).

INSTALL DEPENDENCY
-------------------
    pip install py7zr pandas matplotlib

CARA PAKAI
----------
    python check_data_integrity.py --data-dir ./data

    # kalau file besar (Answers_Contain 3.27GB dst) mau dilewati row-count
    # exact karena terlalu lama, cek ukuran & header saja:
    python check_data_integrity.py --data-dir ./data --quick

    # paksa regenerate walau laporan sudah ada:
    python check_data_integrity.py --force

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

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

def check_size(path: Path, expected_bytes: int) -> tuple[bool, str, int]:
    actual = path.stat().st_size
    diff_ratio = abs(actual - expected_bytes) / expected_bytes
    ok = diff_ratio <= SIZE_TOLERANCE
    msg = (
        f"size actual={actual/1e6:,.1f}MB expected={expected_bytes/1e6:,.1f}MB "
        f"(selisih {diff_ratio*100:.1f}%)"
    )
    return ok, msg, actual


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
# Chart: actual vs expected (size & rows) + ringkasan status
# ---------------------------------------------------------------------

def generate_charts(records: list[dict], img_dir: Path) -> list[str]:
    """records: list of dict dengan key rel_path, size_actual, size_expected,
    rows_actual (bisa None kalau --quick), rows_expected, overall."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib tidak terinstall -- chart dilewati. "
              "Install dengan: pip install matplotlib")
        return []

    img_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    names = [r["rel_path"].split("/")[-1] for r in records]

    # --- Chart 1: ukuran file actual vs expected (log scale, MB) ---
    size_actual = [r["size_actual"] / 1e6 for r in records]
    size_expected = [r["size_expected"] / 1e6 for r in records]

    y = np.arange(len(names))
    height = 0.35
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names))))
    ax.barh(y + height / 2, size_expected, height, label="Expected (paper)", color="#C44E52")
    ax.barh(y - height / 2, size_actual, height, label="Actual", color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("Ukuran file (MB, log scale)")
    ax.set_title("Ukuran File: Actual vs Expected (Tabel 11 Paper SORD)")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fname1 = "integrity_size_comparison.png"
    fig.savefig(img_dir / fname1, dpi=150)
    plt.close(fig)
    generated.append(fname1)

    # --- Chart 2: jumlah baris actual vs expected (log scale), kalau ada ---
    rows_records = [r for r in records if r.get("rows_actual") is not None]
    if rows_records:
        rnames = [r["rel_path"].split("/")[-1] for r in rows_records]
        rows_actual = [r["rows_actual"] for r in rows_records]
        rows_expected = [r["rows_expected"] for r in rows_records]
        ry = np.arange(len(rnames))
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rnames))))
        ax.barh(ry + height / 2, rows_expected, height, label="Expected (paper)", color="#C44E52")
        ax.barh(ry - height / 2, rows_actual, height, label="Actual", color="#55A868")
        ax.set_yticks(ry)
        ax.set_yticklabels(rnames)
        ax.set_xscale("log")
        ax.set_xlabel("Jumlah baris (log scale)")
        ax.set_title("Jumlah Baris: Actual vs Expected (Tabel 7/8 Paper SORD)")
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        fname2 = "integrity_rows_comparison.png"
        fig.savefig(img_dir / fname2, dpi=150)
        plt.close(fig)
        generated.append(fname2)

    # --- Chart 3: ringkasan status PASS/WARN/FAIL ---
    statuses = [r["overall"] for r in records]
    counts = {"PASS": statuses.count("PASS"), "WARN": statuses.count("WARN"),
              "FAIL": statuses.count("FAIL")}
    colors = {"PASS": "#55A868", "WARN": "#DD8452", "FAIL": "#C44E52"}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.keys(), counts.values(), color=[colors[k] for k in counts])
    ax.set_ylabel("Jumlah file")
    ax.set_title("Ringkasan Status Integritas Data")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 0.05, str(v), ha="center")
    plt.tight_layout()
    fname3 = "integrity_status_summary.png"
    fig.savefig(img_dir / fname3, dpi=150)
    plt.close(fig)
    generated.append(fname3)

    return generated


def show_existing_report(output_path: Path):
    print(f"\n[i] Laporan integritas sudah ada -> {output_path.resolve()}")
    print("[i] Menampilkan laporan yang ada (tidak proses ulang). "
          "Pakai --force untuk regenerate.\n")
    print("-" * 78)
    print(output_path.read_text(encoding="utf-8"))
    print("-" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../00_datasource/raw",
                         help="Root folder data SORD mentah (default: ../00_datasource/raw, "
                              "relatif terhadap folder 01_data_cleaning/)")
    parser.add_argument("--quick", action="store_true",
                         help="Lewati exact row-count (hanya cek ukuran & header, jauh lebih cepat)")
    parser.add_argument("--skip-extract", action="store_true",
                         help="Jangan coba ekstrak .7z otomatis")
    parser.add_argument("--output", default=None,
                         help="Path file Markdown output. Default: <repo_root>/report/2_integrity_report.md")
    parser.add_argument("--img-dir", default=None,
                         help="Folder chart .png. Default: <repo_root>/report/img")
    parser.add_argument("--no-charts", action="store_true",
                         help="Lewati pembuatan chart")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate laporan walau sudah ada (default: kalau sudah ada, "
                              "langsung tampilkan laporan lama tanpa proses ulang)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (REPO_ROOT / "report" / "2_integrity_report.md")
    img_dir = Path(args.img_dir) if args.img_dir else (REPO_ROOT / "report" / "img")

    # --- Kalau laporan sudah ada dan tidak --force: langsung tampilkan, JANGAN proses ulang ---
    if output_path.exists() and not args.force:
        show_existing_report(output_path)
        return

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Folder {data_dir} tidak ditemukan.")
        sys.exit(1)

    if not args.skip_extract:
        extract_archives(data_dir)

    print("\n" + "=" * 78)
    print("LAPORAN INTEGRITAS DATA SORD")
    print("=" * 78)

    results = []       # (rel_path, status, note) -- untuk console, sama seperti sebelumnya
    records = []        # data lengkap per file -- untuk report .md + chart
    md_sections = []

    for rel_path, spec in FILE_SPECS.items():
        full_path = data_dir / rel_path
        category = rel_path.split("/")[0]
        print(f"\n--- {rel_path} ---")
        section = [f"### `{rel_path}`", ""]

        try:
            exists = full_path.exists()
        except PermissionError:
            print("  [FAIL] permission denied saat mengakses file — cek izin baca "
                  "(coba: chmod -R u+rw ./data , dan di macOS: xattr -cr ./data "
                  "untuk hapus flag quarantine hasil download/ekstraksi)")
            results.append((rel_path, "FAIL", "permission denied"))
            section.append("**[FAIL]** permission denied")
            md_sections.append("\n".join(section))
            continue

        if not exists:
            print("  [FAIL] file tidak ditemukan")
            results.append((rel_path, "FAIL", "file tidak ditemukan"))
            section.append("**[FAIL]** file tidak ditemukan")
            md_sections.append("\n".join(section))
            continue

        # 1. Ukuran
        try:
            size_ok, size_msg, size_actual = check_size(full_path, spec["size_bytes"])
        except PermissionError:
            print("  [FAIL] permission denied saat membaca ukuran file")
            results.append((rel_path, "FAIL", "permission denied"))
            section.append("**[FAIL]** permission denied")
            md_sections.append("\n".join(section))
            continue
        print(f"  [{'OK' if size_ok else 'WARN'}] {size_msg}")
        section.append(f"- Ukuran: {'OK' if size_ok else 'WARN'} -- {size_msg}")

        # 2. Row count + kolom (kecuali --quick)
        row_ok = None
        col_ok = None
        n_rows = None
        if not args.quick:
            print("  menghitung baris (bisa lama untuk file >1GB)...")
            try:
                n_rows, columns, had_bad_lines = count_rows_and_columns(full_path)
            except PermissionError:
                print("  [FAIL] permission denied saat membaca isi file")
                results.append((rel_path, "FAIL", "permission denied"))
                section.append("**[FAIL]** permission denied saat membaca isi file")
                md_sections.append("\n".join(section))
                continue
            if n_rows is None:
                results.append((rel_path, "FAIL", "gagal parse file"))
                section.append("**[FAIL]** gagal parse file")
                md_sections.append("\n".join(section))
                continue
            diff_ratio = abs(n_rows - spec["rows"]) / spec["rows"]
            row_ok = diff_ratio <= ROW_TOLERANCE
            print(f"  [{'OK' if row_ok else 'WARN'}] rows actual={n_rows:,} "
                  f"expected={spec['rows']:,} (selisih {diff_ratio*100:.2f}%)")
            section.append(f"- Baris: {'OK' if row_ok else 'WARN'} -- actual={n_rows:,}, "
                            f"expected={spec['rows']:,} (selisih {diff_ratio*100:.2f}%)")
            if had_bad_lines:
                print("  [WARN] ada baris bermasalah yang di-skip parser python "
                      "— kemungkinan ada quoting/newline tidak standar di Body/Text")
                section.append("- [WARN] ada baris bermasalah yang di-skip parser python")

            expected_min_cols = MIN_COLUMNS.get(category, set())
            if expected_min_cols:
                missing = expected_min_cols - (columns or set())
                col_ok = not missing
                if not col_ok:
                    print(f"  [WARN] kolom minimal hilang: {missing}")
                    section.append(f"- [WARN] kolom minimal hilang: {missing}")
                else:
                    print(f"  [OK] kolom minimal lengkap")
                    section.append("- Kolom minimal: OK, lengkap")

        # 3. Duplikat Id (sample)
        try:
            n_dup = check_duplicate_ids(full_path)
        except PermissionError:
            print("  [WARN] permission denied saat cek duplikat Id — dilewati")
            n_dup = None
        if n_dup is not None:
            status = "OK" if n_dup == 0 else "WARN"
            print(f"  [{status}] duplikat Id pada {min(500_000, spec['rows']):,} baris pertama: {n_dup}")
            section.append(f"- Duplikat Id (sample): {status} -- {n_dup} duplikat")

        overall = "PASS"
        if not size_ok:
            overall = "WARN"
        if row_ok is False:
            overall = "WARN"
        if col_ok is False:
            overall = "WARN"
        results.append((rel_path, overall, ""))
        section.insert(1, f"**Status keseluruhan: {overall}**")
        section.append("")
        md_sections.append("\n".join(section))

        records.append({
            "rel_path": rel_path,
            "size_actual": size_actual,
            "size_expected": spec["size_bytes"],
            "rows_actual": n_rows,
            "rows_expected": spec["rows"],
            "overall": overall,
        })

    print("\n" + "=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    for rel_path, status, note in results:
        print(f"  [{status:4s}] {rel_path} {note}")

    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    n_total = len(results)
    print(f"\nTotal: {n_total} file dicek, {n_fail} FAIL, {n_warn} WARN, "
          f"{n_total-n_fail-n_warn} PASS bersih.")

    if n_fail:
        action_msg = ("Ada file FAIL (tidak ditemukan/tidak bisa diparse) — "
                       "unduh ulang atau ekstrak ulang file tersebut sebelum lanjut ke Fase 0.2.")
    elif n_warn:
        action_msg = ("Ada WARN — biasanya masih aman dipakai (selisih kecil "
                       "karena pembulatan), tapi cek manual dulu terutama kalau selisih "
                       ">5% atau ada baris bermasalah (quoting/newline).")
    else:
        action_msg = ("Semua file lolos cek integritas dasar. Lanjut ke Fase 0.2 "
                       "(keputusan Contain vs LikeMinusContain, sampling).")
    print(f"\n[ACTION] {action_msg}")

    # --- Bangun laporan markdown ---
    md_parts = [
        "# Laporan Integritas Data SORD",
        "",
        f"Total file dicek: {n_total} | FAIL: {n_fail} | WARN: {n_warn} | "
        f"PASS: {n_total-n_fail-n_warn}",
        "",
        f"**Tindakan:** {action_msg}",
        "",
        "## Cara Membaca Laporan Ini",
        "",
        "Tabel **Ringkasan Status per File** di bawah berisi kolom:",
        "",
        "| Kolom | Isi |",
        "| --- | --- |",
        "| `File` | Path file CSV mentah SORD relatif terhadap `--data-dir` "
        "(mis. `questions/QuestionsTitle_Contain.csv`) |",
        "| `Status` | Hasil cek keseluruhan untuk file tsb: `PASS`, `WARN`, atau `FAIL` (lihat definisi di bawah) |",
        "",
        "**Definisi status:**",
        "",
        f"- **PASS** -- file ditemukan, ukuran & jumlah baris sesuai referensi paper SORD "
        f"(toleransi ukuran {SIZE_TOLERANCE*100:.0f}%, toleransi jumlah baris {ROW_TOLERANCE*100:.0f}%), "
        f"kolom minimal lengkap, tidak ada duplikat Id pada sample.",
        f"- **WARN** -- file ditemukan & bisa diproses, tapi ada SATU ATAU LEBIH dari: "
        f"selisih ukuran/jumlah baris di luar toleransi, kolom minimal hilang, ada baris "
        f"bermasalah (quoting/newline tidak standar), atau ada duplikat Id. Biasanya masih "
        f"aman dipakai, tapi sebaiknya dicek manual kalau selisih >5%.",
        "- **FAIL** -- file tidak ditemukan, tidak bisa diparse sama sekali, atau permission "
        "denied saat membaca. Perlu diunduh/diekstrak ulang sebelum lanjut ke Fase 0.2.",
        "",
        "**Detail per file** (di bagian bawah laporan) merinci masing-masing pengecekan yang "
        "berkontribusi ke status di atas: ukuran file (actual vs expected dari Tabel 11 paper "
        "SORD), jumlah baris (actual vs expected dari Tabel 7/8 paper SORD, dilewati kalau "
        "`--quick`), kelengkapan kolom minimal, dan jumlah duplikat Id pada sample "
        "(maks. 500.000 baris pertama).",
        "",
    ]

    if not args.no_charts and records:
        print(f"\nMembuat chart integritas -> {img_dir}")
        chart_files = generate_charts(records, img_dir)
        if chart_files:
            md_parts.append("## Ringkasan Visual")
            md_parts.append("")
            for cf in chart_files:
                md_parts.append(f"![{cf}](img/{cf})")
                md_parts.append("")

    md_parts.append("## Ringkasan Status per File")
    md_parts.append("")
    md_parts.append("| File | Status |")
    md_parts.append("| --- | --- |")
    for rel_path, status, note in results:
        md_parts.append(f"| `{rel_path}` | {status} {note} |")
    md_parts.append("")
    md_parts.append("## Detail per File")
    md_parts.append("")
    md_parts.extend(md_sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n[OK] Laporan tersimpan -> {output_path.resolve()}")


if __name__ == "__main__":
    sys.exit(main())