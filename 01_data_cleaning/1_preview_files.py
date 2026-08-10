"""
preview_files.py
=================
Preview cepat isi tiap file CSV di dataset SORD: header + N baris pertama,
tanpa load seluruh file (penting untuk file 2-4GB) -- hasil ditulis sebagai
tabel Markdown ke satu file .md, supaya mudah dibaca/dibagikan (bisa dibuka
langsung di VS Code preview, GitHub, atau dilampirkan ke catatan tesis).

Kenapa cepat: pandas.read_csv(..., nrows=N) dengan engine C berhenti
membaca begitu N baris terkumpul -- tidak parse/scan seluruh file.

CARA PAKAI
----------
    python preview_files.py --data-dir ./data --output preview_report.md
    python preview_files.py --data-dir ./data --rows 10 --output preview_report.md
    python preview_files.py --data-dir ./data --only questions/QuestionsBody_Contain.csv --output preview_qbody.md
    python preview_files.py --data-dir ./data --max-chars 150 --output preview_report.md

Output: satu file .md berisi, untuk tiap CSV yang ditemukan:
  - nama file & ukuran
  - daftar kolom + tipe data hasil inferensi + jumlah nilai kosong di sample
  - tabel Markdown berisi N baris pertama (field panjang dipotong & di-escape
    supaya tidak merusak format tabel)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_csv_files(data_dir: Path, only: str | None):
    if only:
        p = data_dir / only
        return [p] if p.exists() else []
    return sorted(data_dir.rglob("*.csv"))


def escape_md_cell(val, max_chars: int) -> str:
    """Escape nilai supaya aman dimasukkan ke sel tabel Markdown:
    - pipe '|' di-escape (kalau tidak, akan merusak struktur tabel)
    - newline diganti '<br>' (Markdown table tidak boleh multi-baris mentah)
    - dipotong sampai max_chars supaya tabel tetap terbaca
    """
    if pd.isna(val):
        return ""
    s = str(val)
    s = s.replace("|", "\\|")
    s = s.replace("\r\n", " ").replace("\n", "<br>").replace("\r", " ")
    if len(s) > max_chars:
        s = s[:max_chars] + f"…[+{len(s)-max_chars} char]"
    return s


def df_to_markdown_table(df: pd.DataFrame, max_chars: int) -> str:
    """Bangun tabel Markdown manual (tanpa dependency 'tabulate')."""
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        cells = [escape_md_cell(row[c], max_chars) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_file_section(path: Path, n_rows: int, max_chars: int) -> str:
    size_mb = path.stat().st_size / 1e6
    section = [f"## `{path}`", "", f"**Ukuran:** {size_mb:,.1f} MB", ""]

    try:
        df = pd.read_csv(path, nrows=n_rows, low_memory=False)
    except Exception as e:
        section.append(f"[warn] gagal baca dengan encoding default ({e}), mencoba latin-1...")
        try:
            df = pd.read_csv(path, nrows=n_rows, low_memory=False, encoding="latin-1")
        except Exception as e2:
            section.append(f"**[ERROR]** tetap gagal dibaca: {e2}")
            return "\n".join(section) + "\n"

    section.append(f"**Jumlah kolom:** {len(df.columns)}")
    section.append("")
    section.append("**Kolom & tipe data (hasil inferensi dari sample):**")
    section.append("")
    section.append("| Kolom | Tipe | Kosong di sample |")
    section.append("| --- | --- | --- |")
    for col, dtype in df.dtypes.items():
        n_null = df[col].isna().sum()
        section.append(f"| {col} | {dtype} | {n_null}/{len(df)} |")

    section.append("")
    section.append(f"**{min(n_rows, len(df))} baris pertama:**")
    section.append("")
    section.append(df_to_markdown_table(df, max_chars))
    section.append("")
    return "\n".join(section)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../00_datasource/raw",
                         help="Root folder data SORD mentah (default: ../00_datasource/raw, "
                              "relatif terhadap folder 01_data_cleaning/)")
    parser.add_argument("--rows", type=int, default=5, help="Jumlah baris preview per file")
    parser.add_argument("--max-chars", type=int, default=200,
                         help="Maksimum karakter per sel tabel untuk field teks panjang (Body/Text)")
    parser.add_argument("--only", default=None,
                         help="Preview satu file saja, path relatif terhadap --data-dir "
                              "(mis. questions/QuestionsBody_Contain.csv)")
    parser.add_argument("--output", default="preview_report.md",
                         help="Path file Markdown output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Folder {data_dir} tidak ditemukan.")
        sys.exit(1)

    files = find_csv_files(data_dir, args.only)
    if not files:
        print("[ERROR] Tidak ada file CSV ditemukan.")
        sys.exit(1)

    print(f"Ditemukan {len(files)} file CSV. Preview {args.rows} baris pertama tiap file...")

    md_parts = [
        "# Preview Dataset SORD",
        "",
        f"Total file: {len(files)} | Baris preview per file: {args.rows}",
        "",
        "---",
        "",
    ]
    for f in files:
        print(f"  - {f}")
        md_parts.append(build_file_section(f, args.rows, args.max_chars))
        md_parts.append("---")
        md_parts.append("")

    output_path = Path(args.output)
    output_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"\n[OK] Laporan tersimpan -> {output_path.resolve()}")


if __name__ == "__main__":
    sys.exit(main())