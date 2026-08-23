"""
preview_files.py
=================
Preview cepat isi tiap file CSV di dataset SORD

LOKASI OUTPUT 
---------------------
    <repo_root>/report/1_preview_report.md
    <repo_root>/report/img/*.png   (chart pendukung)
Override lokasi ini kapan saja lewat --output / --img-dir.

KALAU LAPORAN SUDAH ADA
-------------------------
Default-nya: kalau laporan sudah ada, LANGSUNG TAMPILKAN isi laporan lama
gunakan --force kalau memang mau regenerate (mis. setelah data sumber berubah).

COMMAND
----------
    python preview_files.py                                   # semua default
    python preview_files.py --data-dir ./data --output preview_report.md
    python preview_files.py --rows 10
    python preview_files.py --only questions/QuestionsBody_Contain.csv
    python preview_files.py --max-chars 150
    python preview_files.py --force                            # overwrite

Output: report dalam format markdown. 
  
DEPENDENCY
-----------------------------
    pip install matplotlib
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Root repo = dua level di atas file ini (.../01_data_cleaning/1_preview_files.py -> repo root)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


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


def build_file_section(path: Path, n_rows: int, max_chars: int, size_mb: float):
    """Return (markdown_section, df_or_None, n_columns_or_None)."""
    section = [f"## `{path}`", "", f"**Ukuran:** {size_mb:,.1f} MB", ""]

    try:
        df = pd.read_csv(path, nrows=n_rows, low_memory=False)
    except Exception as e:
        section.append(f"[warn] gagal baca dengan encoding default ({e}), mencoba latin-1...")
        try:
            df = pd.read_csv(path, nrows=n_rows, low_memory=False, encoding="latin-1")
        except Exception as e2:
            section.append(f"**[ERROR]** tetap gagal dibaca: {e2}")
            return "\n".join(section) + "\n", None, None

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
    return "\n".join(section), df, len(df.columns)


def generate_charts(file_stats: list[dict], img_dir: Path) -> list[str]:
    """Bangun chart ringkasan datasource dari file_stats
    (list of {"name": str, "size_mb": float, "n_columns": int}).
    Return list nama file gambar yang berhasil dibuat (relatif ke img_dir)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, aman dijalankan tanpa display
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib tidak terinstall -- chart dilewati. "
              "Install dengan: pip install matplotlib")
        return []

    img_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    names = [s["name"] for s in file_stats]
    sizes = [s["size_mb"] for s in file_stats]
    ncols = [s["n_columns"] for s in file_stats if s["n_columns"] is not None]
    ncols_names = [s["name"] for s in file_stats if s["n_columns"] is not None]

    # --- Chart 1: ukuran file (MB) per file, urut menurun ---
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    sorted_names = [names[i] for i in order]
    sorted_sizes = [sizes[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(sorted_names))))
    ax.barh(sorted_names, sorted_sizes, color="#4C72B0")
    ax.set_xlabel("Ukuran file (MB)")
    ax.set_title("Ukuran File CSV Dataset SORD")
    ax.invert_yaxis()
    plt.tight_layout()
    fname1 = "file_sizes.png"
    fig.savefig(img_dir / fname1, dpi=150)
    plt.close(fig)
    generated.append(fname1)

    # --- Chart 2: jumlah kolom per file ---
    if ncols_names:
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(ncols_names))))
        ax.barh(ncols_names, ncols, color="#55A868")
        ax.set_xlabel("Jumlah kolom")
        ax.set_title("Jumlah Kolom per File CSV")
        ax.invert_yaxis()
        plt.tight_layout()
        fname2 = "column_counts.png"
        fig.savefig(img_dir / fname2, dpi=150)
        plt.close(fig)
        generated.append(fname2)

    return generated


def show_existing_report(output_path: Path):
    """Laporan sudah ada -> langsung tampilkan isinya, TIDAK proses ulang."""
    print(f"\n[i] Laporan sudah ada di -> {output_path.resolve()}")
    print("[i] Menampilkan laporan yang ada (tidak proses ulang). "
          "Pakai --force untuk regenerate.\n")
    print("-" * 78)
    print(output_path.read_text(encoding="utf-8"))
    print("-" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="../00_datasource/raw",
                         help="Root folder data SORD mentah (default: ../00_datasource/raw, "
                              "relatif terhadap folder 01_data_cleaning/)")
    parser.add_argument("--rows", type=int, default=5, help="Jumlah baris preview per file")
    parser.add_argument("--max-chars", type=int, default=200,
                         help="Maksimum karakter per sel tabel untuk field teks panjang (Body/Text)")
    parser.add_argument("--only", default=None,
                         help="Preview satu file saja, path relatif terhadap --data-dir "
                              "(mis. questions/QuestionsBody_Contain.csv)")
    parser.add_argument("--output", default=None,
                         help="Path file Markdown output. Default: <repo_root>/report/1_preview_report.md")
    parser.add_argument("--img-dir", default=None,
                         help="Folder untuk menyimpan chart .png. Default: <repo_root>/report/img")
    parser.add_argument("--no-charts", action="store_true",
                         help="Lewati pembuatan chart (hanya generate tabel Markdown)")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate laporan walau sudah ada (default: kalau sudah ada, "
                              "langsung tampilkan laporan lama tanpa proses ulang)")
    args = parser.parse_args()

    # --- Lokasi output: default SELALU relatif ke repo root, bukan cwd ---
    output_path = Path(args.output) if args.output else (REPO_ROOT / "report" / "1_preview_report.md")
    img_dir = Path(args.img_dir) if args.img_dir else (REPO_ROOT / "report" / "img")

    # --- Kalau laporan sudah ada dan tidak --force: langsung tampilkan, JANGAN proses ulang ---
    if output_path.exists() and not args.force:
        show_existing_report(output_path)
        return

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
    ]

    file_stats = []
    file_sections = []
    for f in files:
        print(f"  - {f}")
        size_mb = f.stat().st_size / 1e6
        section_md, _df, n_columns = build_file_section(f, args.rows, args.max_chars, size_mb)
        file_sections.append(section_md)
        file_stats.append({"name": f.name, "size_mb": size_mb, "n_columns": n_columns})

    # --- Chart ringkasan datasource (di atas, sebelum detail per-file) ---
    if not args.no_charts:
        print(f"\nMembuat chart ringkasan datasource -> {img_dir}")
        chart_files = generate_charts(file_stats, img_dir)
        if chart_files:
            md_parts.append("## Ringkasan Visual Datasource")
            md_parts.append("")
            for cf in chart_files:
                # path relatif dari file .md ke folder img (img/ ada di folder yang sama)
                rel = f"img/{cf}"
                md_parts.append(f"![{cf}]({rel})")
                md_parts.append("")
            md_parts.append("---")
            md_parts.append("")

    md_parts.append("---")
    md_parts.append("")
    for section_md in file_sections:
        md_parts.append(section_md)
        md_parts.append("---")
        md_parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"\n[OK] Laporan tersimpan -> {output_path.resolve()}")


if __name__ == "__main__":
    sys.exit(main())