"""
8_convert_users_xml.py
========================
Konversi `Users.xml` (dari `stackoverflow.com-Users.7z`, dump StackExchange
LENGKAP -- bukan subset SORD) menjadi `users.parquet` di `00_datasource/merged/`.

Menggantikan `FilteredUsers.csv` sepenuhnya sebagai sumber `reputation` untuk
edge AUTHOR_TRUST di `7_build_knowledge_graph.py` -- dump penuh ini lebih
lengkap/otoritatif daripada subset SORD.

KENAPA PERLU KONVERSI DULU (tidak bisa langsung dipakai)
----------------------------------------------------------
1. DuckDB tidak bisa baca XML native (read_csv_auto/read_parquet cuma untuk
   CSV/Parquet) -- perlu parsing dulu.
2. Users.xml dump PENUH StackOverflow bisa berisi puluhan juta baris. Script
   ini menulis ke Parquet secara INCREMENTAL per chunk (pyarrow ParquetWriter),
   BUKAN accumulate semua baris ke satu list dulu baru ditulis -- supaya
   memory terpakai TETAP KONSTAN (~chunk-size baris) berapa pun ukuran file
   XML aslinya. Ini lebih aman daripada pola accumulate-lalu-tulis yang
   dipakai di 6_postlinks_to_sord.py, karena Users.xml jauh lebih besar
   daripada PostLinks.xml.

Kolom yang diambil: Id, Reputation, DisplayName, CreationDate (cukup untuk
AUTHOR_TRUST + berguna untuk eksplorasi/debug; kolom lain di Users.xml
seperti AboutMe, WebsiteUrl, dst sengaja TIDAK diambil -- tidak dipakai
skema KG saat ini, dan bikin file lebih besar tanpa manfaat).

INSTALL DEPENDENCY
-------------------
    pip install pyarrow

CARA PAKAI
----------
    python 8_convert_users_xml.py
    python 8_convert_users_xml.py --users-xml ../00_datasource/Users.xml --output ../00_datasource/merged/users.parquet
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

PROGRESS_EVERY = 1_000_000


def convert(xml_path: Path, output_path: Path, chunk_size: int) -> None:
    print(f"Sumber : {xml_path} ({xml_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Output : {output_path}")
    print(f"Chunk size: {chunk_size:,} baris per batch tulis")

    schema = pa.schema([
        ("id", pa.int64()),
        ("reputation", pa.int64()),
        ("display_name", pa.string()),
        ("creation_date", pa.string()),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(str(output_path), schema)

    t0 = time.time()
    total_rows = 0
    skipped = 0
    buffer = []

    # Pola sama dengan parsing PostLinks.xml: iterparse start+end + root.remove()
    # supaya elemen yang sudah diproses tidak menumpuk di memori.
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)

    def flush():
        nonlocal buffer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)
        buffer = []

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            continue

        total_rows += 1

        id_raw = elem.get("Id")
        rep_raw = elem.get("Reputation")

        if id_raw is None or rep_raw is None:
            skipped += 1
        else:
            buffer.append({
                "id": int(id_raw),
                "reputation": int(rep_raw),
                "display_name": elem.get("DisplayName"),
                "creation_date": elem.get("CreationDate"),
            })

        elem.clear()
        root.remove(elem)

        if len(buffer) >= chunk_size:
            flush()

        if total_rows % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = total_rows / elapsed if elapsed > 0 else 0
            print(f"  ... {total_rows:,} baris diproses ({rate:,.0f} baris/detik)")

    flush()
    writer.close()

    elapsed = time.time() - t0
    print(f"\nTotal baris User    : {total_rows:,}")
    if skipped:
        print(f"Baris dilewati (atribut hilang): {skipped:,}")
    print(f"Selesai dalam {elapsed:.1f}s")
    print(f"File tersimpan: {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users-xml", default="../00_datasource/Users.xml",
                         help="Path ke Users.xml hasil ekstraksi 7z (default: ../00_datasource/Users.xml)")
    parser.add_argument("--output", default="../00_datasource/merged/users.parquet",
                         help="Path output parquet (default: ../00_datasource/merged/users.parquet)")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                         help="Jumlah baris per batch tulis ke parquet (default: 500000)")
    args = parser.parse_args()

    xml_path = Path(args.users_xml)
    if not xml_path.exists():
        print(f"[ERROR] Users.xml tidak ditemukan di {xml_path}")
        sys.exit(1)

    convert(xml_path, Path(args.output), args.chunk_size)


if __name__ == "__main__":
    sys.exit(main())