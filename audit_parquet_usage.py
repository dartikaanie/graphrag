#!/usr/bin/env python3
"""
Audit .parquet usage across the repo.

Cara pakai:
    cd ~/path/ke/repo/graphrag
    python audit_parquet_usage.py

Yang dilakukan:
1. Cari semua file .parquet di bawah root repo (termasuk _kg_workspace/, dsb.)
2. Cari semua file .py di repo
3. Untuk tiap parquet, cek apakah nama filenya (atau stem-nya) disebut
   di baris kode manapun -> dianggap "referenced"
4. Print laporan: dipakai vs tidak dipakai, plus ukuran file (MB)

CATATAN PENTING (baca sebelum hapus apapun):
- Ini pencarian teks sederhana (substring match), bukan analisis AST.
  Kalau ada script yang membentuk nama file secara dinamis
  (misal glob("*.parquet") atau f"condition_{cond}_{n}.parquet"),
  file itu bisa saja tetap dipakai walau tidak "match" persis di sini.
- File yang muncul di "UNUSED" perlu dicek manual dulu:
    a) apakah dia OUTPUT akhir suatu tahap pipeline yang sudah selesai
       (misal output densifikasi final -> memang tidak perlu direferensikan
       lagi oleh script lain, tapi tetap dipakai sebagai INPUT Neo4j load)
    b) apakah dia hasil percobaan lama / parameter tuning sebelumnya
       (misal versi sebelum --max-df di-tuning) -> ini kandidat aman dihapus
    c) apakah dia file debug/sample kecil dari testing -> kandidat aman dihapus
"""

import os
from pathlib import Path

REPO_ROOT = Path(".").resolve()  # jalankan dari root repo

EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules"}


def find_files(root, suffix):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith(suffix):
                found.append(Path(dirpath) / f)
    return found


def load_text_cache(py_files):
    cache = {}
    for pf in py_files:
        try:
            cache[pf] = pf.read_text(errors="ignore").splitlines()
        except Exception:
            continue
    return cache


def main():
    parquet_files = find_files(REPO_ROOT, ".parquet")
    py_files = find_files(REPO_ROOT, ".py")
    text_cache = load_text_cache(py_files)

    print(f"Ditemukan {len(parquet_files)} file .parquet")
    print(f"Memindai {len(py_files)} file .py untuk referensi...\n")

    results = []
    for pq in parquet_files:
        name, stem = pq.name, pq.stem
        hits = []
        for pf, lines in text_cache.items():
            for i, line in enumerate(lines, 1):
                if name in line or stem in line:
                    hits.append((pf, i, line.strip()))
        results.append((pq, hits))

    used = [r for r in results if r[1]]
    unused = [r for r in results if not r[1]]

    print("=" * 80)
    print(f"DIPAKAI ({len(used)} file):")
    print("=" * 80)
    for pq, hits in sorted(used, key=lambda r: -r[0].stat().st_size):
        size_mb = pq.stat().st_size / (1024 * 1024)
        print(f"\n📄 {pq.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")
        for pf, ln, line in hits[:3]:
            print(f"   ↳ {pf.relative_to(REPO_ROOT)}:{ln}: {line[:100]}")
        if len(hits) > 3:
            print(f"   ... +{len(hits) - 3} referensi lain")

    print("\n" + "=" * 80)
    print(f"⚠️  TIDAK ADA REFERENSI TERDETEKSI ({len(unused)} file):")
    print("=" * 80)
    total_unused_mb = 0.0
    for pq, _ in sorted(unused, key=lambda r: -r[0].stat().st_size):
        size_mb = pq.stat().st_size / (1024 * 1024)
        total_unused_mb += size_mb
        print(f"   {pq.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")

    print(f"\nTotal ukuran file 'unused': {total_unused_mb:.1f} MB")
    print("\n⚠️  Cek manual dulu (lihat catatan di docstring) sebelum menghapus.")
   

if __name__ == "__main__":
    main()