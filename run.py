#!/usr/bin/env python3
"""
run.py — Launcher 
===========================================================

CARA PAKAI
----------
    python run.py

Lalu pilih nomor aksi dari menu.

Untuk aksi yang MEMANGGIL LLM (test koneksi, Kondisi A/B/C) akan ditanya:
    "Gunakan argumen default? [Y/n]"
  - Y / Enter -> jalankan dengan argumen default (sesuai .env / dokumentasi script)
  - n         -> kamu bisa ketik argumen tambahan/override secara bebas,
                 contoh: --n-sample 10 --provider ollama --model qwen2.5:1.5b
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Definisi menu: (nomor tampil otomatis dari urutan list)
# Setiap entri: key, label, deskripsi singkat, script (relatif ke repo root),
# working_dir (relatif ke repo root -- tempat script dijalankan),
# default_args (list[str]), catatan (opsional, mis. warning biaya API)
# ---------------------------------------------------------------------
MENU = [
    {
        "label": "Test koneksi LLM",
        "desc": "Kirim beberapa prompt uji singkat untuk memastikan provider & model bisa diakses.",
        "script": "config/test_llm_connection.py",
        "cwd": ".",
        "default_args": [],
        "note": "Pakai provider/model dari .env di root. Override cepat: --provider ollama --model qwen2.5:1.5b",
    },
    {
        "label": "Preview file data SORD",
        "desc": "Preview cepat isi tiap file CSV mentah SORD tanpa load penuh.",
        "script": "1_preview_files.py",
        "cwd": "01_data_cleaning",
        "default_args": ["--data-dir", "../00_datasource/raw"],
        "note": "Output & chart otomatis tersimpan ke <repo_root>/report/ (path fix, tidak tergantung cwd). "
                "Kalau report sudah ada, langsung ditampilkan tanpa proses ulang (pakai --force di script untuk regenerate).",
        "skip_arg_confirm": True,
    },
    {
        "label": "Cek integritas data SORD (Fase 0.1)",
        "desc": "Validasi ukuran file, jumlah baris vs paper SORD, header kolom, duplikat Id.",
        "script": "2_check_data_integrity.py",
        "cwd": "01_data_cleaning",
        "default_args": ["--data-dir", "../00_datasource/raw"],
        "note": "Tambahkan --quick kalau mau skip row-count exact (lebih cepat untuk file besar). "
                "Kalau report sudah ada, langsung ditampilkan tanpa proses ulang (pakai --force di script untuk regenerate).",
        "skip_arg_confirm": True,
    },
    {
        "label": "Merge sumber SORD -> Parquet (Fase 0.2)",
        "desc": "Gabungkan Contain + LikeMinusContain per kategori jadi *_raw_union.parquet.",
        "script": "3_merge_sord_sources.py",
        "cwd": "01_data_cleaning",
        "default_args": ["--data-dir", "../00_datasource/raw", "--output-dir", "../00_datasource/merged"],
        "note": None,
        "skip_arg_confirm": True,
    },
    {
        "label": "EDA sinyal kepercayaan & karakteristik datasource (19 chart)",
        "desc": "Grup A-E: kualitas data, konten/topik, sinyal kepercayaan komunitas, dimensi waktu, kelayakan KG.",
        "script": "5_eda_trust_signals.py",
        "cwd": "01_data_cleaning",
        "default_args": ["--merged-dir", "../00_datasource/merged", "--raw-dir", "../00_datasource/raw"],
        "note": "Butuh Parquet hasil merge (Fase 0.2). Tambah --skip-heavy kalau mau lewati "
                "FilteredVotes/FilteredBadges (puluhan juta baris). Report sudah ada -> langsung "
                "ditampilkan, tidak proses ulang (pakai --force untuk regenerate).",
        "skip_arg_confirm": True,
    },
    {
        "label": "Test infrastruktur Neo4j & FAISS (Fase 0.3)",
        "desc": "Validasi koneksi Neo4j Desktop dan operasi dasar FAISS sebelum lanjut ke KG construction.",
        "script": "4_test_infra.py",
        "cwd": "01_data_cleaning",
        "default_args": [],
        "note": "Butuh .env berisi NEO4J_URI/USER/PASSWORD/DATABASE di 01_data_cleaning/.",
    },
    {
        "label": "Cek match accepted answer (Question <-> Answer parquet)",
        "desc": "Cek rasio AcceptedAnswerId yang TIDAK ketemu di Answers parquet -- diagnostik sebelum sampling.",
        "script": "config/check_accepted_answer_match.py",
        "cwd": "02_baseline_replication",
        "default_args": [],
        "note": None,
        "skip_arg_confirm": True,
    },
    {
        "label": "Analisis kelayakan Knowledge Graph dari subset matched",
        "desc": "Distribusi tag, comment density, distribusi Score -- menilai kelayakan subset sebagai basis KG.",
        "script": "config/analyze_kg_feasibility.py",
        "cwd": "02_baseline_replication",
        "default_args": [],
        "note": None,
        "skip_arg_confirm": True,
    },
    {
        "label": "Stratified sampling benchmark (replikasi pola Kabir/Da Silva)",
        "desc": "Sampling benchmark uji berdasarkan popularitas, tipe pertanyaan, dan recency.",
        "script": "stratified_sampling.py",
        "cwd": ".",
        "default_args": [
            "--input", "00_datasource/raw/questions/QuestionsBody_Contain.csv",
            "--output", "00_datasource/benchmark_sample.csv",
        ],
        "note": "--input dan --output WAJIB diisi -- sesuaikan path kalau berbeda dari default di atas.",
        "skip_arg_confirm": True,
    },
    {
        "label": "Run Kondisi A -- Baseline LLM murni (replikasi Da Silva dkk. 2025)",
        "desc": "Sampling 384 pertanyaan -> prompt LLM 1x per pertanyaan -> cosine similarity vs accepted answer.",
        "script": "a_baseline_replecation.py",
        "cwd": "02_baseline_replication",
        "default_args": [],
        "note": "",
        "confirm_cost": True,
    },
    {
        "label": "Run Kondisi B -- RAG Konvensional (dense retrieval)",
        "desc": "Sama seperti Kondisi A + retrieval FAISS top-k sebagai konteks tambahan sebelum prompting.",
        "script": "b_condition_b_rag.py",
        "cwd": "03_rag",
        "default_args": [],
        "note": "",
        "confirm_cost": True,
    },
    {
        "label": "Run Kondisi C -- GraphRAG Framework  [BELUM TERSEDIA]",
        "desc": "05_condition_c_graphrag/ belum dibangun di repo ini.",
        "script": None,
        "cwd": None,
        "default_args": [],
        "note": None,
    },
]


def print_menu():
    print("\n" + "=" * 70)
    print("GraphRAG THESIS PIPELINE -- LAUNCHER")
    print("=" * 70)
    for i, item in enumerate(MENU, 1):
        print(f"  {i:>2}. {item['label']}")
    print("   0. Keluar")
    print("=" * 70)


def choose_action():
    while True:
        raw = input("\nPilih nomor aksi: ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(MENU):
            return MENU[int(raw) - 1]
        print("[!] Input tidak valid, coba lagi.")


def confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default_yes
    return raw in ("y", "yes")


def run_action(item: dict):
    if item["script"] is None:
        print(f"\n[i] '{item['label']}' belum tersedia di repo ini. Lewati.")
        return

    print("\n" + "-" * 70)
    print(f"AKSI : {item['label']}")
    print(f"       {item['desc']}")
    if item.get("note"):
        print(f"[catatan] {item['note']}")
    print("-" * 70)

    if item.get("confirm_cost"):
        ok = confirm(
            "[!] Aksi ini bisa memanggil API LLM berbayar tergantung provider di .env. Lanjutkan?",
            default_yes=False,
        )
        if not ok:
            print("[i] Dibatalkan.")
            return

    if item.get("skip_arg_confirm"):
        # Aksi seputar datasource -- langsung jalan dengan default, tanpa tanya apa pun.
        # Script terkait sudah punya mekanisme skip-kalau-report-sudah-ada sendiri.
        extra_args = item["default_args"]
        print(f"[i] Menjalankan langsung dengan argumen default: {' '.join(extra_args) or '(tidak ada)'}")
    else:
        use_default = confirm("Gunakan argumen default?", default_yes=True)
        if use_default:
            extra_args = item["default_args"]
            print(f"[i] Menjalankan dengan argumen default: {' '.join(extra_args) or '(tidak ada)'}")
        else:
            print("[i] Ketik argumen tambahan/override (contoh: --n-sample 10 --provider ollama).")
            print(f"    Argumen default untuk aksi ini (referensi): {' '.join(item['default_args']) or '(tidak ada)'}")
            raw = input("    Argumen custom: ").strip()
            extra_args = raw.split() if raw else []

    script_path = REPO_ROOT / item["cwd"] / item["script"]
    if not script_path.exists():
        print(f"[ERROR] Script tidak ditemukan: {script_path}")
        return

    cmd = [sys.executable, item["script"]] + extra_args
    cwd_path = REPO_ROOT / item["cwd"]

    print(f"\n>>> Menjalankan di '{cwd_path}':")
    print(f">>> {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, cwd=str(cwd_path), check=False)
    except KeyboardInterrupt:
        print("\n[i] Dihentikan oleh user (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] Gagal menjalankan script: {e}")


def main():
    print("Selamat datang di launcher pipeline GraphRAG Thesis.")
    print(f"Repo root terdeteksi: {REPO_ROOT}")
    while True:
        print_menu()
        item = choose_action()
        if item is None:
            print("\nSampai jumpa!")
            break
        run_action(item)


if __name__ == "__main__":
    sys.exit(main())