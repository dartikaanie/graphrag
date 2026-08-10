"""
test_infra.py
==============
Fase 0.3 — Validasi infrastruktur sebelum lanjut ke Fase 2 (KG construction).
Cek koneksi Neo4j (Desktop) dan fungsi dasar FAISS dengan operasi sederhana.

INSTALL DEPENDENCY
-------------------
    pip install neo4j faiss-cpu sentence-transformers python-dotenv

SETUP .env (di folder yang sama)
----------------------------------
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=yourpassword
    NEO4J_DATABASE=neo4j

    Catatan soal NEO4J_DATABASE: ini nama DATABASE di dalam DBMS Anda (bukan
    nama DBMS/project di Neo4j Desktop). Defaultnya "neo4j" kalau Anda tidak
    membuat database baru secara eksplisit. Cek di Neo4j Browser dengan
    perintah `SHOW DATABASES;` kalau tidak yakin.

    Catatan untuk Neo4j DESKTOP: cek port Bolt yang sebenarnya di detail
    DBMS Anda (klik nama DBMS -> "Connection details" / info koneksi) --
    Neo4j Desktop kadang assign port lain dari default 7687 kalau ada
    konflik (mis. sudah ada instance lain jalan). Sesuaikan NEO4J_URI kalau
    beda.

CARA PAKAI
----------
    python test_infra.py
"""

import argparse
import os
import subprocess
import sys

# HARUS di atas sebelum import faiss/torch/sentence-transformers: memperbaiki
# crash "OMP: Error #15" yang umum terjadi di macOS ketika FAISS dan
# PyTorch/sentence-transformers sama-sama membawa runtime OpenMP sendiri dan
# bentrok saat di-load di proses yang sama. Workaround resmi dari OpenMP
# project (tidak ideal, tapi aman untuk kasus testing/development ini).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Matikan parallelism internal Rust tokenizer (dependency sentence-transformers)
# -- ini penyebab umum lain dari deadlock/crash native saat berbarengan
# dengan library multi-thread lain (FAISS/OpenMP) di proses yang sama.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# RAYON adalah thread pool Rust yang dipakai library 'tokenizers' -- ini
# thread pool TERPISAH dari OpenMP, jadi perlu dibatasi secara eksplisit
# juga. Konflik antara Rayon (tokenizers) dan thread pool internal PyTorch
# adalah penyebab umum crash "mutex lock failed" di macOS Apple Silicon.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Matikan hf_transfer (downloader Rust berbasis async runtime terpisah lagi)
# kalau kebetulan terinstall -- sumber crash native lain yang mirip.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Paksa transformers HANYA pakai backend PyTorch, jangan coba auto-detect
# atau import TensorFlow/JAX sama sekali. Kalau TensorFlow kebetulan
# ter-install di environment ini, proses auto-detect backend transformers
# bisa memicu import TF di belakang layar -- dan TF membawa bundel library
# native sendiri (termasuk Abseil, sumber error 'mutex.cc' yang muncul
# sebelumnya) yang gampang bentrok dengan runtime lain di proses yang sama.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from dotenv import load_dotenv

load_dotenv()

ENV_PATH_HINT = "(.env tidak ditemukan atau kosong -- pastikan file .env ada di folder yang sama dengan skrip ini)"


def test_neo4j():
    print("\n=== Test Neo4j (Desktop) ===")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[FAIL] Package 'neo4j' belum terinstall. Jalankan: pip install neo4j")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    # Bersihkan kesalahan umum: tanda kutip ikut kebawa dari .env, atau
    # trailing whitespace yang bikin password mismatch tanpa terlihat.
    if password:
        original_len = len(password)
        password = password.strip().strip('"').strip("'")
        if len(password) != original_len:
            print(f"    [warn] password di .env terdeteksi ada whitespace/kutip "
                  f"yang dibersihkan otomatis ({original_len} -> {len(password)} karakter). "
                  f"Sebaiknya edit .env supaya bersih dari awal.")

    print(f"    URI      : {uri}")
    print(f"    User     : {user}")
    print(f"    Database : {database}")
    print(f"    Pass     : {'*' * len(password) + f' ({len(password)} karakter)' if password else '(kosong! ' + ENV_PATH_HINT + ')'}")

    if not password:
        print("[FAIL] NEO4J_PASSWORD tidak ditemukan di .env")
        return False

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"[OK] Terhubung ke Neo4j di {uri}")

        with driver.session(database=database) as session:
            # Info versi/edition -- berguna untuk dokumentasi reprodusibilitas
            # di Bab IV ("environment implementasi"). Pakai .data() bukan
            # .single() karena dbms.components() bisa kembalikan >1 baris
            # (mis. Neo4j Kernel + plugin seperti APOC).
            comp_rows = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition"
            ).data()
            if comp_rows:
                comp = comp_rows[0]
                print(f"[OK] {comp['name']} versi {comp['versions'][0]} ({comp['edition']})")

            # Test tulis & baca sederhana
            session.run(
                "MERGE (n:InfraTest {id: 'ping'}) SET n.timestamp = timestamp()"
            )
            result = session.run(
                "MATCH (n:InfraTest {id: 'ping'}) RETURN n.timestamp AS ts"
            )
            record = result.single()
            print(f"[OK] Write+read test berhasil di database '{database}', timestamp: {record['ts']}")

            # Bersihkan node test
            session.run("MATCH (n:InfraTest {id: 'ping'}) DELETE n")
            print("[OK] Cleanup node test berhasil")

        driver.close()
        return True

    except Exception as e:
        print(f"[FAIL] Gagal terhubung/operasi ke Neo4j: {e}")
        print("       Troubleshooting untuk Neo4j DESKTOP:")
        print("       1. Cek DBMS berstatus 'Active' (hijau) di Neo4j Desktop, bukan cuma dibuat")
        print("       2. Cek port Bolt sebenarnya: klik nama DBMS -> lihat 'Connection details'")
        print("          (kadang bukan 7687 kalau ada instance lain yang sudah pakai port itu)")
        print("       3. Cek NEO4J_PASSWORD di .env sama persis dengan password saat DBMS dibuat")
        print("       4. Kalau baru ubah dbms.memory.* di Settings, pastikan sudah restart DBMS "
              "(Stop lalu Start ulang, bukan cuma Apply)")
        print("       5. Cek NEO4J_DATABASE benar -- di Neo4j Browser jalankan `SHOW DATABASES;` "
              "untuk lihat daftar database yang ada dan mana yang 'online'")
        return False


def test_faiss():
    print("\n=== Test FAISS ===")
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("[FAIL] Package 'faiss-cpu' atau 'numpy' belum terinstall.")
        return False

    try:
        dim = 384  # dimensi umum untuk all-MiniLM-L6-v2
        n_vectors = 100

        index = faiss.IndexFlatL2(dim)
        vectors = np.random.random((n_vectors, dim)).astype("float32")
        index.add(vectors)
        print(f"[OK] FAISS index dibuat & diisi {n_vectors} vektor dummy (dim={dim})")

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)
        assert indices[0][0] == 0, "Vektor pertama harus jadi hasil pencarian terdekat dari dirinya sendiri"
        print(f"[OK] Search test berhasil, top-5 nearest indices: {indices[0].tolist()}")
        return True

    except Exception as e:
        print(f"[FAIL] FAISS test gagal: {e}")
        return False


def test_sentence_transformers():
    print("\n=== Test Sentence-Transformers (untuk embedding) ===")
    try:
        print("  [1/4] Importing torch...")
        import torch
        torch.set_num_threads(1)  # cegah thread pool internal torch bentrok dgn Rayon (tokenizers)
        print(f"  [OK] torch {torch.__version__} di-import, single-threaded mode aktif")
    except ImportError:
        print("[FAIL] Package 'torch' belum terinstall.")
        return False
    except Exception as e:
        print(f"[FAIL] Crash saat import/setup torch: {e}")
        return False

    try:
        print("  [2/4] Importing SentenceTransformer class...")
        from sentence_transformers import SentenceTransformer
        print("  [OK] Import berhasil")
    except ImportError:
        print("[FAIL] Package 'sentence-transformers' belum terinstall.")
        return False

    try:
        print("  [3/4] Load/download model all-MiniLM-L6-v2 (pertama kali unduh dari "
              "HuggingFace Hub, bisa agak lama; titik crash paling mungkin ada di sini)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [OK] Model berhasil di-load")

        print("  [4/4] Encoding kalimat uji coba...")
        emb = model.encode(["contoh kalimat untuk uji coba embedding"])
        print(f"[OK] Model berhasil load & encode, dimensi embedding: {emb.shape[1]}")
        return True
    except Exception as e:
        print(f"[FAIL] Gagal load/encode model: {e}")
        return False


def run_isolated_check(check_name: str, timeout_sec: int = 180) -> bool:
    """Jalankan satu jenis test (faiss / sentence_transformers) di proses
    Python TERPISAH, supaya thread pool native library-nya (OpenMP, Rust
    tokenizers, dst.) tidak saling bentrok dengan test lain yang sudah
    jalan di proses utama. Ini yang memperbaiki deadlock 'mutex.cc RAW:
    Lock blocking' yang muncul waktu FAISS dan Sentence-Transformers
    di-load berurutan di satu proses yang sama.

    Output di-stream REAL-TIME (bukan ditahan sampai proses selesai) --
    penting terutama untuk test sentence_transformers yang pertama kali
    jalan perlu download model dari HuggingFace Hub, supaya progress
    kelihatan dan tidak seperti macet."""
    proc = subprocess.Popen(
        [sys.executable, "-u", __file__, "--run-check", check_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    try:
        for line in proc.stdout:
            print(line, end="")
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[FAIL] Proses '{check_name}' melebihi timeout {timeout_sec}s, "
              f"dipaksa berhenti. Kemungkinan macet karena unduhan model "
              f"terlalu lambat/stuck -- cek koneksi internet, atau coba lagi.")
        return False

    if proc.returncode != 0:
        print(f"[FAIL] Proses terpisah untuk '{check_name}' keluar dengan error "
              f"(exit code {proc.returncode}) -- kemungkinan crash native library")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-check", choices=["faiss", "sentence_transformers"],
                         default=None,
                         help=argparse.SUPPRESS)  # dipakai internal untuk subprocess isolation
    args = parser.parse_args()

    # Mode internal: dipanggil ulang oleh diri sendiri sebagai subprocess
    # terisolasi untuk satu jenis check saja.
    if args.run_check == "faiss":
        sys.exit(0 if test_faiss() else 1)
    if args.run_check == "sentence_transformers":
        sys.exit(0 if test_sentence_transformers() else 1)

    print("=" * 70)
    print("VALIDASI INFRASTRUKTUR — FASE 0.3")
    print("=" * 70)

    if not os.path.exists(".env"):
        print("\n[WARN] File .env tidak ditemukan di folder ini. Skrip tetap jalan "
              "dengan default (bolt://localhost:7687, user=neo4j), tapi kemungkinan "
              "besar akan gagal kalau password belum diset lewat .env.")
        print("       Buat file .env di folder yang sama berisi:")
        print("       NEO4J_URI=bolt://localhost:7687")
        print("       NEO4J_USER=neo4j")
        print("       NEO4J_PASSWORD=password_anda")
        print("       NEO4J_DATABASE=neo4j")

    results = {
        "Neo4j": test_neo4j(),
        "FAISS": run_isolated_check("faiss"),
        "Sentence-Transformers": run_isolated_check("sentence_transformers"),
    }

    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    if all(results.values()):
        print("\n[SUCCESS] Semua infrastruktur siap. Lanjut ke Fase 2 (KG construction).")
    else:
        print("\n[ACTION] Ada komponen yang gagal — perbaiki dulu sebelum lanjut.")
        sys.exit(1)


if __name__ == "__main__":
    main()