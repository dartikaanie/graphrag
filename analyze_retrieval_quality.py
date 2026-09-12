#!/usr/bin/env python3
"""
analyze_retrieval_quality.py
=====================================
Metrik retrieval post-hoc -- Precision@k dan MRR@k -- dihitung MURNI dari
file JSONL hasil Kondisi B/C/D yang SUDAH ADA (sudah selesai dijalankan).
TIDAK ADA panggilan LLM baru, TIDAK ADA biaya API tambahan -- script ini
hanya membaca `retrieved_context` yang sudah dipersist di setiap record
dan membandingkannya dengan tag pertanyaan sumbernya (DuckDB, sekali query
batch). Kondisi A dilewati otomatis (tidak melakukan retrieval sama sekali).

DEFINISI RELEVANSI -- PROXY, BUKAN GROUND TRUTH MANUSIA
------------------------------------------------------------
Tidak ada label relevansi manual untuk retrieval di pilot data ini. Sebagai
proxy, satu item `retrieved_context` dianggap "relevan" terhadap pertanyaan
evaluasi kalau overlap tag (Jaccard) antara tags pertanyaan evaluasi dan
tags pertanyaan SUMBER item itu > --tag-overlap-threshold (default 0.0,
artinya minimal 1 tag yang sama sudah dianggap relevan). INI ADALAH PROXY
KASAR, bukan penilaian relevansi manusia yang sesungguhnya -- dua
pertanyaan bisa berbagi tag tapi secara substansi tidak relevan, atau
sebaliknya relevan secara substansi tanpa tag yang identik. Keterbatasan
metodologis ini WAJIB disebutkan di Bab V sebagai batasan evaluasi, bukan
diklaim sebagai pengukuran presisi retrieval yang sesungguhnya.

KENAPA TIDAK ADA RECALL@k DI SINI (JUJUR, WAJIB DIBACA)
------------------------------------------------------------
Recall@k butuh tahu TOTAL item relevan yang TERSEDIA sebelum dipotong ke
top-k (bukan cuma yang lolos ke dalam top-k) -- data pilot Kondisi B/C/D
yang SUDAH ADA saat ini hanya mencatat JUMLAH kandidat sebelum fusi
(n_graph_candidates/n_expansion_candidates dst.), BUKAN daftar
question_id-nya. Tanpa daftar itu, tidak mungkin menghitung berapa banyak
dari SELURUH kandidat yang sebenarnya relevan (proxy tag-overlap) --
Recall@k TIDAK BISA dihitung secara valid untuk run yang sudah ada. Sebagai
solusi ke depan (BUKAN diimplementasikan di script ini), Kondisi B/C
punya flag opt-in baru `--log-full-candidates` yang menyimpan
`all_candidate_question_ids` (daftar lengkap SEBELUM top-k cutoff) ke
record JSONL -- run BERIKUTNYA yang memakai flag ini bisa dihitung
Recall@k sesungguhnya di versi lanjutan script ini.

LANGKAH
-------
1. load_question_tags() -- satu query DuckDB batch untuk semua question_id
   yang muncul di retrieved_context seluruh file input (BUKAN per-baris).
2. jaccard_tag_overlap() -- overlap Jaccard dua set tag.
3. compute_precision_at_k() / compute_mrr_at_k() -- per record, urutan
   ranking retrieved_context APA ADANYA (tidak diurutkan ulang).
4. Agregasi mean/median per file, cetak tabel ringkasan (gaya tampilan
   mirip compare_condition_c_runs.py).
5. Detail per-pertanyaan disimpan ke retrieval_quality_{nama_file_asli}.csv
   untuk lampiran apendiks tesis.

CARA PAKAI
----------
    python analyze_retrieval_quality.py --input-path llm/c_graphrag/results/condition_c_openai_gpt-4o-mini_n30_seed42_fw0-7-0-3.jsonl
    python analyze_retrieval_quality.py --input-glob "llm/*/results/condition_*.jsonl"
    python analyze_retrieval_quality.py --input-path <file> --tag-overlap-threshold 0.1

Requires QUESTIONS_PARQUET (root .env, sama seperti kondisi B/C/D) untuk
melihat tags pertanyaan sumber retrieved_context.
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

TAG_RE = re.compile(r"<([^<>]+)>")

# Kondisi A tidak melakukan retrieval sama sekali -- selalu dilewati.
RETRIEVAL_CONDITIONS = ("B", "C", "D")


def parse_tags(tags_str) -> set[str]:
    """Parse format tag SO ("<tag-a><tag-b>") jadi set string, dipakai
    baik untuk tags pertanyaan evaluasi (sudah ada di record JSONL) maupun
    tags pertanyaan sumber (di-query dari Questions parquet)."""
    if not tags_str:
        return set()
    return set(TAG_RE.findall(str(tags_str)))


def load_question_tags(con, questions_parquet: str, question_ids: list[int]) -> dict[int, set[str]]:
    """Satu query DuckDB batch untuk tags semua question_id yang muncul di
    retrieved_context SELURUH file input -- bukan per-baris, supaya tidak
    membuka ulang parquet ratusan kali untuk satu file hasil."""
    if not question_ids:
        return {}
    ids_str = ",".join(str(int(i)) for i in set(question_ids))
    query = f"""
        SELECT Id, Tags
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY Id ORDER BY match_source) AS rn
            FROM read_parquet('{questions_parquet}')
            WHERE Id IN ({ids_str})
        )
        WHERE rn = 1
    """
    df = con.execute(query).df()
    return {int(row.Id): parse_tags(row.Tags) for row in df.itertuples()}


def jaccard_tag_overlap(tags_a: set[str], tags_b: set[str]) -> float:
    if not tags_a or not tags_b:
        return 0.0
    intersection = len(tags_a & tags_b)
    union = len(tags_a | tags_b)
    return intersection / union if union else 0.0


def compute_precision_at_k(record: dict, eval_tags: set[str], source_tags_map: dict[int, set[str]],
                            tag_overlap_threshold: float) -> tuple[float, int] | None:
    """Precision@k = (# item retrieved yang relevan) / (# item retrieved
    AKTUAL -- bisa < top_k kalau retrieval sebagian kosong untuk
    pertanyaan ini). Urutan retrieved_context TIDAK diurutkan ulang --
    dipakai apa adanya sesuai urutan ranking asli generator. Return None
    kalau retrieved_context kosong (tidak ada dasar untuk precision)."""
    retrieved = record.get("retrieved_context") or []
    if not retrieved:
        return None
    relevant_flags = []
    for item in retrieved:
        qid = item.get("question_id")
        source_tags = source_tags_map.get(qid, set()) if qid is not None else set()
        overlap = jaccard_tag_overlap(eval_tags, source_tags)
        relevant_flags.append(overlap > tag_overlap_threshold)
    n = len(relevant_flags)
    return sum(relevant_flags) / n, n


def compute_mrr_at_k(record: dict, eval_tags: set[str], source_tags_map: dict[int, set[str]],
                      tag_overlap_threshold: float) -> float | None:
    """1/rank item relevan PERTAMA (rank mulai dari 1), 0.0 kalau tidak
    ada yang relevan, None kalau retrieved_context kosong sama sekali."""
    retrieved = record.get("retrieved_context") or []
    if not retrieved:
        return None
    for rank, item in enumerate(retrieved, start=1):
        qid = item.get("question_id")
        source_tags = source_tags_map.get(qid, set()) if qid is not None else set()
        overlap = jaccard_tag_overlap(eval_tags, source_tags)
        if overlap > tag_overlap_threshold:
            return 1.0 / rank
    return 0.0


def _detect_condition(path: Path) -> str | None:
    name = path.name
    for cond in ("A", "B", "C", "D"):
        if name.startswith(f"condition_{cond.lower()}_"):
            return cond
    return None


def _load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze_file(con, path: Path, questions_parquet: str, tag_overlap_threshold: float,
                  top_k: int) -> dict | None:
    condition = _detect_condition(path)
    if condition == "A":
        print(f"[skip] {path.name} -- Kondisi A tidak melakukan retrieval, tidak ada yang dihitung.")
        return None
    if condition is None:
        print(f"[WARN] {path.name} -- tidak bisa deteksi kondisi dari nama file, dilewati.")
        return None

    records = _load_records(path)
    if not records:
        print(f"[WARN] {path.name} -- kosong/tidak bisa dibaca, dilewati.")
        return None

    all_source_qids: set[int] = set()
    for r in records:
        for item in (r.get("retrieved_context") or []):
            qid = item.get("question_id")
            if qid is not None:
                all_source_qids.add(int(qid))
    source_tags_map = load_question_tags(con, questions_parquet, list(all_source_qids))

    detail_rows = []
    precisions, mrrs = [], []
    n_short_context = 0
    for r in records:
        eval_tags = parse_tags(r.get("tags"))
        precision_result = compute_precision_at_k(r, eval_tags, source_tags_map, tag_overlap_threshold)
        mrr = compute_mrr_at_k(r, eval_tags, source_tags_map, tag_overlap_threshold)
        if precision_result is None:
            precision, n_context = None, 0
        else:
            precision, n_context = precision_result
            if n_context < top_k:
                n_short_context += 1
        detail_rows.append({
            "question_id": r.get("question_id"),
            "n_context_actual": n_context,
            "top_k_expected": top_k,
            "precision_at_k": round(precision, 4) if precision is not None else "",
            "mrr_at_k": round(mrr, 4) if mrr is not None else "",
        })
        if precision is not None:
            precisions.append(precision)
        if mrr is not None:
            mrrs.append(mrr)

    csv_path = path.parent / f"retrieval_quality_{path.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "n_context_actual", "top_k_expected", "precision_at_k", "mrr_at_k"])
        writer.writeheader()
        writer.writerows(detail_rows)

    return {
        "file": path.name,
        "condition": condition,
        "n_records": len(records),
        "n_with_context": len(precisions),
        "n_short_context": n_short_context,
        "precision_mean": round(statistics.mean(precisions), 4) if precisions else None,
        "precision_median": round(statistics.median(precisions), 4) if precisions else None,
        "mrr_mean": round(statistics.mean(mrrs), 4) if mrrs else None,
        "mrr_median": round(statistics.median(mrrs), 4) if mrrs else None,
        "csv_path": str(csv_path),
    }


def print_summary_table(rows: list[dict]) -> None:
    cols = [
        ("Kondisi", 8), ("File", 46), ("n", 5), ("n_ctx", 7),
        ("Prec@k(mean)", 13), ("Prec@k(med)", 12), ("MRR@k(mean)", 12), ("MRR@k(med)", 11),
    ]
    header = "".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        row = [
            r["condition"], r["file"][:45], str(r["n_records"]), str(r["n_with_context"]),
            f"{r['precision_mean']:.4f}" if r["precision_mean"] is not None else "-",
            f"{r['precision_median']:.4f}" if r["precision_median"] is not None else "-",
            f"{r['mrr_mean']:.4f}" if r["mrr_mean"] is not None else "-",
            f"{r['mrr_median']:.4f}" if r["mrr_median"] is not None else "-",
        ]
        print("".join(f"{val:<{w}}" for val, (_, w) in zip(row, cols)))
    print(f"\nTotal file dianalisis: {len(rows)}")
    for r in rows:
        if r["n_short_context"]:
            print(f"[catatan] {r['file']}: {r['n_short_context']}/{r['n_records']} pertanyaan punya "
                  f"retrieved_context lebih pendek dari top_k.")
    print("\nCATATAN METODOLOGIS WAJIB:")
    print("- Relevansi adalah PROXY tag-overlap (Jaccard), BUKAN ground truth manusia -- lihat docstring modul.")
    print("- Recall@k TIDAK dihitung di sini -- data pilot yang sudah ada tidak menyimpan daftar kandidat penuh "
          "sebelum top-k cutoff. Lihat --log-full-candidates di c_graphrag.py/b_condition_b_rag.py untuk run "
          "berikutnya.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-path", help="Satu file JSONL hasil Kondisi B/C/D")
    group.add_argument("--input-glob", help="Glob pattern untuk banyak file sekaligus, mis. 'llm/*/results/condition_*.jsonl'")
    parser.add_argument("--tag-overlap-threshold", type=float, default=0.0,
                         help="Ambang Jaccard tag-overlap utk dianggap 'relevan' (default 0.0 = minimal 1 tag sama)")
    parser.add_argument("--top-k", type=int, default=5,
                         help="Dipakai HANYA utk validasi konsistensi panjang retrieved_context (default 5)")
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    args = parser.parse_args()

    if not args.questions_parquet:
        print("[ERROR] QUESTIONS_PARQUET harus diset di .env atau lewat --questions-parquet")
        sys.exit(1)

    if args.input_path:
        paths = [Path(args.input_path)]
    else:
        paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
        if not paths:
            print(f"[ERROR] Tidak ada file cocok dengan --input-glob '{args.input_glob}'")
            sys.exit(1)

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")

    rows = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] {path} tidak ditemukan, dilewati.")
            continue
        result = analyze_file(con, path, args.questions_parquet, args.tag_overlap_threshold, args.top_k)
        if result:
            rows.append(result)
            print(f"[ok] {path.name} -> {result['csv_path']}")

    if not rows:
        print("Tidak ada file yang berhasil dianalisis (Kondisi A dilewati otomatis).")
        return

    print()
    print_summary_table(rows)


if __name__ == "__main__":
    main()
