#!/usr/bin/env python3
"""
Rangkum & bandingkan semua run Kondisi C dari logs/run_history.jsonl
jadi 1 tabel, supaya gampang pilih model mana yang layak lanjut ke n=384.

Cara pakai (dari folder 04_graphrag/):
    python compare_condition_c_runs.py
    python compare_condition_c_runs.py --n-sample 10   # filter hanya run n=10
    python compare_condition_c_runs.py --sort similarity
"""

import argparse
import json
from pathlib import Path


def load_history(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        print(f"[ERROR] Tidak ditemukan: {path}")
        return records
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path", default="logs/run_history.jsonl")
    parser.add_argument("--n-sample", type=int, default=None,
                         help="Filter hanya run dengan n_sample_target tertentu")
    parser.add_argument("--sort", default="similarity",
                         choices=["similarity", "citation", "latency", "duration"])
    args = parser.parse_args()

    records = load_history(Path(args.history_path))
    condition_c = [
        r for r in records
        if r.get("condition") == "C" and r.get("status") == "success"
    ]
    if args.n_sample is not None:
        condition_c = [r for r in condition_c if r.get("n_sample_target") == args.n_sample]

    if not condition_c:
        print("Tidak ada run Kondisi C (status=success) yang cocok dengan filter.")
        return

    sort_key_map = {
        "similarity": lambda r: r.get("cosine_similarity_mean", 0),
        "citation": lambda r: r.get("pct_with_citation", 0),
        "latency": lambda r: r.get("avg_retrieval_latency_sec", 999),
        "duration": lambda r: r.get("duration_sec", 999),
    }
    condition_c.sort(key=sort_key_map[args.sort], reverse=(args.sort != "latency" and args.sort != "duration"))

    # Header
    cols = [
        ("Provider", 10), ("Model", 22), ("n", 4), ("Sim.Mean", 9),
        ("Sim.Med", 8), ("%>0.5", 7), ("NF2%", 7), ("Lat.avg", 8), ("Durasi(s)", 10),
    ]
    header = "".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))

    for r in condition_c:
        row = [
            str(r.get("provider", "?"))[:9],
            str(r.get("model", "?"))[:21],
            str(r.get("n_processed", "?")),
            f"{r.get('cosine_similarity_mean', 0):.4f}",
            f"{r.get('cosine_similarity_median', 0):.4f}",
            f"{r.get('pct_similarity_above_0_5', 0):.1f}%",
            f"{r.get('pct_with_citation', 0):.1f}%",
            f"{r.get('avg_retrieval_latency_sec', 0):.2f}s",
            f"{r.get('duration_sec', 0):.1f}",
        ]
        line = "".join(f"{val:<{w}}" for val, (_, w) in zip(row, cols))
        print(line)

    print(f"\nTotal run ditampilkan: {len(condition_c)} (diurutkan berdasarkan '{args.sort}')")
    print("Catatan: pastikan semua run pakai --oversample-pool & --seed yang SAMA "
          "supaya perbandingan antar model adil (pertanyaan yang diuji identik).")


if __name__ == "__main__":
    main()