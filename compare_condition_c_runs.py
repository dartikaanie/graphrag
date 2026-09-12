#!/usr/bin/env python3
"""
Rangkum & bandingkan SEMUA run (Kondisi A, B, C, D) jadi 1 tabel, supaya
gampang membandingkan baseline vs conventional RAG vs GraphRAG vs Dual-Level
Retrieval (adaptasi LightRAG), dan memilih model mana yang layak lanjut ke
n=384.

Keempat kondisi punya run_history.jsonl masing-masing di folder yang berbeda
(bukan 1 file gabungan):
    Kondisi A: a_pure_llm/logs/run_history.jsonl
    Kondisi B: b_rag/logs/run_history.jsonl
    Kondisi C: c_graphrag/logs/run_history.jsonl
    Kondisi D: d_lightrag/logs/run_history.jsonl

Script ini otomatis membaca keempatnya relatif terhadap folder root repo.
Kalau struktur foldermu beda, override lewat --history-path-a/-b/-c/-d.

Catatan penting:
- Kondisi A (pure LLM baseline) TIDAK melakukan retrieval, jadi field
  seperti `avg_retrieval_latency_sec` biasanya tidak relevan/ada untuk A.
  Script ini menampilkan "-" untuk field yang memang tidak ada di record,
  supaya tidak disalahartikan sebagai 0.00 (nilai asli).
- NF2 (pct_with_citation) tetap ditampilkan untuk ketiga kondisi bila
  datanya ada, karena berguna sebagai pembanding provenance/grounding
  meskipun kontribusi utamanya ada di Kondisi C.

Cara pakai (dari folder root repo, mis. graphrag/):
    python compare_condition_c_runs.py
    python compare_condition_c_runs.py --condition ABCD          # default, semua
    python compare_condition_c_runs.py --condition CD            # cuma C vs D
    python compare_condition_c_runs.py --n-sample 30            # filter n_sample_target tertentu
    python compare_condition_c_runs.py --sort similarity
    python compare_condition_c_runs.py --best-per-condition      # 1 baris terbaik per kondisi
    python compare_condition_c_runs.py --group                   # kelompokkan per (model, n), A/B/C/D berdampingan

Kalau dijalankan dari lokasi lain / nama folder beda, override path log-nya:
    python compare_condition_c_runs.py \\
        --history-path-a path/to/kondisi_a/run_history.jsonl \\
        --history-path-b path/to/kondisi_b/run_history.jsonl \\
        --history-path-c path/to/kondisi_c/run_history.jsonl \\
        --history-path-d path/to/kondisi_d/run_history.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

NA = "-"
OUTLIER_MARK = "⚠"
# Run dianggap outlier terhadap run SEJENIS-nya (kondisi+model+n sama) kalau
# similarity-nya jatuh di bawah rasio ini dari nilai MAX run sejenis dalam
# grup tsb. Ambang 0.5 dipilih karena kasus nyata yang ditemukan (mis. 0.26
# vs run sejenis 0.57, 0.20 vs run sejenis 0.58) jelas jauh di bawah ini,
# sementara variasi noise wajar antar-seed biasanya tidak sampai separuh.
OUTLIER_RATIO = 0.5

# Default path per kondisi, relatif terhadap folder tempat script dijalankan
# (diasumsikan folder root repo, mis. graphrag/).
DEFAULT_HISTORY_PATHS = {
    "A": "llm/a_pure_llm/logs/run_history.jsonl",
    "B": "llm/b_rag/logs/run_history.jsonl",
    "C": "llm/c_graphrag/logs/run_history.jsonl",
    "D": "llm/d_lightrag/logs/run_history.jsonl",
}

# Fallback tambahan per kondisi kalau default tidak ketemu (mis. script
# dijalankan dari dalam salah satu subfolder, bukan dari root).
FALLBACK_HISTORY_PATHS = {
    "A": ["logs/run_history.jsonl", "../a_pure_llm/logs/run_history.jsonl"],
    "B": ["logs/run_history.jsonl", "../b_rag/logs/run_history.jsonl"],
    "C": ["logs/run_history.jsonl", "../c_graphrag/logs/run_history.jsonl"],
    "D": ["logs/run_history.jsonl", "../d_lightrag/logs/run_history.jsonl"],
}


def resolve_history_path(path: Path, condition: str) -> Path:
    """Fallback pencarian kalau path default tidak ketemu, supaya script tetap
    jalan baik dipanggil dari folder root repo maupun dari dalam subfoldernya."""
    if path.exists():
        return path
    for c in FALLBACK_HISTORY_PATHS.get(condition, []):
        cp = Path(c)
        if cp.exists():
            return cp
    return path  # biar error message nunjukin path aslinya yang dicoba


def load_history(path: Path, condition: str) -> list[dict]:
    """Load 1 file run_history.jsonl dan tag tiap record dengan kondisinya,
    sebagai fallback kalau field 'condition' di dalam record ternyata tidak
    ada/salah (masing-masing file harusnya sudah cuma berisi 1 kondisi)."""
    records = []
    path = resolve_history_path(path, condition)
    if not path.exists():
        print(f"[WARN] Log Kondisi {condition} tidak ditemukan: {path}")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.setdefault("condition", condition)
            records.append(r)
    return records


def fmt(value, spec=None, na=NA):
    """Format a value, or return `na` placeholder if the field is missing (None)."""
    if value is None:
        return na
    if spec is None:
        return str(value)
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return na


def flag_outliers(runs: list[dict]) -> None:
    """Tandai in-place (menambah key '_outlier': bool) run yang similarity-nya
    jatuh drastis dibanding run sejenis lain (kondisi+model+n_processed sama).
    Butuh minimal 2 run sejenis untuk bisa dibandingkan; kalau cuma 1 run,
    tidak ada dasar pembanding jadi tidak ditandai."""
    subgroups: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r.get("condition"), r.get("model"), r.get("n_processed"))
        subgroups.setdefault(key, []).append(r)

    for key, group in subgroups.items():
        sims = [r.get("cosine_similarity_mean") for r in group if r.get("cosine_similarity_mean") is not None]
        if len(sims) < 2:
            for r in group:
                r["_outlier"] = False
            continue
        # Pakai MAX (bukan median) sebagai acuan "performa normal" grup ini —
        # median dari grup kecil (mis. 2 run) ikut tertarik turun oleh outlier
        # itu sendiri, sehingga kurang sensitif mendeteksinya.
        max_sim = max(sims)
        for r in group:
            sim = r.get("cosine_similarity_mean")
            r["_outlier"] = (sim is not None and max_sim > 0 and sim < OUTLIER_RATIO * max_sim)


def _fmt_grounding(r: dict) -> str:
    rg = r.get("require_grounding")
    return NA if rg is None else ("Yes" if rg else "No")


def print_table(runs: list[dict], sort_label: str, extra_note: str = "") -> None:
    has_grounding_col = any(r.get("require_grounding") is not None for r in runs)
    cols = [
        ("Kondisi", 8), ("Provider", 10), ("Model", 22), ("n", 4),
        ("Sim.Mean", 10), ("Sim.Med", 8), ("%>0.5", 7), ("NF2%", 7),
        ("Lat.avg", 8), ("Durasi(s)", 10),
    ]
    if has_grounding_col:
        cols.append(("Ground(D)", 9))
    header = "".join(f"{name:<{w}}" for name, w in cols)
    print(header)
    print("-" * len(header))

    any_outlier = False
    for r in runs:
        sim_mean_str = fmt(r.get("cosine_similarity_mean"), ".4f")
        if r.get("_outlier"):
            sim_mean_str += f" {OUTLIER_MARK}"
            any_outlier = True
        row = [
            str(r.get("condition", "?")),
            str(r.get("provider", "?"))[:9],
            str(r.get("model", "?"))[:21],
            str(r.get("n_processed", "?")),
            sim_mean_str,
            fmt(r.get("cosine_similarity_median"), ".4f"),
            fmt(r.get("pct_similarity_above_0_5"), ".1f") + ("%" if r.get("pct_similarity_above_0_5") is not None else ""),
            fmt(r.get("pct_with_citation"), ".1f") + ("%" if r.get("pct_with_citation") is not None else ""),
            fmt(r.get("avg_retrieval_latency_sec"), ".2f") + ("s" if r.get("avg_retrieval_latency_sec") is not None else ""),
            fmt(r.get("duration_sec"), ".1f"),
        ]
        if has_grounding_col:
            row.append(_fmt_grounding(r))
        line = "".join(f"{val:<{w}}" for val, (_, w) in zip(row, cols))
        print(line)

    print(f"\nTotal run ditampilkan: {len(runs)} ({sort_label})")
    if extra_note:
        print(extra_note)
    print("Catatan: pastikan semua run pakai --oversample-pool & --seed yang SAMA "
          "supaya perbandingan antar kondisi/model adil (pertanyaan yang diuji identik).")
    print("Catatan: '-' berarti field tidak ada di record run tsb (mis. Kondisi A "
          "biasanya tidak punya Lat.avg karena tidak melakukan retrieval), bukan nilai 0.")
    if any_outlier:
        print(f"Catatan: '{OUTLIER_MARK}' menandai run yang similarity-nya jatuh di bawah "
              f"{int(OUTLIER_RATIO * 100)}% dari nilai terbaik run sejenis (kondisi+model+n sama) — "
              "kemungkinan gejala truncation/timeout/error, cek log run tsb sebelum dipakai "
              "untuk kesimpulan.")


def print_grouped(runs: list[dict], sort_key_map: dict, sort_choice: str) -> None:
    """Kelompokkan runs berdasarkan (model, n_processed), lalu di dalam tiap
    kelompok tampilkan tiap kondisi berdampingan plus ringkasan mean±std per
    kondisi kalau ada >1 run sejenis. Ini yang dipakai untuk perbandingan
    apple-to-apple A vs B vs C pada sampel & model yang sama."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r.get("model"), r.get("n_processed"))
        groups.setdefault(key, []).append(r)

    # Urutkan kelompok: model alfabetis, lalu n menaik
    def group_sort_key(key):
        model, n = key
        return (str(model), n if n is not None else -1)

    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    any_outlier = False

    for (model, n) in sorted(groups.keys(), key=group_sort_key):
        group_runs = groups[(model, n)]
        group_runs.sort(key=lambda r: order.get(r.get("condition"), 99))

        conditions_present = sorted({r.get("condition") for r in group_runs})
        print(f"\n=== Model: {model} | n={n} | Kondisi: {', '.join(conditions_present)} ===")

        has_grounding_col = any(r.get("require_grounding") is not None for r in group_runs)
        cols = [
            ("Kondisi", 8), ("Sim.Mean", 10), ("Sim.Med", 8), ("%>0.5", 7),
            ("NF2%", 7), ("Lat.avg", 8), ("Durasi(s)", 10),
        ]
        if has_grounding_col:
            cols.append(("Ground(D)", 9))
        header = "".join(f"{name:<{w}}" for name, w in cols)
        print(header)
        print("-" * len(header))

        for r in group_runs:
            sim_mean_str = fmt(r.get("cosine_similarity_mean"), ".4f")
            if r.get("_outlier"):
                sim_mean_str += f" {OUTLIER_MARK}"
                any_outlier = True
            row = [
                str(r.get("condition", "?")),
                sim_mean_str,
                fmt(r.get("cosine_similarity_median"), ".4f"),
                fmt(r.get("pct_similarity_above_0_5"), ".1f") + ("%" if r.get("pct_similarity_above_0_5") is not None else ""),
                fmt(r.get("pct_with_citation"), ".1f") + ("%" if r.get("pct_with_citation") is not None else ""),
                fmt(r.get("avg_retrieval_latency_sec"), ".2f") + ("s" if r.get("avg_retrieval_latency_sec") is not None else ""),
                fmt(r.get("duration_sec"), ".1f"),
            ]
            if has_grounding_col:
                row.append(_fmt_grounding(r))
            line = "".join(f"{val:<{w}}" for val, (_, w) in zip(row, cols))
            print(line)

        # Ringkasan per kondisi kalau ada >1 run sejenis (misal beberapa kali re-run)
        by_cond: dict[str, list[dict]] = {}
        for r in group_runs:
            by_cond.setdefault(r.get("condition"), []).append(r)
        summary_lines = []
        for cond in sorted(by_cond.keys(), key=lambda c: order.get(c, 99)):
            group_c = by_cond[cond]
            sims = [r.get("cosine_similarity_mean") for r in group_c if r.get("cosine_similarity_mean") is not None]
            if len(sims) > 1:
                mean_ = statistics.mean(sims)
                stdev_ = statistics.stdev(sims)
                summary_lines.append(f"  {cond}: mean={mean_:.4f} ± std={stdev_:.4f} ({len(sims)} run)")
        if summary_lines:
            print("Ringkasan (run sejenis berulang):")
            for line in summary_lines:
                print(line)

    print(f"\n\nTotal kelompok (model, n) ditampilkan: {len(groups)}")
    print("Catatan: pastikan semua run pakai --oversample-pool & --seed yang SAMA "
          "supaya perbandingan antar kondisi/model adil (pertanyaan yang diuji identik).")
    print("Catatan: '-' berarti field tidak ada di record run tsb (mis. Kondisi A "
          "biasanya tidak punya Lat.avg karena tidak melakukan retrieval), bukan nilai 0.")
    if any_outlier:
        print(f"Catatan: '{OUTLIER_MARK}' menandai run yang similarity-nya jatuh di bawah "
              f"{int(OUTLIER_RATIO * 100)}% dari nilai terbaik run sejenis (kondisi+model+n sama) — "
              "kemungkinan gejala truncation/timeout/error, cek log run tsb sebelum dipakai "
              "untuk kesimpulan.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path-a", default=DEFAULT_HISTORY_PATHS["A"],
                         help="Path ke run_history.jsonl Kondisi A "
                              f"(default: {DEFAULT_HISTORY_PATHS['A']})")
    parser.add_argument("--history-path-b", default=DEFAULT_HISTORY_PATHS["B"],
                         help="Path ke run_history.jsonl Kondisi B "
                              f"(default: {DEFAULT_HISTORY_PATHS['B']})")
    parser.add_argument("--history-path-c", default=DEFAULT_HISTORY_PATHS["C"],
                         help="Path ke run_history.jsonl Kondisi C "
                              f"(default: {DEFAULT_HISTORY_PATHS['C']})")
    parser.add_argument("--history-path-d", default=DEFAULT_HISTORY_PATHS["D"],
                         help="Path ke run_history.jsonl Kondisi D "
                              f"(default: {DEFAULT_HISTORY_PATHS['D']})")
    parser.add_argument("--condition", default="ABCD",
                         help="Kondisi mana yang ditampilkan, gabungan huruf A/B/C/D "
                              "(mis. 'ABCD' untuk semua, 'CD' untuk C vs D saja). Default: ABCD")
    parser.add_argument("--n-sample", type=int, default=None,
                         help="Filter hanya run dengan n_sample_target tertentu")
    parser.add_argument("--sort", default="similarity",
                         choices=["similarity", "citation", "latency", "duration"])
    parser.add_argument("--best-per-condition", action="store_true",
                         help="Hanya tampilkan 1 baris terbaik (menurut --sort) per kondisi")
    parser.add_argument("--group", action="store_true",
                         help="Kelompokkan berdasarkan (model, n) yang sama, tampilkan "
                              "Kondisi A/B/C berdampingan per kelompok — untuk perbandingan "
                              "apple-to-apple, bukan sekadar diurutkan global.")
    args = parser.parse_args()

    wanted_conditions = {c for c in args.condition.upper() if c in ("A", "B", "C", "D")}
    if not wanted_conditions:
        print("[ERROR] --condition harus berisi kombinasi huruf A/B/C/D, contoh: ABCD, CD, AC")
        return

    path_map = {
        "A": args.history_path_a, "B": args.history_path_b,
        "C": args.history_path_c, "D": args.history_path_d,
    }
    records = []
    for cond in ("A", "B", "C", "D"):
        if cond in wanted_conditions:
            records.extend(load_history(Path(path_map[cond]), cond))

    runs = [
        r for r in records
        if r.get("condition") in wanted_conditions and r.get("status") == "success"
    ]
    if args.n_sample is not None:
        runs = [r for r in runs if r.get("n_sample_target") == args.n_sample]

    if not runs:
        print("Tidak ada run (status=success) yang cocok dengan filter "
              f"(condition={sorted(wanted_conditions)}, n_sample={args.n_sample}).")
        return

    sort_key_map = {
        "similarity": lambda r: r.get("cosine_similarity_mean") or 0,
        "citation": lambda r: r.get("pct_with_citation") or 0,
        "latency": lambda r: r.get("avg_retrieval_latency_sec")
        if r.get("avg_retrieval_latency_sec") is not None else float("inf"),
        "duration": lambda r: r.get("duration_sec")
        if r.get("duration_sec") is not None else float("inf"),
    }

    flag_outliers(runs)  # tandai run yang anomali dibanding run sejenisnya (sebelum sort/filter tampilan)

    if args.group:
        print_grouped(runs, sort_key_map, args.sort)
        return

    reverse = args.sort not in ("latency", "duration")
    runs.sort(key=sort_key_map[args.sort], reverse=reverse)

    if args.best_per_condition:
        seen = set()
        best = []
        for r in runs:  # already sorted best-first
            cond = r.get("condition")
            if cond not in seen:
                seen.add(cond)
                best.append(r)
        runs = best
        # keep a sensible display order: A, B, C, D
        order = {"A": 0, "B": 1, "C": 2, "D": 3}
        runs.sort(key=lambda r: order.get(r.get("condition"), 99))

    label = "terbaik per kondisi" if args.best_per_condition else f"diurutkan berdasarkan '{args.sort}'"
    print_table(runs, label)


if __name__ == "__main__":
    main()