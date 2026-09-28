#!/usr/bin/env python3
"""
compare_conditions_stats.py
=====================================
TIDAK ADA panggilan LLM baru. Desain statistik berpasangan (paired) untuk
membandingkan Kondisi A/B/C/D pada level pertanyaan yang SAMA (bukan
membandingkan rata-rata agregat mentah) -- ini desain statistik tesis yang
BELUM ADA implementasinya di repo sebelum script ini.

METRIK PER PERTANYAAN
------------------------------------------------------------
- cosine_similarity        : dari file hasil kondisi (semua kondisi)
- answer_relevance_score   : dari llm_judge_answer_relevance.py (reference-
  free) -- semua kondisi
- context_precision_strict : dari llm_judge_context_relevance.py -- HANYA
  Kondisi B/C/D (A tidak punya retrieval)
- sufficiency (ordinal)    : CUKUP=2, SEBAGIAN=1, TIDAK_CUKUP=0 -- HANYA
  B/C/D. Lebih TINGGI = lebih baik (konsisten dgn metrik lain).
- hallucination (ordinal)  : FAKTUAL=0, HALUSINASI_SEBAGIAN=1,
  HALUSINASI_PENUH=2 -- semua kondisi. ARAH TERBALIK dari metrik lain:
  di sini LEBIH RENDAH = LEBIH BAIK (0 = tidak ada halusinasi). WAJIB
  diingat saat membaca tabel Cliff's delta -- delta positif pada metrik
  ini berarti kondisi pertama LEBIH BANYAK berhalusinasi, BUKAN lebih baik.

SUMBER JUDGE DI-RESOLVE OTOMATIS via manifest (entri TERBARU per
input_path), pola SAMA dengan analyze_error_attribution.py:
  logs/judge_run_history.jsonl, logs/context_relevance_run_history.jsonl,
  logs/answer_relevance_run_history.jsonl.

PERBANDINGAN
------------------------------------------------------------
Untuk SETIAP PASANGAN kondisi yang diberikan (mis. --a-path + --b-path +
--c-path -> A-B, A-C, B-C; tambah --d-path -> + A-D, B-D, C-D): pairing
by intersection question_id, n berpasangan dilaporkan eksplisit per
metrik per pasangan (bisa beda-beda karena tidak semua pertanyaan punya
semua metrik, mis. ctxrel gagal parse utk sebagian pertanyaan).

UJI STATISTIK
------------------------------------------------------------
- `scipy.stats.wilcoxon` (two-sided) per pasangan per metrik.
  `zero_method="wilcox"` (default scipy, dipakai eksplisit bukan
  implisit) -- pasangan dengan selisih PERSIS 0 DIBUANG dari test
  (bukan diberi rank 0 seperti metode "pratt"/"zsplit"). Dipilih karena
  metrik ordinal di sini (hallucination, sufficiency) punya BANYAK ties
  (skala cuma 3 nilai), dan "wilcox" adalah konvensi klasik yang paling
  umum dilaporkan -- menyertakan selisih 0 ke ranking (pratt) bisa
  mengubah kesimpulan signifikansi tanpa alasan substantif yang jelas
  untuk kasus proporsi-tinggi-tie seperti ini.
- Cliff's delta (bands Romano dkk.: negligible<0.147, small<0.33,
  medium<0.474, large selainnya) -- lihat _stats_common.py.
- Koreksi Holm-Bonferroni DIAPLIKASIKAN PER METRIK, atas seluruh
  perbandingan pasangan yang punya p-value untuk metrik itu (bukan
  digabung lintas metrik) -- p_raw DAN p_holm_adjusted dilaporkan
  berdampingan supaya keduanya bisa dibaca.

CARA PAKAI
----------
    python compare_conditions_stats.py \\
        --a-path ../a_pure_llm/results/condition_a_....jsonl \\
        --b-path ../b_rag/results/condition_b_....jsonl \\
        --c-path ../c_graphrag/results/condition_c_....jsonl \\
        [--d-path ../d_lightrag/results/condition_d_....jsonl]
"""

import argparse
import csv
import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats_common import cliffs_delta, holm_correction  # noqa: E402

ANALYSIS_DIR = Path("analysis")
LOG_DIR = Path("logs")

SUFFICIENCY_ORDINAL = {"TIDAK_CUKUP": 0, "SEBAGIAN": 1, "CUKUP": 2}
HALLUCINATION_ORDINAL = {"FAKTUAL": 0, "HALUSINASI_SEBAGIAN": 1, "HALUSINASI_PENUH": 2}

METRICS_ALL_CONDITIONS = ["cosine_similarity", "answer_relevance_score"]
METRICS_RETRIEVAL_ONLY = ["context_precision_strict", "sufficiency_ordinal"]
METRICS_HALLUCINATION = ["hallucination_ordinal"]
ALL_METRICS = METRICS_ALL_CONDITIONS + METRICS_RETRIEVAL_ONLY + METRICS_HALLUCINATION

LOWER_IS_BETTER = {"hallucination_ordinal"}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _read_jsonl(path) -> list[dict]:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _resolve_path_str(p) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def find_latest_manifest_entry(manifest_path: Path, target_input_path) -> dict | None:
    target_resolved = _resolve_path_str(target_input_path)
    best = None
    for rec in _read_jsonl(manifest_path):
        if _resolve_path_str(rec.get("input_path", "")) != target_resolved:
            continue
        if best is None or str(rec.get("timestamp", "")) > str(best.get("timestamp", "")):
            best = rec
    return best


def load_condition_metrics(condition: str, result_path: Path) -> dict[int, dict]:
    """Return {question_id: {metric_name: value}} -- HANYA metrik yang
    berhasil dimuat (key hilang, BUKAN None, kalau sumbernya tidak ada
    sama sekali -- caller membedakan "tidak ada data" dari "None karena
    field-nya memang null")."""
    records = _read_jsonl(result_path)
    per_question: dict[int, dict] = {}
    for r in records:
        qid = r.get("question_id")
        if qid is None:
            continue
        entry = {}
        if r.get("cosine_similarity") is not None:
            entry["cosine_similarity"] = r["cosine_similarity"]
        per_question[qid] = entry

    hall_entry = find_latest_manifest_entry(LOG_DIR / "judge_run_history.jsonl", result_path)
    if hall_entry:
        for r in _read_jsonl(hall_entry["output_path"]):
            qid = r.get("question_id")
            label = r.get("hallucination_label")
            if qid in per_question and label in HALLUCINATION_ORDINAL:
                per_question[qid]["hallucination_ordinal"] = HALLUCINATION_ORDINAL[label]

    ansrel_entry = find_latest_manifest_entry(LOG_DIR / "answer_relevance_run_history.jsonl", result_path)
    if ansrel_entry:
        for r in _read_jsonl(ansrel_entry["output_path"]):
            qid = r.get("question_id")
            if qid in per_question and r.get("status") == "ok" and r.get("answer_relevance_score") is not None:
                per_question[qid]["answer_relevance_score"] = r["answer_relevance_score"]

    if condition != "A":
        ctxrel_entry = find_latest_manifest_entry(LOG_DIR / "context_relevance_run_history.jsonl", result_path)
        if ctxrel_entry:
            for r in _read_jsonl(ctxrel_entry["output_path"]):
                qid = r.get("question_id")
                if qid not in per_question or r.get("status") != "ok":
                    continue
                if r.get("context_precision_strict") is not None:
                    per_question[qid]["context_precision_strict"] = r["context_precision_strict"]
                suff = r.get("sufficiency")
                if suff in SUFFICIENCY_ORDINAL:
                    per_question[qid]["sufficiency_ordinal"] = SUFFICIENCY_ORDINAL[suff]

    return per_question


# ---------------------------------------------------------------------
# Per-condition summary (mean/median atas SELURUH pertanyaan yg punya
# metrik itu, TIDAK dibatasi ke irisan pasangan manapun).
# ---------------------------------------------------------------------

def per_condition_summary(condition_data: dict[str, dict[int, dict]]) -> dict:
    summary = {}
    for cond, per_question in condition_data.items():
        summary[cond] = {}
        for metric in ALL_METRICS:
            vals = [v[metric] for v in per_question.values() if metric in v]
            summary[cond][metric] = {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 4) if vals else None,
                "median": round(statistics.median(vals), 4) if vals else None,
            }
    return summary


# ---------------------------------------------------------------------
# Paired comparisons + Holm correction per metrik
# ---------------------------------------------------------------------

def paired_comparisons(condition_data: dict[str, dict[int, dict]]) -> list[dict]:
    conditions = list(condition_data.keys())
    pairs = list(itertools.combinations(conditions, 2))

    raw_results = []
    for metric in ALL_METRICS:
        for x_cond, y_cond in pairs:
            x_data = condition_data[x_cond]
            y_data = condition_data[y_cond]
            common_qids = sorted(
                qid for qid in (set(x_data) & set(y_data))
                if metric in x_data[qid] and metric in y_data[qid]
            )
            x_vals = [x_data[qid][metric] for qid in common_qids]
            y_vals = [y_data[qid][metric] for qid in common_qids]
            n = len(common_qids)

            stat, p_raw = (None, None)
            if n >= 1 and any((xv - yv) != 0 for xv, yv in zip(x_vals, y_vals)):
                stat, p_raw = wilcoxon(x_vals, y_vals, zero_method="wilcox", alternative="two-sided")
                stat, p_raw = float(stat), float(p_raw)

            delta, band = cliffs_delta(x_vals, y_vals) if n >= 1 else (None, None)

            raw_results.append({
                "metric": metric,
                "lower_is_better": metric in LOWER_IS_BETTER,
                "condition_x": x_cond,
                "condition_y": y_cond,
                "n_paired": n,
                "wilcoxon_statistic": round(stat, 4) if stat is not None else None,
                "p_raw": round(p_raw, 6) if p_raw is not None else None,
                "cliffs_delta": round(delta, 4) if delta is not None else None,
                "cliffs_delta_band": band,
            })

    # Holm correction PER METRIK, atas seluruh perbandingan yg punya p_raw.
    for metric in ALL_METRICS:
        idxs = [i for i, r in enumerate(raw_results) if r["metric"] == metric and r["p_raw"] is not None]
        if not idxs:
            continue
        p_raws = [raw_results[i]["p_raw"] for i in idxs]
        p_adj = holm_correction(p_raws)
        for i, p in zip(idxs, p_adj):
            raw_results[i]["p_holm"] = round(p, 6)
    for r in raw_results:
        r.setdefault("p_holm", None)

    return raw_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-path", default=None, help="File hasil Kondisi A")
    parser.add_argument("--b-path", default=None, help="File hasil Kondisi B")
    parser.add_argument("--c-path", default=None, help="File hasil Kondisi C")
    parser.add_argument("--d-path", default=None, help="File hasil Kondisi D (opsional)")
    args = parser.parse_args()

    given = {"A": args.a_path, "B": args.b_path, "C": args.c_path, "D": args.d_path}
    given = {k: Path(v) for k, v in given.items() if v}

    if len(given) < 2:
        print("[ERROR] Minimal 2 kondisi (dari --a-path/--b-path/--c-path/--d-path) harus diisi utk perbandingan berpasangan.")
        sys.exit(1)

    for cond, p in given.items():
        if not p.exists():
            print(f"[ERROR] --{cond.lower()}-path tidak ditemukan: {p}")
            sys.exit(1)

    condition_data = {}
    for cond, p in given.items():
        condition_data[cond] = load_condition_metrics(cond, p)
        print(f"[ok] Kondisi {cond}: {p.name} -> {len(condition_data[cond])} pertanyaan dimuat")

    summary = per_condition_summary(condition_data)
    comparisons = paired_comparisons(condition_data)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = ANALYSIS_DIR / f"compare_conditions_stats_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "lower_is_better", "condition_x", "condition_y",
                                                "n_paired", "wilcoxon_statistic", "p_raw", "p_holm",
                                                "cliffs_delta", "cliffs_delta_band"])
        writer.writeheader()
        writer.writerows(comparisons)

    md_path = ANALYSIS_DIR / f"compare_conditions_stats_{ts}.md"
    lines = ["# Compare Conditions Stats", "", f"Generated: {datetime.now(timezone.utc).isoformat()}",
              f"Kondisi dibandingkan: {', '.join(sorted(given))}", "",
              "## Ringkasan per kondisi (mean/median, n)", "", "```",
              json.dumps(summary, indent=2, default=str), "```", "",
              "## Perbandingan berpasangan (Wilcoxon + Cliff's delta, Holm-corrected per metrik)", "",
              "| metric | pair | n | wilcoxon stat | p_raw | p_holm | delta | band |",
              "|---|---|---|---|---|---|---|---|"]
    for r in comparisons:
        lines.append(
            f"| {r['metric']} | {r['condition_x']}-{r['condition_y']} | {r['n_paired']} | "
            f"{r['wilcoxon_statistic']} | {r['p_raw']} | {r['p_holm']} | {r['cliffs_delta']} | {r['cliffs_delta_band']} |"
        )
    lines.append("")
    lines.append("**Catatan arah**: untuk `hallucination_ordinal`, LEBIH RENDAH lebih baik (0=FAKTUAL) -- "
                  "kebalikan dari metrik lain di tabel ini.")
    md_path.write_text("\n".join(lines))

    print("\n" + "=" * 70)
    print("RINGKASAN PERBANDINGAN KONDISI")
    print("=" * 70)
    print(json.dumps({"summary": summary, "comparisons": comparisons}, indent=2, default=str))
    print(f"\nCSV -> {csv_path}")
    print(f"Markdown -> {md_path}")


if __name__ == "__main__":
    main()
