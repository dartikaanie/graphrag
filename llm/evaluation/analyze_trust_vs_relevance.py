#!/usr/bin/env python3
"""
analyze_trust_vs_relevance.py
=====================================
TIDAK ADA panggilan LLM baru. Menjawab pertanyaan inti kontribusi tesis
ini (trust-weighted GraphRAG): apakah item yang diberi `trust_weight`/
`combined_score` LEBIH TINGGI oleh Kondisi C benar-benar LEBIH RELEVAN
(menurut context-relevance judge, llm_judge_context_relevance.py), dan
apakah trust-weighting benar-benar mengungguli fusion mode "uniform"
(ablasi) pada Precision@k yang sama.

Item context-relevance (`item_labels`) di-JOIN BALIK ke `retrieved_context`
sumbernya (file hasil Kondisi C) per question_id+answer_id -- kedua file
ini TIDAK diubah, murni dibaca.

SUMBER DATA
------------------------------------------------------------
- `--input-path` (bisa >1): file hasil Kondisi C mode trust_weighted
  (fusion-mode default).
- `--uniform-path` (opsional, bisa >1): file hasil Kondisi C mode
  `--fusion-mode uniform` (ablasi) PADA SAMPEL YANG SAMA -- kalau diisi,
  bagian ablasi (Precision@k strict trust_weighted vs uniform) dihitung.
- ctxrel judge output utk tiap file di atas di-RESOLVE OTOMATIS lewat
  manifest `llm/evaluation/logs/context_relevance_run_history.jsonl`
  (entri terbaru per input_path) -- sama seperti
  analyze_error_attribution.py. Override manual: `--ctxrel-path`/
  `--uniform-ctxrel-path` (urutan sejajar dgn --input-path/--uniform-path,
  HANYA valid kalau jumlah file & override sama banyak).

TIGA BAGIAN ANALISIS
------------------------------------------------------------
1. ITEM LEVEL (unit analisis: satu item retrieved_context):
   - Spearman rho: trust_weight vs relevance (ordinal RELEVAN=1,
     SEBAGIAN=0.5, TIDAK_RELEVAN=0), dan combined_score vs relevance.
   - Mann-Whitney U: trust_weight item RELEVAN vs trust_weight item
     TIDAK_RELEVAN.
   - Relevance rate (strict, % RELEVAN) per source_stage, per is_accepted,
     per hop.
2. QUESTION LEVEL (unit analisis: satu pertanyaan -- MENGHINDARI
   pseudo-replication karena item DALAM satu pertanyaan TIDAK independen):
   per pertanyaan, (mean trust item relevan) - (mean trust item TIDAK
   relevan), HANYA utk pertanyaan yang punya KEDUA jenis item. One-sample
   Wilcoxon signed-rank test terhadap 0 atas selisih-selisih ini.
3. ABLASI (hanya kalau --uniform-path diisi): paired Wilcoxon signed-rank
   + Cliff's delta pada context_precision_strict per-pertanyaan,
   trust_weighted vs uniform, atas IRISAN question_id (status "ok" di
   KEDUA ctxrel output). n berpasangan dilaporkan eksplisit.

KETERBATASAN METODOLOGIS (WAJIB dibaca)
------------------------------------------------------------
- "Relevan" di sini adalah label LLM-judge (llm_judge_context_relevance.py),
  BUKAN ground truth manusia -- lihat keterbatasan judge itu sendiri.
- Item-level dan question-level TIDAK dijamin konsisten arah temuannya --
  question-level SENGAJA dilaporkan terpisah karena item dalam satu
  pertanyaan bukan observasi independen (item-level test bisa terlalu
  percaya diri/"anti-conservative" kalau dipakai sebagai satu-satunya
  bukti).
- Trust-weight join butuh answer_id yang PERSIS SAMA di kedua file
  (retrieved_context vs item_labels) -- item yang gagal di-judge (status
  "failed"/"no_context" di ctxrel) tidak ikut dihitung, dilaporkan sebagai
  n_unmatched.

CARA PAKAI
----------
    python analyze_trust_vs_relevance.py \\
        --input-path ../c_graphrag/results/condition_c_..._fw0-7-0-3.jsonl

    python analyze_trust_vs_relevance.py \\
        --input-path ../c_graphrag/results/condition_c_..._fw0-7-0-3.jsonl \\
        --uniform-path ../c_graphrag/results/condition_c_..._uniform.jsonl
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats_common import cliffs_delta  # noqa: E402

ANALYSIS_DIR = Path("analysis")
LOG_DIR = Path("logs")

LABEL_ORDINAL = {"RELEVAN": 1.0, "SEBAGIAN": 0.5, "TIDAK_RELEVAN": 0.0}


# ---------------------------------------------------------------------
# IO helpers (pola sama dengan analyze_error_attribution.py)
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


def resolve_ctxrel_path(input_path: Path, override) -> Path | None:
    if override:
        return Path(override)
    entry = find_latest_manifest_entry(LOG_DIR / "context_relevance_run_history.jsonl", input_path)
    return Path(entry["output_path"]) if entry else None


# ---------------------------------------------------------------------
# Flatten items: join retrieved_context (file kondisi C) <-> item_labels
# (output ctxrel judge) per question_id (eval) + answer_id.
# ---------------------------------------------------------------------

def flatten_items(c_records: list[dict], ctxrel_by_qid: dict[int, dict]) -> tuple[list[dict], int]:
    rows = []
    n_unmatched = 0
    for rec in c_records:
        qid = rec.get("question_id")
        ctx_rec = ctxrel_by_qid.get(qid)
        if ctx_rec is None or ctx_rec.get("status") != "ok":
            continue
        item_labels_by_answer = {it["answer_id"]: it for it in ctx_rec.get("item_labels", [])}
        for rank, item in enumerate(rec.get("retrieved_context") or [], start=1):
            aid = item.get("answer_id")
            label_entry = item_labels_by_answer.get(aid)
            if label_entry is None:
                n_unmatched += 1
                continue
            label = label_entry["label"]
            rows.append({
                "question_id": qid,
                "rank": rank,
                "answer_id": aid,
                "trust_weight": item.get("trust_weight"),
                "combined_score": item.get("combined_score"),
                "source_stage": item.get("source_stage"),
                "hop": item.get("hop"),
                "rel_type": item.get("rel_type"),
                "is_accepted": item.get("is_accepted"),
                "label": label,
                "ordinal": LABEL_ORDINAL[label],
            })
    return rows, n_unmatched


# ---------------------------------------------------------------------
# Item level
# ---------------------------------------------------------------------

def item_level_stats(items: list[dict]) -> dict:
    trust = [it["trust_weight"] for it in items if it["trust_weight"] is not None]
    ordinal_for_trust = [it["ordinal"] for it in items if it["trust_weight"] is not None]
    combined = [it["combined_score"] for it in items if it["combined_score"] is not None]
    ordinal_for_combined = [it["ordinal"] for it in items if it["combined_score"] is not None]

    trust_rho, trust_p = (None, None)
    if len(trust) >= 3:
        trust_rho, trust_p = spearmanr(trust, ordinal_for_trust)

    combined_rho, combined_p = (None, None)
    if len(combined) >= 3:
        combined_rho, combined_p = spearmanr(combined, ordinal_for_combined)

    trust_relevan = [it["trust_weight"] for it in items if it["label"] == "RELEVAN" and it["trust_weight"] is not None]
    trust_tidak_relevan = [it["trust_weight"] for it in items if it["label"] == "TIDAK_RELEVAN" and it["trust_weight"] is not None]
    mwu_stat, mwu_p = (None, None)
    if trust_relevan and trust_tidak_relevan:
        mwu_stat, mwu_p = mannwhitneyu(trust_relevan, trust_tidak_relevan, alternative="two-sided")

    def _relevance_rate_by(key):
        groups = defaultdict(list)
        for it in items:
            groups[it.get(key)].append(1 if it["label"] == "RELEVAN" else 0)
        return {
            str(k): {"n": len(v), "relevance_rate_strict": round(sum(v) / len(v), 4) if v else None}
            for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }

    return {
        "n_items": len(items),
        "spearman_trust_weight_vs_relevance": {
            "rho": round(float(trust_rho), 4) if trust_rho is not None else None,
            "p_value": round(float(trust_p), 6) if trust_p is not None else None,
            "n": len(trust),
        },
        "spearman_combined_score_vs_relevance": {
            "rho": round(float(combined_rho), 4) if combined_rho is not None else None,
            "p_value": round(float(combined_p), 6) if combined_p is not None else None,
            "n": len(combined),
        },
        "mannwhitney_trust_relevan_vs_tidak_relevan": {
            "u_statistic": round(float(mwu_stat), 4) if mwu_stat is not None else None,
            "p_value": round(float(mwu_p), 6) if mwu_p is not None else None,
            "n_relevan": len(trust_relevan),
            "n_tidak_relevan": len(trust_tidak_relevan),
        },
        "relevance_rate_by_source_stage": _relevance_rate_by("source_stage"),
        "relevance_rate_by_is_accepted": _relevance_rate_by("is_accepted"),
        "relevance_rate_by_hop": _relevance_rate_by("hop"),
    }


# ---------------------------------------------------------------------
# Question level (menghindari pseudo-replication)
# ---------------------------------------------------------------------

def question_level_stats(items: list[dict]) -> dict:
    by_question = defaultdict(list)
    for it in items:
        by_question[it["question_id"]].append(it)

    diffs = []
    for qid, its in by_question.items():
        relevant_trust = [it["trust_weight"] for it in its if it["label"] == "RELEVAN" and it["trust_weight"] is not None]
        nonrelevant_trust = [it["trust_weight"] for it in its if it["label"] != "RELEVAN" and it["trust_weight"] is not None]
        if relevant_trust and nonrelevant_trust:
            diffs.append(statistics.mean(relevant_trust) - statistics.mean(nonrelevant_trust))

    wilcoxon_stat, wilcoxon_p = (None, None)
    if len(diffs) >= 1 and any(d != 0 for d in diffs):
        wilcoxon_stat, wilcoxon_p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")

    return {
        "n_questions_with_both_groups": len(diffs),
        "mean_diff": round(statistics.mean(diffs), 4) if diffs else None,
        "median_diff": round(statistics.median(diffs), 4) if diffs else None,
        "wilcoxon_one_sample_vs_zero": {
            "statistic": round(float(wilcoxon_stat), 4) if wilcoxon_stat is not None else None,
            "p_value": round(float(wilcoxon_p), 6) if wilcoxon_p is not None else None,
        },
        "caveat": (
            "Item dalam satu pertanyaan TIDAK independen (semua respons terhadap query yang sama) -- "
            "unit analisis di sini SENGAJA per-pertanyaan (mean trust relevan - mean trust non-relevan), "
            "BUKAN per-item, justru untuk menghindari pseudo-replication yang bisa membuat item-level "
            "test terlalu percaya diri."
        ),
    }


# ---------------------------------------------------------------------
# Ablasi: trust_weighted vs uniform, context_precision_strict per-pertanyaan
# ---------------------------------------------------------------------

def ablation_stats(trust_ctxrel_by_qid: dict[int, dict], uniform_ctxrel_by_qid: dict[int, dict]) -> dict:
    common_ids = sorted(
        qid for qid in (set(trust_ctxrel_by_qid) & set(uniform_ctxrel_by_qid))
        if trust_ctxrel_by_qid[qid].get("status") == "ok" and uniform_ctxrel_by_qid[qid].get("status") == "ok"
    )
    trust_vals = [trust_ctxrel_by_qid[qid]["context_precision_strict"] for qid in common_ids]
    uniform_vals = [uniform_ctxrel_by_qid[qid]["context_precision_strict"] for qid in common_ids]

    wilcoxon_stat, wilcoxon_p = (None, None)
    if common_ids and any((t - u) != 0 for t, u in zip(trust_vals, uniform_vals)):
        wilcoxon_stat, wilcoxon_p = wilcoxon(trust_vals, uniform_vals, zero_method="wilcox", alternative="two-sided")

    delta, band = cliffs_delta(trust_vals, uniform_vals)

    return {
        "n_paired": len(common_ids),
        "trust_weighted_mean": round(statistics.mean(trust_vals), 4) if trust_vals else None,
        "uniform_mean": round(statistics.mean(uniform_vals), 4) if uniform_vals else None,
        "wilcoxon_paired": {
            "statistic": round(float(wilcoxon_stat), 4) if wilcoxon_stat is not None else None,
            "p_value": round(float(wilcoxon_p), 6) if wilcoxon_p is not None else None,
        },
        "cliffs_delta": round(delta, 4) if delta is not None else None,
        "cliffs_delta_band": band,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", action="append", required=True,
                         help="File hasil Kondisi C mode trust_weighted (bisa dipakai berkali-kali).")
    parser.add_argument("--uniform-path", action="append", default=None,
                         help="File hasil Kondisi C mode --fusion-mode uniform (ablasi, opsional).")
    parser.add_argument("--ctxrel-path", action="append", default=None,
                         help="Override manual, urutan sejajar dgn --input-path.")
    parser.add_argument("--uniform-ctxrel-path", action="append", default=None,
                         help="Override manual, urutan sejajar dgn --uniform-path.")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_path]
    uniform_paths = [Path(p) for p in args.uniform_path] if args.uniform_path else []

    if args.ctxrel_path and len(args.ctxrel_path) != len(input_paths):
        print("[ERROR] --ctxrel-path (kalau diisi) harus sejumlah --input-path.")
        sys.exit(1)
    if args.uniform_ctxrel_path and len(args.uniform_ctxrel_path) != len(uniform_paths):
        print("[ERROR] --uniform-ctxrel-path (kalau diisi) harus sejumlah --uniform-path.")
        sys.exit(1)

    all_items = []
    trust_ctxrel_by_qid: dict[int, dict] = {}
    total_unmatched = 0
    for i, p in enumerate(input_paths):
        if not p.exists():
            print(f"[WARN] {p} tidak ditemukan, dilewati.")
            continue
        ctxrel_override = args.ctxrel_path[i] if args.ctxrel_path else None
        ctxrel_path = resolve_ctxrel_path(p, ctxrel_override)
        if ctxrel_path is None:
            print(f"[WARN] Tidak ketemu hasil context-relevance judge utk {p.name}, dilewati "
                  f"(jalankan llm_judge_context_relevance.py dulu).")
            continue
        c_records = _read_jsonl(p)
        ctxrel_by_qid = {r["question_id"]: r for r in _read_jsonl(ctxrel_path) if "question_id" in r}
        trust_ctxrel_by_qid.update(ctxrel_by_qid)
        rows, unmatched = flatten_items(c_records, ctxrel_by_qid)
        all_items.extend(rows)
        total_unmatched += unmatched
        print(f"[ok] {p.name} -> {len(rows)} item (ctxrel: {ctxrel_path.name})")

    if not all_items:
        print("[ERROR] Tidak ada item yang berhasil di-flatten -- cek apakah ctxrel judge sudah dijalankan.")
        sys.exit(1)

    item_stats = item_level_stats(all_items)
    question_stats = question_level_stats(all_items)

    ablation = None
    if uniform_paths:
        uniform_ctxrel_by_qid: dict[int, dict] = {}
        for i, p in enumerate(uniform_paths):
            if not p.exists():
                print(f"[WARN] {p} tidak ditemukan, dilewati.")
                continue
            override = args.uniform_ctxrel_path[i] if args.uniform_ctxrel_path else None
            ctxrel_path = resolve_ctxrel_path(p, override)
            if ctxrel_path is None:
                print(f"[WARN] Tidak ketemu hasil context-relevance judge utk {p.name} (uniform), dilewati.")
                continue
            uniform_ctxrel_by_qid.update({r["question_id"]: r for r in _read_jsonl(ctxrel_path) if "question_id" in r})
        if uniform_ctxrel_by_qid:
            ablation = ablation_stats(trust_ctxrel_by_qid, uniform_ctxrel_by_qid)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = ANALYSIS_DIR / f"trust_vs_relevance_items_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "rank", "answer_id", "trust_weight",
                                                "combined_score", "source_stage", "hop", "rel_type",
                                                "is_accepted", "label", "ordinal"])
        writer.writeheader()
        writer.writerows(all_items)

    md_path = ANALYSIS_DIR / f"trust_vs_relevance_{ts}.md"
    lines = ["# Trust vs Relevance Report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}",
              f"n_items unmatched (item_labels tidak ketemu, dilewati): {total_unmatched}", ""]
    lines.append("## Item level")
    lines.append("```")
    lines.append(json.dumps(item_stats, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Question level (menghindari pseudo-replication)")
    lines.append("```")
    lines.append(json.dumps(question_stats, indent=2, default=str))
    lines.append("```")
    lines.append("")
    if ablation is not None:
        lines.append("## Ablasi: trust_weighted vs uniform (context_precision_strict)")
        lines.append("```")
        lines.append(json.dumps(ablation, indent=2, default=str))
        lines.append("```")
    md_path.write_text("\n".join(lines))

    print("\n" + "=" * 70)
    print("RINGKASAN TRUST VS RELEVANCE")
    print("=" * 70)
    print(json.dumps({"item_level": item_stats, "question_level": question_stats, "ablation": ablation},
                      indent=2, default=str))
    print(f"\nCSV item-level -> {csv_path}")
    print(f"Laporan markdown -> {md_path}")


if __name__ == "__main__":
    main()
