#!/usr/bin/env python3
"""
analyze_error_attribution.py
=====================================
TIDAK ADA panggilan LLM baru. Join HASIL yang sudah ada dari tiga sumber
(judge hallucination lama, context-relevance judge baru, answer-relevance
judge baru) per question_id, lalu terapkan ATURAN DETERMINISTIK untuk
mengaitkan tiap kegagalan (hallucination) ke SATU tahap penyebab yang
paling mungkin: retrieval (relevansi konteks), entity anchoring/cakupan
(context kosong), atau generation (LLM mengabaikan/mendistorsi konteks
yang sebenarnya sudah cukup). Ini yang dibutuhkan Bab V untuk menjawab
"di mana pipeline gagal", bukan cuma "seberapa sering gagal".

SUMBER OTOMATIS (manifest-based)
------------------------------------------------------------
Untuk tiap file hasil kondisi (--input-path), script ini MENCARI SENDIRI
entri TERBARU (timestamp maksimum) di:
  - llm/evaluation/logs/judge_run_history.jsonl            (hallucination)
  - llm/evaluation/logs/context_relevance_run_history.jsonl (ctxrel)
  - llm/evaluation/logs/answer_relevance_run_history.jsonl  (ansrel)
yang field `input_path`-nya cocok (dibandingkan sebagai path RESOLVED,
supaya cwd yang berbeda saat run judge vs run script ini tidak jadi
masalah pencocokan). Kalau ada BEBERAPA run judge utk file sumber yang
sama (mis. ganti model judge), yang dipakai SELALU yang timestamp-nya
paling baru -- run lama tidak dihapus, cuma tidak dipakai di analisis ini.

--hallucination-path/--ctxrel-path/--ansrel-path override manual HANYA
didukung kalau --input-path cuma SATU file (single-file mode) -- kalau
--input-path lebih dari satu, override tidak bisa dipasangkan per-file
secara tidak ambigu, jadi resolusi manifest otomatis WAJIB dipakai.

ATURAN DETERMINISTIK (didokumentasikan di sini SUPAYA bisa dikutip
langsung di Bab III -- BUKAN ditentukan oleh judge LLM mana pun)
------------------------------------------------------------
retrieval_status (Kondisi B/C/D saja):
  - EMPTY       : n_items == 0 (ctxrel status "no_context")
  - SUFFICIENT  : sufficiency == "CUKUP"
  - IRRELEVANT  : TIDAK ADA item berlabel RELEVAN DAN sufficiency == "TIDAK_CUKUP"
  - PARTIAL     : selain ketiga kondisi di atas

hallucinated := hallucination_label in {HALUSINASI_SEBAGIAN, HALUSINASI_PENUH}
  (--strict-hallucination: hanya HALUSINASI_PENUH yang dihitung)

attribution (Kondisi B/C/D):
  1. EMPTY      + hallucinated -> ANCHORING_OR_COVERAGE_FAILURE
  2. IRRELEVANT + hallucinated -> RETRIEVAL_FAILURE
  3. PARTIAL    + hallucinated -> PARTIAL_RETRIEVAL
  4. SUFFICIENT + hallucinated -> GENERATION_FAILURE
  5. NOT hallucinated, answer_relevance_label == TIDAK_MENJAWAB -> OFF_TARGET_ANSWER
  6. NOT hallucinated, retrieval_status in {EMPTY, IRRELEVANT}  -> CORRECT_FROM_PARAMETRIC
  7. selain semua di atas -> SUCCESS
  (Aturan dicek berurutan 1->7, berhenti di yang pertama cocok.)

attribution (Kondisi A -- TIDAK ADA retrieval sama sekali, jadi TIDAK
ADA retrieval_status; hanya hallucination x answer_relevance):
  1. hallucinated -> PARAMETRIC_HALLUCINATION
  2. NOT hallucinated, answer_relevance_label == TIDAK_MENJAWAB -> OFF_TARGET_ANSWER
  3. selain itu -> SUCCESS

Baris dengan sumber yang HILANG (belum di-judge oleh salah satu/lebih
dari tiga judge) dilaporkan terpisah (n_missing_*), attribution-nya
"UNKNOWN" -- TIDAK ditebak, TIDAK dipaksa masuk salah satu kategori di atas.

CARA PAKAI
----------
    python analyze_error_attribution.py \\
        --input-path ../c_graphrag/results/condition_c_....jsonl

    python analyze_error_attribution.py \\
        --input-path ../a_pure_llm/results/condition_a_....jsonl \\
        --input-path ../b_rag/results/condition_b_....jsonl \\
        --input-path ../c_graphrag/results/condition_c_....jsonl

    # override manual (HANYA valid dgn 1 --input-path)
    python analyze_error_attribution.py --input-path <file c> \\
        --hallucination-path ../evaluation/results/judged__....jsonl \\
        --ctxrel-path ../evaluation/results/ctxrel__....jsonl \\
        --ansrel-path ../evaluation/results/ansrel__....jsonl
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ANALYSIS_DIR = Path("analysis")
LOG_DIR = Path("logs")

HALLUCINATED_LABELS_LOOSE = {"HALUSINASI_SEBAGIAN", "HALUSINASI_PENUH"}
HALLUCINATED_LABELS_STRICT = {"HALUSINASI_PENUH"}

KNOWN_ZERO_CONTEXT_QIDS = {241991, 52175918}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _detect_condition(path: Path) -> str | None:
    name = path.name
    for cond in ("A", "B", "C", "D"):
        if name.startswith(f"condition_{cond.lower()}_"):
            return cond
    return None


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


def _load_jsonl_by_qid(path) -> dict[int, dict]:
    return {r["question_id"]: r for r in _read_jsonl(path) if "question_id" in r}


def _resolve_path_str(p) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def find_latest_manifest_entry(manifest_path: Path, target_input_path) -> dict | None:
    """Entri manifest TERBARU (timestamp maksimum, string ISO 8601 -- bisa
    dibandingkan leksikografis) yang `input_path`-nya cocok dgn
    target_input_path, dibandingkan setelah di-resolve (path absolut)
    supaya cwd yang beda saat run judge vs saat script ini dijalankan
    tidak menggagalkan pencocokan."""
    target_resolved = _resolve_path_str(target_input_path)
    best = None
    for rec in _read_jsonl(manifest_path):
        if _resolve_path_str(rec.get("input_path", "")) != target_resolved:
            continue
        if best is None or str(rec.get("timestamp", "")) > str(best.get("timestamp", "")):
            best = rec
    return best


# ---------------------------------------------------------------------
# Deterministic rules
# ---------------------------------------------------------------------

def compute_retrieval_status(ctxrel_record: dict | None) -> str | None:
    if ctxrel_record is None:
        return None
    status = ctxrel_record.get("status")
    if status == "no_context" or ctxrel_record.get("n_items") == 0:
        return "EMPTY"
    if status != "ok":
        return None  # "failed" -- tidak bisa disimpulkan
    sufficiency = ctxrel_record.get("sufficiency")
    item_labels = ctxrel_record.get("item_labels") or []
    has_relevant = any(it.get("label") == "RELEVAN" for it in item_labels)
    if sufficiency == "CUKUP":
        return "SUFFICIENT"
    if not has_relevant and sufficiency == "TIDAK_CUKUP":
        return "IRRELEVANT"
    return "PARTIAL"


def compute_hallucinated(hallucination_label: str | None, strict: bool) -> bool | None:
    if hallucination_label is None:
        return None
    target = HALLUCINATED_LABELS_STRICT if strict else HALLUCINATED_LABELS_LOOSE
    return hallucination_label in target


def compute_attribution(condition: str, retrieval_status: str | None, hallucinated: bool | None,
                         answer_relevance_label: str | None) -> str:
    if hallucinated is None:
        return "UNKNOWN"

    if condition == "A":
        if hallucinated:
            return "PARAMETRIC_HALLUCINATION"
        if answer_relevance_label == "TIDAK_MENJAWAB":
            return "OFF_TARGET_ANSWER"
        return "SUCCESS"

    if retrieval_status is None:
        return "UNKNOWN"

    if retrieval_status == "EMPTY" and hallucinated:
        return "ANCHORING_OR_COVERAGE_FAILURE"
    if retrieval_status == "IRRELEVANT" and hallucinated:
        return "RETRIEVAL_FAILURE"
    if retrieval_status == "PARTIAL" and hallucinated:
        return "PARTIAL_RETRIEVAL"
    if retrieval_status == "SUFFICIENT" and hallucinated:
        return "GENERATION_FAILURE"
    if not hallucinated and answer_relevance_label == "TIDAK_MENJAWAB":
        return "OFF_TARGET_ANSWER"
    if not hallucinated and retrieval_status in ("EMPTY", "IRRELEVANT"):
        return "CORRECT_FROM_PARAMETRIC"
    return "SUCCESS"


# ---------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------

def analyze_file(input_path: Path, hallucination_override, ctxrel_override, ansrel_override,
                  strict_hallucination: bool) -> dict:
    condition = _detect_condition(input_path)
    if condition is None:
        print(f"[WARN] {input_path.name} -- tidak bisa deteksi kondisi dari nama file, dilewati.")
        return {"condition": None, "rows": [], "file": str(input_path)}

    source_records = _read_jsonl(input_path)
    question_ids = [r["question_id"] for r in source_records if "question_id" in r]

    hallucination_manifest = LOG_DIR / "judge_run_history.jsonl"
    ctxrel_manifest = LOG_DIR / "context_relevance_run_history.jsonl"
    ansrel_manifest = LOG_DIR / "answer_relevance_run_history.jsonl"

    hallucination_path = Path(hallucination_override) if hallucination_override else None
    ctxrel_path = Path(ctxrel_override) if ctxrel_override else None
    ansrel_path = Path(ansrel_override) if ansrel_override else None

    if hallucination_path is None:
        entry = find_latest_manifest_entry(hallucination_manifest, input_path)
        hallucination_path = Path(entry["output_path"]) if entry else None
    if ctxrel_path is None and condition != "A":
        entry = find_latest_manifest_entry(ctxrel_manifest, input_path)
        ctxrel_path = Path(entry["output_path"]) if entry else None
    if ansrel_path is None:
        entry = find_latest_manifest_entry(ansrel_manifest, input_path)
        ansrel_path = Path(entry["output_path"]) if entry else None

    hallucination_by_qid = _load_jsonl_by_qid(hallucination_path) if hallucination_path else {}
    ctxrel_by_qid = _load_jsonl_by_qid(ctxrel_path) if ctxrel_path else {}
    ansrel_by_qid = _load_jsonl_by_qid(ansrel_path) if ansrel_path else {}

    n_missing_hallucination = sum(1 for qid in question_ids if qid not in hallucination_by_qid)
    n_missing_ctxrel = sum(1 for qid in question_ids if qid not in ctxrel_by_qid) if condition != "A" else 0
    n_missing_ansrel = sum(1 for qid in question_ids if qid not in ansrel_by_qid)

    rows = []
    for qid in question_ids:
        hall_rec = hallucination_by_qid.get(qid)
        ctx_rec = ctxrel_by_qid.get(qid) if condition != "A" else None
        ans_rec = ansrel_by_qid.get(qid)

        hallucination_label = hall_rec.get("hallucination_label") if hall_rec else None
        hallucinated = compute_hallucinated(hallucination_label, strict_hallucination)
        retrieval_status = compute_retrieval_status(ctx_rec) if condition != "A" else None
        answer_relevance_label = ans_rec.get("answer_relevance_label") if ans_rec else None
        attribution = compute_attribution(condition, retrieval_status, hallucinated, answer_relevance_label)

        rows.append({
            "question_id": qid,
            "condition": condition,
            "hallucination_label": hallucination_label,
            "hallucinated": hallucinated,
            "retrieval_status": retrieval_status,
            "sufficiency": ctx_rec.get("sufficiency") if ctx_rec else None,
            "answer_relevance_label": answer_relevance_label,
            "attribution": attribution,
            "known_zero_context_case": qid in KNOWN_ZERO_CONTEXT_QIDS,
        })

    return {
        "condition": condition,
        "file": str(input_path),
        "hallucination_path": str(hallucination_path) if hallucination_path else None,
        "ctxrel_path": str(ctxrel_path) if ctxrel_path else None,
        "ansrel_path": str(ansrel_path) if ansrel_path else None,
        "n_total": len(question_ids),
        "n_missing_hallucination": n_missing_hallucination,
        "n_missing_ctxrel": n_missing_ctxrel,
        "n_missing_ansrel": n_missing_ansrel,
        "rows": rows,
    }


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def build_cross_tab(rows: list[dict]) -> tuple[list[str], list[str], dict]:
    statuses = sorted({r["retrieval_status"] for r in rows if r["retrieval_status"] is not None})
    labels = sorted({r["hallucination_label"] for r in rows if r["hallucination_label"] is not None})
    table = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["retrieval_status"] is None or r["hallucination_label"] is None:
            continue
        table[r["retrieval_status"]][r["hallucination_label"]] += 1
    return statuses, labels, table


def write_markdown_report(results: list[dict], strict_hallucination: bool, out_path: Path) -> None:
    lines = []
    lines.append("# Error Attribution Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"strict_hallucination: {strict_hallucination} "
                  f"({'hanya HALUSINASI_PENUH dihitung hallucinated' if strict_hallucination else 'HALUSINASI_SEBAGIAN + HALUSINASI_PENUH dihitung hallucinated'})")
    lines.append("")

    for res in results:
        condition = res["condition"]
        if condition is None:
            continue
        rows = res["rows"]
        lines.append(f"## Kondisi {condition} -- `{Path(res['file']).name}`")
        lines.append("")
        lines.append(f"- n_total: {res['n_total']}")
        lines.append(f"- missing hallucination judge: {res['n_missing_hallucination']}")
        if condition != "A":
            lines.append(f"- missing context-relevance judge: {res['n_missing_ctxrel']}")
        lines.append(f"- missing answer-relevance judge: {res['n_missing_ansrel']}")
        lines.append("")

        if condition != "A":
            statuses, labels, table = build_cross_tab(rows)
            n_crosstab_total = sum(table[s][l] for s in statuses for l in labels)
            lines.append("### retrieval_status × hallucination_label (count / %)")
            lines.append("")
            header = "| retrieval_status | " + " | ".join(labels) + " |"
            sep = "|---" * (len(labels) + 1) + "|"
            lines.append(header)
            lines.append(sep)
            for s in statuses:
                cells = []
                for l in labels:
                    c = table[s][l]
                    pct = f"{c / n_crosstab_total * 100:.1f}%" if n_crosstab_total else "0.0%"
                    cells.append(f"{c} ({pct})")
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")

        attribution_counts = Counter(r["attribution"] for r in rows)
        n_attr_total = len(rows)
        lines.append("### Distribusi attribution")
        lines.append("")
        lines.append("| attribution | count | % |")
        lines.append("|---|---|---|")
        for attr, c in attribution_counts.most_common():
            pct = f"{c / n_attr_total * 100:.1f}%" if n_attr_total else "0.0%"
            lines.append(f"| {attr} | {c} | {pct} |")
        lines.append("")

        lines.append("### Contoh question_id per kategori attribution (maks. 5)")
        lines.append("")
        by_attr = defaultdict(list)
        for r in rows:
            by_attr[r["attribution"]].append(r["question_id"])
        for attr in sorted(by_attr):
            examples = by_attr[attr][:5]
            lines.append(f"- **{attr}**: {', '.join(str(q) for q in examples)}")
        lines.append("")

        known_present = [r for r in rows if r["known_zero_context_case"]]
        if known_present:
            lines.append("### Kasus zero-context yang diketahui (241991 / 52175918)")
            lines.append("")
            for r in known_present:
                lines.append(f"- question_id={r['question_id']}: retrieval_status={r['retrieval_status']}, "
                              f"hallucination_label={r['hallucination_label']}, attribution={r['attribution']}")
            lines.append("")

    out_path.write_text("\n".join(lines))


def write_csv(results: list[dict], out_path: Path) -> None:
    fieldnames = ["question_id", "condition", "hallucination_label", "hallucinated", "retrieval_status",
                  "sufficiency", "answer_relevance_label", "attribution", "known_zero_context_case"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            for r in res["rows"]:
                writer.writerow(r)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", action="append", required=True,
                         help="File JSONL hasil Kondisi A/B/C/D (bisa dipakai berkali-kali utk banyak file).")
    parser.add_argument("--hallucination-path", default=None,
                         help="Override manual -- HANYA valid kalau cuma 1 --input-path.")
    parser.add_argument("--ctxrel-path", default=None,
                         help="Override manual -- HANYA valid kalau cuma 1 --input-path.")
    parser.add_argument("--ansrel-path", default=None,
                         help="Override manual -- HANYA valid kalau cuma 1 --input-path.")
    parser.add_argument("--strict-hallucination", action="store_true",
                         help="Hanya HALUSINASI_PENUH dihitung 'hallucinated' (default: HALUSINASI_SEBAGIAN + HALUSINASI_PENUH).")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_path]

    if (args.hallucination_path or args.ctxrel_path or args.ansrel_path) and len(input_paths) != 1:
        print("[ERROR] --hallucination-path/--ctxrel-path/--ansrel-path hanya valid kalau --input-path "
              "cuma SATU file (single-file mode) -- dengan banyak file, override tidak bisa dipasangkan "
              "per-file secara tidak ambigu. Gunakan resolusi manifest otomatis untuk banyak file.")
        sys.exit(1)

    results = []
    for p in input_paths:
        if not p.exists():
            print(f"[WARN] {p} tidak ditemukan, dilewati.")
            continue
        res = analyze_file(
            p,
            args.hallucination_path if len(input_paths) == 1 else None,
            args.ctxrel_path if len(input_paths) == 1 else None,
            args.ansrel_path if len(input_paths) == 1 else None,
            args.strict_hallucination,
        )
        if res["condition"] is not None:
            results.append(res)
            print(f"[ok] {p.name} (Kondisi {res['condition']}) -- n={res['n_total']}, "
                  f"missing hallucination={res['n_missing_hallucination']}, "
                  f"missing ctxrel={res['n_missing_ctxrel']}, missing ansrel={res['n_missing_ansrel']}")

    if not results:
        print("[ERROR] Tidak ada file yang berhasil dianalisis.")
        sys.exit(1)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = ANALYSIS_DIR / f"error_attribution_{ts}.csv"
    md_path = ANALYSIS_DIR / f"error_attribution_{ts}.md"

    write_csv(results, csv_path)
    write_markdown_report(results, args.strict_hallucination, md_path)

    print(f"\nCSV per-pertanyaan -> {csv_path}")
    print(f"Laporan markdown    -> {md_path}")


if __name__ == "__main__":
    main()
