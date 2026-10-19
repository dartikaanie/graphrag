"""
reputation_stage1_recheck.py
=====================================
Stage-1 ulang (dev set) dengan PERBAIKAN reputasi pada trustScore, untuk
memutuskan apakah reputasi kontributor dipakai di Kondisi C. Lanjutan dari
reputation_ablation_check.py (hasil n=50: BERPENGARUH -- top-5 berubah pada
4% / 6% / 10% / 40% pertanyaan untuk alpha 0.25 / 0.5 / 0.75 / 1).

Aturan keputusan dicatat di docs/DECISION_C_SCORING.md (Amendment 5)
SEBELUM skrip ini dijalankan dengan --judge.

APA YANG DILAKUKAN
------------------------------------------------------------
1. Dev set, retrieval, dan fuse_and_rank() v3 PERSIS seperti stage1_sweep.py.
   Kandidat diambil sekali per pertanyaan, lalu diperingkat untuk setiap
   alpha dalam dua versi:
     - base   : trustScore apa adanya (reputasi = konstanta 0.1).
     - repfix : trust_A' = trust_A - 0.2*0.5 + 0.2*author_weight
                (rumus sama dgn reputation_ablation_check.apply_reputation_fix).
2. Pemeriksaan reproduksi: top-5 `base` harus IDENTIK dengan file
   retrieval-only stage 1 yang sudah di-commit (c5a21d7). Kalau tidak,
   skrip berhenti -- perbandingan tidak sah.
3. Item `repfix` yang belum pernah dinilai pada stage 1 dinilai dengan
   ctxrel-v2, judge PRIMARY, input judge identik dengan stage 1. Item yang
   sudah dinilai dipakai ulang (kunci: question_id, answer_id, text_hash).
   Tanpa --judge, skrip berhenti setelah menampilkan rencana (0 panggilan API).
4. Metrik stage 1 dihitung untuk repfix dengan fungsi yang sama
   (stage1_sweep.compute_alpha_metrics), lalu selection helper dijalankan.
   Hasil base dihitung ulang dari file stage 1 dan harus sama dengan laporan
   stage 1 (pemeriksaan kedua).

ATURAN KEPUTUSAN (salin ke Amendment 5 sebelum --judge)
------------------------------------------------------------
Margin seri sama dengan stage 1: <= 2 poin persen (1 pertanyaan) untuk
kriteria utama, <= 0.02 untuk rata-rata skor relevansi.
  R1. Untuk alpha = 0.25 dan alpha = 0.5 (kandidat stage 2 saat ini),
      repfix TIDAK lebih buruk dari base melebihi margin seri pada kriteria
      utama (% pertanyaan dgn >=1 RELEVANT) DAN pada rata-rata skor.
  Jika R1 terpenuhi -> SERTAKAN reputasi. Stage 1 versi repfix menjadi
      hasil stage 1 resmi; kandidat stage 2 = top_two dari selection helper
      versi repfix (dilaporkan apakah sama dengan {0.5, 0.25}).
  Jika R1 tidak terpenuhi -> JANGAN sertakan reputasi. Stage 1 asli tetap
      berlaku. Reputasi dilaporkan sebagai temuan eksploratif.

KELUARAN (llm/c_graphrag/results/reputation_check/)
------------------------------------------------------------
  stage1_repfix_alpha<a>.jsonl                 retrieval repfix per alpha
  ctxrel_v2_repfix_primary.jsonl               penilaian item baru (resume aman)
  stage1_repfix_report_<ts>.md / .json         perbandingan + keputusan

CARA PAKAI (dari root repo)
------------------------------------------------------------
    python3 llm/c_graphrag/reputation_stage1_recheck.py           (rencana saja, gratis)
    python3 llm/c_graphrag/reputation_stage1_recheck.py --judge   (nilai item baru + laporan)
    python3 llm/c_graphrag/reputation_stage1_recheck.py --verify-applied
        (setelah 13_apply_reputation_trust.py --apply; gratis)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import c_graphrag as cg  # noqa: E402
import reputation_ablation_check as rac  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STAGE2_ALPHAS = (0.25, 0.5)
CTXREL_VERSION = "ctxrel-v2"
JUDGE_ID = "primary"
PCT_TIE = 2.0
SCORE_TIE = 0.02
from llm import archive_paths  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT_DIR = RESULTS / "reputation_check"
# Inputs of record were moved to the pre-official archive (2026-10-19);
# resolve() returns the original path while it exists, else the archived copy.
STAGE1_JUDGMENTS = archive_paths.resolve(RESULTS / "ctxrel_v2_stage1_sweep_primary.jsonl")
NEW_JUDGMENTS = archive_paths.resolve(OUT_DIR / "ctxrel_v2_repfix_primary.jsonl")


# ---------------------------------------------------------------------
# Logika murni -- diuji di tests/test_reputation_stage1_recheck.py
# ---------------------------------------------------------------------

def stage1_file_for_alpha(alpha: float, results_dir: Path | None = None) -> Path:
    """File retrieval-only stage 1 (dev, n50, seed42) untuk satu alpha."""
    if results_dir is None:  # default: results/, or its pre-official archive copy
        results_dir = RESULTS if any(RESULTS.glob("condition_c_*_n50_seed42_*_dev_v3_*.jsonl")) \
            else (archive_paths.archived_dir(RESULTS) or RESULTS)
    matches = []
    for p in sorted(results_dir.glob("condition_c_*_n50_seed42_*_dev_v3_*.jsonl")):
        with open(p) as f:
            first = f.readline()
        if first and json.loads(first).get("config", {}).get("alpha") == alpha:
            matches.append(p)
    if len(matches) != 1:
        raise FileNotFoundError(f"alpha={alpha}: butuh tepat 1 file stage 1, ditemukan {len(matches)}: {matches}")
    return matches[0]


def selected_ids(path: Path) -> dict[int, list[int]]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[int(r["question_id"])] = [int(c["answer_id"]) for c in r["retrieved_context"]]
    return out


def apply_decision(base: dict, repfix: dict, alphas=STAGE2_ALPHAS) -> dict:
    """Aturan R1. base/repfix: {alpha: metrics dari compute_alpha_metrics}."""
    checks = []
    for a in alphas:
        b, r = base[a], repfix[a]
        d_pct = r["pct_questions_with_relevant"] - b["pct_questions_with_relevant"]
        d_score = r["mean_relevance_score"] - b["mean_relevance_score"]
        ok = d_pct >= -PCT_TIE and d_score >= -SCORE_TIE
        checks.append({"alpha": a, "delta_pct_questions_with_relevant": round(d_pct, 2),
                       "delta_mean_relevance_score": round(d_score, 4), "ok": ok})
    include = all(c["ok"] for c in checks)
    return {"include_reputation": include, "checks": checks,
            "label": "SERTAKAN reputasi" if include else "JANGAN sertakan reputasi"}


def unjudged_items(records_by_alpha: dict[float, list[dict]], judged_keys: set, text_hash) -> list[dict]:
    """Item unik (question_id, answer_id, text_hash) yang belum punya label."""
    seen, out = set(), []
    for a in sorted(records_by_alpha):
        for rec in records_by_alpha[a]:
            for c in rec["retrieved_context"]:
                key = (rec["question_id"], int(c["answer_id"]), text_hash(c["chunk_text"]))
                if key in judged_keys or key in seen:
                    continue
                seen.add(key)
                out.append({"question_id": rec["question_id"], "answer_id": int(c["answer_id"]),
                            "text_hash": key[2], "chunk_text": c["chunk_text"],
                            "title": rec["title"], "tags": rec["tags"], "alpha_first_seen": a})
    return out


def parse_stage1_report(path: Path) -> dict[float, tuple[float, float]]:
    """{alpha: (pct_questions_with_relevant, mean_relevance_score)} dari tabel
    utama laporan stage 1 (baris '| alpha | pct | ... | mean score | ...')."""
    out = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            try:
                out.setdefault(float(cells[0]), (float(cells[1]), float(cells[5])))
            except ValueError:
                continue
    return out


def _load_judged(paths) -> dict:
    import llm_judge_context_relevance_v1 as ctxrel
    recs = ctxrel.load_ctxrel_records([Path(p) for p in paths], prompt_version=CTXREL_VERSION)
    return {k: v for k, v in recs.items() if v.get("judge_id") == JUDGE_ID}


def verify_applied(args, sample_df, all_answer_ids_map, driver, database) -> int:
    """Graf sudah diperbarui: trustScore tersimpan = repfix. Maka peringkat
    `base` dari graf harus sama persis dengan file stage1_repfix_alpha*.jsonl."""
    from embedding_cache import EmbeddingCache
    with driver.session(database=database) as s:
        rec = s.run("MATCH (a:Answer) RETURN count(a) AS n, "
                    "sum(CASE WHEN a.trustScoreVersion = 'v2-reputation' THEN 1 ELSE 0 END) AS n_v2").single()
    print(f"      Answer: {rec['n']:,}; bertanda v2-reputation: {rec['n_v2']:,}")
    if rec["n"] != rec["n_v2"]:
        print("[STOP] Belum semua Answer diperbarui. Jalankan 13_apply_reputation_trust.py --apply.")
        driver.close()
        return 1
    expected = {a: selected_ids(OUT_DIR / f"stage1_repfix_alpha{str(a).replace('.', '-')}.jsonl") for a in ALPHAS}
    faiss_index, faiss_ids, faiss_emb, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")
    cache = EmbeddingCache(args.embedding_cache_path)
    mismatches = []
    try:
        for _, row in sample_df.iterrows():
            qid = int(row["Id"])
            q_emb, graph_c, exp_c = rac.retrieve_candidates(
                row, faiss_index, faiss_ids, faiss_emb, id_to_row, all_answer_ids_map,
                driver, database, embed_model, 3, 3)
            for a in ALPHAS:
                base = rac.rank_variant("base", a, graph_c, exp_c, {}, embed_model, q_emb, cache)
                if [int(c["answer_id"]) for c in base] != expected[a].get(qid):
                    mismatches.append((qid, a))
    finally:
        cache.close()
        driver.close()
    if mismatches:
        print(f"[GAGAL] {len(mismatches)} (qid, alpha) berbeda dari repfix, contoh {mismatches[:5]}.")
        return 1
    print("[OK] Graf yang diperbarui mereproduksi stage 1 repfix: identik untuk 50 pertanyaan x 5 alpha.")
    return 0


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> int:
    import llm_judge_context_relevance_v1 as ctxrel
    import stage1_sweep as s1
    from ctxrel_sweep import select_stage1_candidates
    from embedding_cache import EmbeddingCache, default_cache_path

    p = argparse.ArgumentParser()
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    p.add_argument("--embedding-cache-path", default=str(default_cache_path()))
    p.add_argument("--judge", action="store_true", help="Nilai item baru dengan ctxrel-v2 primary (berbayar).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--verify-applied", action="store_true",
                   help="Setelah 13_apply_reputation_trust.py --apply: top-5 dari graf yang sudah diperbaiki "
                        "harus identik dengan stage1_repfix_alpha*.jsonl (tanpa panggilan API).")
    args = p.parse_args()
    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET/ANSWERS_PARQUET wajib (flag atau .env).")
        return 1

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_files = {a: stage1_file_for_alpha(a) for a in ALPHAS}
    stage1_selected = {a: selected_ids(f) for a, f in stage1_files.items()}

    print("[1/6] Dev set (n=50, posisi 385-434, seed 42)...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, 50, 42, 385)

    print("[2/6] Neo4j (read-only) + asumsi graf + FAISS + model embedding...")
    driver, database = cg.connect_neo4j(print)
    if args.verify_applied:
        return verify_applied(args, sample_df, all_answer_ids_map, driver, database)
    graph_info = rac.check_graph_assumption(driver, database)
    if not graph_info["ok"]:
        print(f"[STOP] Asumsi graf tidak terpenuhi: {graph_info}")
        driver.close()
        return 1
    faiss_index, faiss_ids, faiss_emb, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")
    cache = EmbeddingCache(args.embedding_cache_path)

    repfix_records = {a: [] for a in ALPHAS}
    mismatches = []
    try:
        print("[3/6] Retrieval sekali per pertanyaan; peringkat base vs repfix untuk 5 alpha...")
        for _, row in sample_df.iterrows():
            qid = int(row["Id"])
            q_emb, graph_c, exp_c = rac.retrieve_candidates(
                row, faiss_index, faiss_ids, faiss_emb, id_to_row, all_answer_ids_map,
                driver, database, embed_model, 3, 3)
            aids = sorted({int(c["answer_id"]) for c in graph_c + exp_c})
            weights = cg.fetch_author_trust(driver, database, aids)
            for a in ALPHAS:
                base = rac.rank_variant("base", a, graph_c, exp_c, weights, embed_model, q_emb, cache)
                if [int(c["answer_id"]) for c in base] != stage1_selected[a].get(qid):
                    mismatches.append((qid, a))
                fixed = rac.rank_variant("rep_in_trustscore", a, graph_c, exp_c, weights, embed_model, q_emb, cache)
                repfix_records[a].append({"question_id": qid, "title": row["Title"], "tags": row["Tags"],
                                          "retrieved_context": fixed})
    finally:
        cache.close()
        driver.close()

    if mismatches:
        print(f"[STOP] Top-5 base tidak sama dengan file stage 1 pada {len(mismatches)} (qid, alpha), "
              f"contoh {mismatches[:5]}. Retrieval tidak lagi mereproduksi stage 1; perbandingan tidak sah.")
        return 1
    print("      Reproduksi OK: top-5 base identik dengan stage 1 untuk 50 pertanyaan x 5 alpha.")

    repfix_files = {}
    for a, recs in repfix_records.items():
        path = OUT_DIR / f"stage1_repfix_alpha{str(a).replace('.', '-')}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        repfix_files[a] = path

    print("[4/6] Mencocokkan dengan penilaian stage 1 (ctxrel-v2, primary)...")
    judged = _load_judged([STAGE1_JUDGMENTS, NEW_JUDGMENTS])
    todo = unjudged_items(repfix_records, set(judged), ctxrel._text_hash)
    print(f"      Item repfix belum dinilai: {len(todo)} (sisanya memakai label stage 1).")

    if todo and not args.judge:
        print("[RENCANA] Jalankan ulang dengan --judge untuk menilai item tersebut "
              f"({len(todo)} panggilan, judge primary). Pastikan Amendment 5 sudah di-commit.")
        return 0

    if todo:
        print(f"[5/6] Menilai {len(todo)} item dengan {CTXREL_VERSION} / {JUDGE_ID}...")
        from judge_clients import get_judge_client
        client, config = get_judge_client(JUDGE_ID)
        bodies = s1._load_question_bodies_text(args.questions_parquet, sorted({i["question_id"] for i in todo}))

        def _one(item):
            judge_item = {**item, "tags": ctxrel.format_tags(item["tags"]),
                          "source_files": [str(repfix_files[item["alpha_first_seen"]])]}
            return ctxrel.judge_one(judge_item, JUDGE_ID, client, config,
                                    bodies.get(item["question_id"], ""), prompt_version=CTXREL_VERSION)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rec in pool.map(_one, todo):
                ctxrel.write_record(NEW_JUDGMENTS, rec)
        judged = _load_judged([STAGE1_JUDGMENTS, NEW_JUDGMENTS])
        still = unjudged_items(repfix_records, {k for k, v in judged.items() if v.get("label")}, ctxrel._text_hash)
        if still:
            print(f"[STOP] {len(still)} item gagal dinilai; jalankan ulang --judge (aman, resume).")
            return 1

    print("[6/6] Metrik stage 1 dan keputusan...")
    sel_base = {a: {q: set(v) for q, v in stage1_selected[a].items()} for a in ALPHAS}
    sel_fix = {a: {r["question_id"]: {int(c["answer_id"]) for c in r["retrieved_context"]}
                   for r in recs} for a, recs in repfix_records.items()}
    base_m = {a: s1.compute_alpha_metrics(str(stage1_files[a]), judged, sel_base, a) for a in ALPHAS}
    fix_m = {a: s1.compute_alpha_metrics(str(repfix_files[a]), judged, sel_fix, a) for a in ALPHAS}
    reported = parse_stage1_report(archive_paths.resolve(RESULTS / "stage1_report_ctxrel_v2_primary_n50_seed42.md"))
    off = [a for a in ALPHAS if reported.get(a) != (base_m[a]["pct_questions_with_relevant"],
                                                    base_m[a]["mean_relevance_score"])]
    if off:
        print(f"[STOP] Metrik base hasil hitung ulang tidak sama dengan laporan stage 1 untuk alpha {off}.")
        return 1
    print("      Metrik base identik dengan laporan stage 1.")
    sel_base_out = select_stage1_candidates(base_m)
    sel_fix_out = select_stage1_candidates(fix_m)
    decision = apply_decision(base_m, fix_m)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "# Stage 1 ulang dengan perbaikan reputasi (dev set, ctxrel-v2 primary)", "",
        f"- Waktu: {stamp}  |  git commit: `{rac._git_commit(Path(__file__).resolve().parents[2])}`",
        f"- Reproduksi base vs file stage 1: identik (50 x {len(ALPHAS)})",
        f"- Item baru yang dinilai: {len(todo)}  |  durasi {round(time.time() - t0, 1)} detik", "",
        "| alpha | % q >=1 RELEVANT (base -> repfix) | mean score (base -> repfix) | % accepted (base -> repfix) |",
        "|---|---|---|---|",
    ]
    for a in ALPHAS:
        b, r = base_m[a], fix_m[a]
        lines.append(f"| {a} | {b['pct_questions_with_relevant']} -> {r['pct_questions_with_relevant']} | "
                     f"{b['mean_relevance_score']} -> {r['mean_relevance_score']} | "
                     f"{b['pct_accepted_selected']} -> {r['pct_accepted_selected']} |")
    lines += [
        "", f"- Selection helper base: top_two {sel_base_out['top_two']} (keep_current {sel_base_out['keep_current']})",
        f"- Selection helper repfix: top_two {sel_fix_out['top_two']} (keep_current {sel_fix_out['keep_current']})", "",
        "## Aturan R1 (alpha 0.25 dan 0.5; margin seri 2 pp / 0.02)", "",
    ]
    for c in decision["checks"]:
        lines.append(f"- alpha={c['alpha']}: delta % q = {c['delta_pct_questions_with_relevant']}, "
                     f"delta skor = {c['delta_mean_relevance_score']} -> {'lolos' if c['ok'] else 'GAGAL'}")
    lines += ["", f"**Keputusan: {decision['label']}**", ""]
    md = "\n".join(lines) + "\n"
    (OUT_DIR / f"stage1_repfix_report_{stamp}.md").write_text(md)
    (OUT_DIR / f"stage1_repfix_report_{stamp}.json").write_text(json.dumps({
        "graph_check": graph_info, "base": base_m, "repfix": fix_m, "selection_base": sel_base_out,
        "selection_repfix": sel_fix_out, "decision": decision, "n_new_judgments": len(todo),
    }, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
