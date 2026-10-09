"""
reputation_ablation_check.py
=====================================
Uji TANPA BIAYA API (tidak ada panggilan LLM, Neo4j hanya DIBACA): apakah
memasukkan reputasi kontributor mengubah konteks top-5 Kondisi C?

LATAR BELAKANG (temuan 2026-10-09)
------------------------------------------------------------
7_build_knowledge_graph.py menghitung
    trustScore_A = 0.5*norm(score) + 0.3*accepted + 0.2*norm(reputation)
tetapi saat KG dibangun (log 20260829T075916Z) FilteredUsers.csv tidak
ditemukan, sehingga reputation kosong untuk SEMUA jawaban. minmax_normalize()
pada deret yang seluruhnya 0 mengembalikan 0.5, jadi suku reputasi menjadi
KONSTANTA 0.2*0.5 = 0.1 untuk setiap jawaban (cocok dengan log:
"trustScore Answer -- min=0.100"). 9_load_author_trust.py kemudian menulis
User node + edge AUTHOR_TRUST, tetapi TIDAK menghitung ulang trustScore.
Akibatnya reputasi tidak pernah memengaruhi urutan kandidat.

APA YANG DIUJI
------------------------------------------------------------
Untuk setiap pertanyaan DEV SET (posisi 385-434, sama persis dengan
stage1_sweep.py -- sampel uji 1-384 TIDAK disentuh), kandidat retrieval
(anchoring -> traversal -> semantic expansion) diambil SATU KALI, lalu
diperingkat ulang dengan fuse_and_rank() v3 untuk setiap varian x alpha:

  base               : trustScore apa adanya di Neo4j (reputasi = konstanta).
  rep_in_trustscore  : PERBAIKAN yang diusulkan -- suku reputasi konstanta
                       diganti bobot edge AUTHOR_TRUST jawaban tsb:
                         trust_A' = trust_A - 0.2*0.5 + 0.2*author_weight
                       (author_weight = log1p(rep)/log1p(rep_P99), dipotong
                       [0,1]; 0 kalau jawaban tidak punya edge AUTHOR_TRUST).
                       Bobot efektif reputasi pada skor akhir ~= 0.06*alpha.
  author_switch      : saklar eksploratif yang SUDAH ADA (use_author_trust,
                       beta=0.3, docs/DECISION_C_SCORING.md) di atas base.
                       Bobot efektif reputasi pada skor akhir ~= 0.3*alpha.
                       Disertakan sbg pembanding "batas atas" pengaruh.

Setiap varian dibandingkan dgn `base` pada alpha yang SAMA. Karena kumpulan
kandidatnya identik, setiap perbedaan top-5 MURNI berasal dari reputasi.
alpha=0 (relevansi saja) wajib menghasilkan 0 perubahan -- dipakai sbg
sanity check otomatis.

KRITERIA BACA (ditetapkan SEBELUM run, deskriptif -- keputusan akhir
tetap di peneliti)
------------------------------------------------------------
  Untuk varian rep_in_trustscore, pada alpha in {0.25, 0.5, 0.75, 1}:
  - "NETRAL"       : persentase pertanyaan dgn himpunan top-5 berubah
                     <= 5% (<= 2 dari 50) pada SEMUA alpha.
  - "BERPENGARUH"  : > 5% pada sedikitnya satu alpha.
  Perubahan BUKAN berarti lebih baik atau lebih buruk -- skrip ini tidak
  menilai relevansi. Kalau BERPENGARUH, langkah berikutnya adalah menilai
  relevansi item yang berubah saja (ctxrel-v2), yang murah karena hanya
  item yang berbeda yang perlu dinilai.

KELUARAN (llm/c_graphrag/results/reputation_check/)
------------------------------------------------------------
  reputation_check_<ts>.md    : ringkasan untuk dibaca / disalin ke Bab IV-V
  reputation_check_<ts>.json  : angka lengkap + konfigurasi + git commit
  reputation_check_<ts>.csv   : detail per pertanyaan x varian x alpha
Skrip ini TIDAK menulis run_history.jsonl dan BUKAN hasil seleksi alpha.

CARA PAKAI (dari root repo; .env sama dgn c_graphrag.py)
------------------------------------------------------------
    python3 llm/c_graphrag/reputation_ablation_check.py
    python3 llm/c_graphrag/reputation_ablation_check.py --n-dev 10   # uji cepat

Catatan memori (M2 8GB): sama seperti stage1_sweep.py -- tutup Finder pada
folder logs/results selama run. Semua embedding kandidat memakai
EmbeddingCache yang sama dgn stage1_sweep.py (kunci: answer_id + teks,
tidak bergantung alpha/varian), jadi run berikutnya akan lebih cepat.
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import c_graphrag as cg  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("base", "rep_in_trustscore", "author_switch")
# Bobot suku reputasi di trustScore_A dan nilai konstanta yang tersimpan
# akibat reputasi kosong saat build (minmax_normalize(all zeros) == 0.5).
REPUTATION_COEF = 0.2
STORED_REPUTATION_NORM = 0.5
NEUTRAL_THRESHOLD = 0.05  # <= 5% pertanyaan berubah -> "NETRAL"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "reputation_check"


# ---------------------------------------------------------------------
# Logika murni (tanpa Neo4j/FAISS) -- diuji di tests/test_reputation_ablation_check.py
# ---------------------------------------------------------------------

def apply_reputation_fix(candidates: list[dict], author_weights: dict[int, float]) -> list[dict]:
    """Salinan kandidat dgn answer_trust_score' = trust - 0.2*0.5 + 0.2*author_weight.
    Kandidat tanpa edge AUTHOR_TRUST memakai author_weight 0. Hasil dipotong
    ke [0, 1] (secara matematis sudah di dalam rentang, ini hanya pengaman
    pembulatan floating point)."""
    out = []
    for c in candidates:
        c2 = copy.deepcopy(c)
        stored = c2.get("answer_trust_score") or 0.0
        aw = author_weights.get(int(c2["answer_id"]), 0.0)
        fixed = stored - REPUTATION_COEF * STORED_REPUTATION_NORM + REPUTATION_COEF * aw
        c2["answer_trust_score"] = min(max(fixed, 0.0), 1.0)
        out.append(c2)
    return out


def attach_author_weight(candidates: list[dict], author_weights: dict[int, float]) -> list[dict]:
    """Salinan kandidat dgn field author_weight (dipakai saklar use_author_trust)."""
    out = []
    for c in candidates:
        c2 = copy.deepcopy(c)
        c2["author_weight"] = author_weights.get(int(c2["answer_id"]), 0.0)
        out.append(c2)
    return out


def rank_variant(variant: str, alpha: float, graph_c: list[dict], exp_c: list[dict],
                 author_weights: dict[int, float], embed_model, query_emb, embedding_cache,
                 top_k: int = 5, token_chunk_limit: int = 400) -> list[dict]:
    """Top-k satu varian pada satu alpha, lewat fuse_and_rank() v3 asli."""
    use_author_trust = False
    if variant == "base":
        g, e = copy.deepcopy(graph_c), copy.deepcopy(exp_c)
    elif variant == "rep_in_trustscore":
        g, e = apply_reputation_fix(graph_c, author_weights), apply_reputation_fix(exp_c, author_weights)
    elif variant == "author_switch":
        g, e = attach_author_weight(graph_c, author_weights), attach_author_weight(exp_c, author_weights)
        use_author_trust = True
    else:
        raise ValueError(f"Varian tidak dikenal: {variant}")
    final, _ = cg.fuse_and_rank(
        g, e, top_k, token_chunk_limit,
        c_retrieval_version="v3", alpha=alpha,
        embed_model=embed_model, query_emb=query_emb, embedding_cache=embedding_cache,
        use_author_trust=use_author_trust,
    )
    return final


def compare_topk(base: list[dict], other: list[dict], author_weights: dict[int, float]) -> dict:
    """Bandingkan dua daftar top-k (urutan penting utk rank-1/urutan)."""
    b_ids = [int(c["answer_id"]) for c in base]
    o_ids = [int(c["answer_id"]) for c in other]
    b_set, o_set = set(b_ids), set(o_ids)
    union = b_set | o_set
    entered = [c for c in other if int(c["answer_id"]) not in b_set]
    left = [c for c in base if int(c["answer_id"]) not in o_set]
    return {
        "set_changed": b_set != o_set,
        "rank1_changed": (b_ids[:1] != o_ids[:1]),
        "order_changed": b_ids != o_ids,
        "jaccard": (len(b_set & o_set) / len(union)) if union else 1.0,
        "n_swapped": len(entered),
        "entered_ids": [int(c["answer_id"]) for c in entered],
        "left_ids": [int(c["answer_id"]) for c in left],
        "entered_accepted": sum(1 for c in entered if c.get("is_accepted")),
        "left_accepted": sum(1 for c in left if c.get("is_accepted")),
        "entered_author_weight": [author_weights.get(int(c["answer_id"]), 0.0) for c in entered],
        "left_author_weight": [author_weights.get(int(c["answer_id"]), 0.0) for c in left],
    }


def summarize(rows: list[dict], n_questions: int) -> dict:
    """Agregasi per (varian, alpha) dari baris hasil compare_topk."""
    summary = {}
    for variant in VARIANTS[1:]:
        for alpha in ALPHAS:
            sel = [r for r in rows if r["variant"] == variant and r["alpha"] == alpha]
            if not sel:
                continue
            n = len(sel)
            entered_aw = [w for r in sel for w in r["entered_author_weight"]]
            left_aw = [w for r in sel for w in r["left_author_weight"]]
            summary[f"{variant}@{alpha}"] = {
                "variant": variant, "alpha": alpha, "n_questions": n,
                "n_set_changed": sum(r["set_changed"] for r in sel),
                "pct_set_changed": round(100 * sum(r["set_changed"] for r in sel) / n, 1),
                "n_rank1_changed": sum(r["rank1_changed"] for r in sel),
                "n_order_changed": sum(r["order_changed"] for r in sel),
                "mean_jaccard": round(float(np.mean([r["jaccard"] for r in sel])), 4),
                "total_swapped_items": sum(r["n_swapped"] for r in sel),
                "entered_accepted": sum(r["entered_accepted"] for r in sel),
                "left_accepted": sum(r["left_accepted"] for r in sel),
                "mean_author_weight_entered": round(float(np.mean(entered_aw)), 4) if entered_aw else None,
                "mean_author_weight_left": round(float(np.mean(left_aw)), 4) if left_aw else None,
            }
    return summary


def verdict(summary: dict, threshold: float = NEUTRAL_THRESHOLD) -> dict:
    """Terapkan kriteria baca (lihat docstring modul) pada rep_in_trustscore."""
    worst = max(
        (v for k, v in summary.items() if v["variant"] == "rep_in_trustscore" and v["alpha"] > 0),
        key=lambda v: v["pct_set_changed"], default=None,
    )
    if worst is None:
        return {"label": "TIDAK ADA DATA", "worst": None}
    label = "NETRAL" if worst["pct_set_changed"] <= threshold * 100 else "BERPENGARUH"
    return {"label": label, "worst_alpha": worst["alpha"], "worst_pct": worst["pct_set_changed"],
            "threshold_pct": threshold * 100}


# ---------------------------------------------------------------------
# Akses data (Neo4j hanya baca)
# ---------------------------------------------------------------------

def check_graph_assumption(driver, database) -> dict:
    """Memastikan asumsi temuan masih berlaku: tidak ada Answer dgn
    authorReputation != 0 dan trustScore minimum ~= 0.1. Kalau KG sudah
    pernah diperbaiki, varian rep_in_trustscore akan salah hitung."""
    with driver.session(database=database) as s:
        rec = s.run(
            """
            MATCH (a:Answer)
            RETURN count(a) AS n, min(a.trustScore) AS min_trust, max(a.trustScore) AS max_trust,
                   sum(CASE WHEN coalesce(a.authorReputation, 0) <> 0 THEN 1 ELSE 0 END) AS n_rep_nonzero
            """
        ).single()
        n_edges = s.run("MATCH (:Answer)-[r:AUTHOR_TRUST]->(:User) RETURN count(r) AS n").single()["n"]
    info = dict(rec)
    info["n_author_trust_edges"] = n_edges
    info["ok"] = (info["n_rep_nonzero"] == 0
                  and info["min_trust"] is not None
                  and abs(info["min_trust"] - REPUTATION_COEF * STORED_REPUTATION_NORM) < 1e-3)
    return info


def retrieve_candidates(row, faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
                        driver, database, embed_model, n_anchor: int, n_expansion: int):
    """Tahap a-c Kondisi C, PERSIS seperti process_sample() (default resmi:
    max_hops=2, semua edge type, semantic expansion aktif)."""
    qid = int(row["Id"])
    exclude_q = {qid}
    exclude_a = set(all_answer_ids_map.get(qid, []))
    exclude_a.add(int(row["AcceptedAnswerId"]))
    if qid in id_to_row:
        query_emb = np.ascontiguousarray(faiss_embeddings[id_to_row[qid]:id_to_row[qid] + 1])
    else:
        query_emb = cg.embed_query(f"{row['Title']} {str(row['Body'])[:1000]}", embed_model)

    anchors = cg.anchor_via_vector_search(query_emb, faiss_index, faiss_ids, n_anchor, exclude_q)
    anchor_ids = [a for a, _ in anchors]
    graph_c = cg.traverse_graph(driver, database, anchor_ids, exclude_q, exclude_a) if anchor_ids else []
    exp_c = cg.semantic_expansion(
        driver, database, graph_c, faiss_index, faiss_ids, embed_model, n_expansion,
        exclude_q | set(anchor_ids), exclude_q, exclude_a,
    )
    return query_emb, graph_c, exp_c


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------

def build_markdown(summary: dict, verdict_info: dict, graph_info: dict, meta: dict) -> str:
    lines = [
        "# Uji pengaruh reputasi kontributor pada top-5 Kondisi C",
        "",
        f"- Waktu: {meta['finished_at']}  |  git commit: `{meta['git_commit']}`",
        f"- Dev set: {meta['n_questions']} pertanyaan (posisi {meta['dev_offset']}-"
        f"{meta['dev_offset'] + meta['n_questions'] - 1}), seed {meta['seed']}, top_k {meta['top_k']}",
        f"- Tanpa panggilan LLM. Neo4j hanya dibaca. Durasi {meta['duration_sec']} detik.",
        "",
        "## Pemeriksaan asumsi graf",
        "",
        f"- Answer: {graph_info['n']:,}; trustScore min {graph_info['min_trust']:.3f}, "
        f"max {graph_info['max_trust']:.3f}",
        f"- Answer dengan authorReputation != 0: {graph_info['n_rep_nonzero']:,}",
        f"- Edge AUTHOR_TRUST: {graph_info['n_author_trust_edges']:,}",
        f"- Cakupan AUTHOR_TRUST pada kandidat: {meta['candidate_author_coverage_pct']}% "
        f"({meta['n_candidates_unique']:,} kandidat unik)",
        f"- Asumsi terpenuhi: **{'ya' if graph_info['ok'] else 'TIDAK'}**",
        "",
        "## Hasil (dibandingkan dengan `base` pada alpha yang sama)",
        "",
        "| Varian | alpha | Top-5 berubah | Rank-1 berubah | Urutan berubah | Rata-rata Jaccard "
        "| Item tertukar | Accepted masuk/keluar | Rata-rata author_weight masuk/keluar |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in summary.values():
        aw_in = "-" if v["mean_author_weight_entered"] is None else f"{v['mean_author_weight_entered']:.3f}"
        aw_out = "-" if v["mean_author_weight_left"] is None else f"{v['mean_author_weight_left']:.3f}"
        lines.append(
            f"| {v['variant']} | {v['alpha']} | {v['n_set_changed']}/{v['n_questions']} "
            f"({v['pct_set_changed']}%) | {v['n_rank1_changed']} | {v['n_order_changed']} | "
            f"{v['mean_jaccard']:.3f} | {v['total_swapped_items']} | "
            f"{v['entered_accepted']}/{v['left_accepted']} | {aw_in}/{aw_out} |"
        )
    lines += [
        "",
        "## Kriteria baca (ditetapkan sebelum run)",
        "",
        f"Varian `rep_in_trustscore`, alpha > 0: NETRAL jika top-5 berubah pada <= "
        f"{verdict_info.get('threshold_pct', NEUTRAL_THRESHOLD * 100):.0f}% pertanyaan di semua alpha; "
        "BERPENGARUH jika lebih.",
        "",
        f"**Hasil: {verdict_info['label']}** (perubahan terbesar {verdict_info.get('worst_pct')}% "
        f"pada alpha={verdict_info.get('worst_alpha')}).",
        "",
        "Catatan: perubahan tidak sama dengan perbaikan. Skrip ini tidak menilai relevansi. "
        "Jika BERPENGARUH, nilai relevansi item yang masuk/keluar dengan ctxrel-v2.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> int:
    import stage1_sweep as s1
    from embedding_cache import EmbeddingCache, default_cache_path

    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    p.add_argument("--embedding-cache-path", default=str(default_cache_path()))
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--dev-offset", type=int, default=385)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-anchor", type=int, default=int(os.getenv("N_ANCHOR", 3)))
    p.add_argument("--n-semantic-expansion", type=int, default=int(os.getenv("N_SEMANTIC_EXPANSION", 3)))
    p.add_argument("--force", action="store_true",
                   help="Tetap jalan walau asumsi graf tidak terpenuhi (tidak disarankan).")
    args = p.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET/ANSWERS_PARQUET wajib (flag atau .env).")
        return 1

    t0 = time.time()
    print(f"[1/5] Menyusun dev set (n={args.n_dev}, dev_offset={args.dev_offset}, seed={args.seed})...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, args.n_dev, args.seed, args.dev_offset)
    print(f"      {len(sample_df)} pertanyaan dev siap.")

    print("[2/5] Koneksi Neo4j + cek asumsi graf (read-only)...")
    driver, database = cg.connect_neo4j(print)
    graph_info = check_graph_assumption(driver, database)
    print(f"      {graph_info}")
    if not graph_info["ok"] and not args.force:
        print("[STOP] Asumsi tidak terpenuhi (KG tampaknya sudah diperbaiki atau berbeda). "
              "Varian rep_in_trustscore akan salah hitung. Pakai --force hanya jika paham risikonya.")
        driver.close()
        return 1

    print("[3/5] Memuat FAISS cache + model embedding (CPU)...")
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")
    embedding_cache = EmbeddingCache(args.embedding_cache_path)

    rows, candidate_ids, candidate_with_author = [], set(), set()
    try:
        print(f"[4/5] Retrieval sekali per pertanyaan, lalu ranking ulang {len(VARIANTS)} varian x "
              f"{len(ALPHAS)} alpha...")
        for i, row in sample_df.iterrows():
            qid = int(row["Id"])
            query_emb, graph_c, exp_c = retrieve_candidates(
                row, faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
                driver, database, embed_model, args.n_anchor, args.n_semantic_expansion)
            aids = sorted({int(c["answer_id"]) for c in graph_c + exp_c})
            author_weights = cg.fetch_author_trust(driver, database, aids)
            candidate_ids.update(aids)
            candidate_with_author.update(a for a in aids if a in author_weights)

            for alpha in ALPHAS:
                base = rank_variant("base", alpha, graph_c, exp_c, author_weights, embed_model,
                                    query_emb, embedding_cache, args.top_k)
                for variant in VARIANTS[1:]:
                    other = rank_variant(variant, alpha, graph_c, exp_c, author_weights, embed_model,
                                         query_emb, embedding_cache, args.top_k)
                    cmp = compare_topk(base, other, author_weights)
                    if alpha == 0.0 and cmp["order_changed"]:
                        raise RuntimeError(f"Sanity check gagal: alpha=0 berubah untuk q{qid} ({variant}).")
                    rows.append({"question_id": qid, "variant": variant, "alpha": alpha,
                                 "candidate_pool_size": len(aids), **cmp})
            print(f"      [{i + 1}/{len(sample_df)}] q{qid}: {len(aids)} kandidat, "
                  f"{sum(1 for a in aids if a in author_weights)} punya AUTHOR_TRUST")
    finally:
        embedding_cache.close()
        driver.close()

    print("[5/5] Menyusun laporan...")
    summary = summarize(rows, len(sample_df))
    verdict_info = verdict(summary)
    finished = datetime.now(timezone.utc)
    meta = {
        "finished_at": finished.isoformat(timespec="seconds"),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "n_questions": len(sample_df), "dev_offset": args.dev_offset, "seed": args.seed,
        "top_k": args.top_k, "n_anchor": args.n_anchor, "n_semantic_expansion": args.n_semantic_expansion,
        "alphas": list(ALPHAS), "variants": list(VARIANTS),
        "reputation_coef": REPUTATION_COEF, "stored_reputation_norm": STORED_REPUTATION_NORM,
        "author_trust_beta": cg.AUTHOR_TRUST_BETA,
        "n_candidates_unique": len(candidate_ids),
        "candidate_author_coverage_pct": round(100 * len(candidate_with_author) / max(len(candidate_ids), 1), 1),
        "duration_sec": round(time.time() - t0, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"reputation_check_{finished.strftime('%Y%m%dT%H%M%SZ')}"
    Path(f"{stem}.json").write_text(json.dumps(
        {"meta": meta, "graph_check": graph_info, "summary": summary, "verdict": verdict_info},
        indent=2, default=str))
    with open(f"{stem}.csv", "w", newline="") as f:
        fields = ["question_id", "variant", "alpha", "candidate_pool_size", "set_changed", "rank1_changed",
                  "order_changed", "jaccard", "n_swapped", "entered_ids", "left_ids",
                  "entered_accepted", "left_accepted", "entered_author_weight", "left_author_weight"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in fields})
    md = build_markdown(summary, verdict_info, graph_info, meta)
    Path(f"{stem}.md").write_text(md)
    print(md)
    print(f"Laporan: {stem}.md / .json / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
