"""
anchors_stage1_recheck.py
=====================================
Stage 1 again on the dev set with ANSWERABLE-ONLY entity anchoring (C retrieval v4, Amendment 17), for alpha in
{0, 0.25, 0.5, 0.75, 1}. Same pattern as expansion_stage1_recheck.py: dev set (positions 385-434, n = 50, seed 42),
retrieval-only C (no generation), leakage mode "thread", the same judge (ctxrel-v2, Llama-3.3-70B-Instruct,
temperature 0) and the same label cache; only items without a cached label are judged, and only with --run-judge and
under --max-cost-usd (default 0.10; above it the script stops BEFORE judging). The unchanged selection rule
(ctxrel_sweep.select_stage1_candidates) is applied and the result is accepted whatever it is.

There is no reproduction check against an earlier result of record (the anchors changed on purpose). The previous
stage-1 result (plain anchors, C v3; the newest expansion_check_thread report) is shown next to the new one.

OUTPUT: llm/c_graphrag/results/anchors_check/ (new file names; nothing is overwritten)

    python3 llm/c_graphrag/anchors_stage1_recheck.py               # retrieval + plan, 0 API calls
    python3 llm/c_graphrag/anchors_stage1_recheck.py --run-judge   # judge new items (<= $0.10)
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import c_graphrag as cg  # noqa: E402
import expansion_stage1_recheck as ex  # noqa: E402
import logscore_stage1_recheck as lsr  # noqa: E402
import reputation_stage1_recheck as rsr  # noqa: E402

ALPHAS = ex.ALPHAS
N_DEV = ex.N_DEV
OUT_DIR = rsr.RESULTS / "anchors_check"
NEW_JUDGMENTS = OUT_DIR / "ctxrel_v2_anchors_primary.jsonl"
JUDGMENT_FILES = (*ex.JUDGMENT_FILES, NEW_JUDGMENTS)
PREVIOUS_REPORT_DIR = rsr.RESULTS / "expansion_check_thread"  # C v3, plain anchors, thread mode


def previous_result() -> dict:
    reports = sorted(PREVIOUS_REPORT_DIR.glob("stage1_expansion_report_*.json"))
    if not reports:
        return {}
    d = json.loads(reports[-1].read_text())
    return {"report": reports[-1].name, "metrics": {float(a): m for a, m in d["new"].items()},
            "selection": d["selection_new"]}


def anchor_stats(records: list[dict]) -> dict:
    found = [r.get("anchors_found") for r in records]
    return {"questions_with_fewer_than_3_seeds": sum(1 for f in found if f is not None and f < 3),
            "skipped_unanswered_mean": round(float(np.mean([r.get("anchors_skipped_unanswered", 0) for r in records])), 2)}


def main() -> int:
    import tempfile as _t  # noqa: F401

    import llm_judge_context_relevance_v1 as ctxrel
    import stage1_sweep as s1
    from ctxrel_sweep import select_stage1_candidates
    from embedding_cache import EmbeddingCache
    from judge_clients import JUDGE_REGISTRY

    from llm.common.leakage import LeakageContext, default_mode

    p = argparse.ArgumentParser(description="Stage-1 recheck with answerable-only anchors (Amendment 17).")
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv("KG_WORKSPACE_DIR"))
    p.add_argument("--embedding-cache-path", default=None)
    p.add_argument("--run-judge", action="store_true", help="judge new items (paid) if under --max-cost-usd")
    p.add_argument("--max-cost-usd", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if ctxrel.REQUEST_TEMPERATURE != 0.0:
        print(f"[STOP] ctxrel judge temperature is {ctxrel.REQUEST_TEMPERATURE}, not 0.")
        return 1
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Dev set (n={N_DEV}, positions 385-434, seed 42)...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, N_DEV, 42, 385)

    print("[2/5] Neo4j (read-only) + FAISS + embedding model (CPU)...")
    driver, database = cg.connect_neo4j(print)
    faiss_index, faiss_ids, faiss_emb, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    from embedding_cache import default_cache_path
    cache = EmbeddingCache(args.embedding_cache_path or str(default_cache_path()))
    leakage = LeakageContext.from_mode(default_mode())
    cg.log = lambda *a, **k: None

    runs = {}
    try:
        print(f"[3/5] Retrieval-only per alpha, C v4 (answerable anchors), leakage {leakage.mode}...")
        for a in ALPHAS:
            with tempfile.TemporaryDirectory() as tmp:
                recs, _ = cg.process_sample(
                    sample_df, None, None, embed_model, driver, database, faiss_index, faiss_ids, faiss_emb,
                    id_to_row, all_answer_ids_map, 5, 3, 3, 400, "none", Path(tmp) / "o.jsonl",
                    retrieval_only=True, c_retrieval_version="v4", alpha=a, embedding_cache=cache,
                    leakage=leakage, expansion_require_answer=True)
            runs[a] = recs
    finally:
        cache.close()
        driver.close()

    files = {}
    for a, recs in runs.items():
        path = OUT_DIR / f"stage1_anchors_v4_alpha{str(a).replace('.', '-')}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        files[a] = path

    print(f"[4/5] Matching against cached labels ({ex.CTXREL_VERSION}, {ex.JUDGE_ID})...")
    config = JUDGE_REGISTRY[ex.JUDGE_ID]
    _, cached_models = lsr.load_judged(JUDGMENT_FILES)
    if cached_models - {config.model}:
        print(f"[STOP] Cached labels from {sorted(map(str, cached_models))} but the judge is {config.model}.")
        return 1
    judged, _ = lsr.load_judged(JUDGMENT_FILES, judge_model=config.model)
    todo = rsr.unjudged_items({a: recs for a, recs in runs.items()}, set(judged), ctxrel._text_hash)
    plan = ex.cost_estimate(len(todo), judged, config)
    per_alpha = lsr.new_items_per_alpha({a: recs for a, recs in runs.items()}, set(judged), ctxrel._text_hash)
    print(f"      cached labels: {len(judged)}; new unique items: {len(todo)} "
          f"(per alpha {dict((str(a), n) for a, n in per_alpha.items())})")
    print(f"      estimate: {plan}")
    (OUT_DIR / "judge_plan.json").write_text(json.dumps({**plan, "per_alpha": {str(a): n for a, n in per_alpha.items()},
                                                         "judge_model": config.model}, indent=2))
    actual_cost = 0.0
    if todo and not args.run_judge:
        print("[PLAN] Nothing judged (0 API calls). Re-run with --run-judge.")
        return 0
    if todo and plan["est_cost_usd"] > args.max_cost_usd:
        print(f"[STOP] Estimated ${plan['est_cost_usd']} > budget ${args.max_cost_usd}. Nothing judged.")
        return 1
    if todo:
        print(f"[4b/5] Judging {len(todo)} item(s) with {ex.CTXREL_VERSION} / {ex.JUDGE_ID} (temperature 0)...")
        from concurrent.futures import ThreadPoolExecutor

        from judge_clients import get_judge_client
        client, jconfig = get_judge_client(ex.JUDGE_ID)
        bodies = s1._load_question_bodies_text(args.questions_parquet, sorted({i["question_id"] for i in todo}))

        def _one(item):
            judge_item = {**item, "tags": ctxrel.format_tags(item["tags"]),
                          "source_files": [str(files[item["alpha_first_seen"]])]}
            return ctxrel.judge_one(judge_item, ex.JUDGE_ID, client, jconfig, bodies.get(item["question_id"], ""),
                                    prompt_version=ex.CTXREL_VERSION)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rec in pool.map(_one, todo):
                ctxrel.write_record(NEW_JUDGMENTS, rec)
                actual_cost += ((rec.get("prompt_tokens") or 0) * config.price_per_m_input
                                + (rec.get("completion_tokens") or 0) * config.price_per_m_output) / 1e6
        judged, _ = lsr.load_judged(JUDGMENT_FILES, judge_model=config.model)
        still = rsr.unjudged_items({a: recs for a, recs in runs.items()}, set(judged), ctxrel._text_hash)
        if still:
            print(f"[STOP] {len(still)} item(s) could not be judged; re-run --run-judge (resumes).")
            return 1

    print("[5/5] Stage-1 metrics and selection (unchanged code)...")
    sel = {a: {r["question_id"]: {int(c["answer_id"]) for c in r["retrieved_context"]} for r in recs}
           for a, recs in runs.items()}
    metrics = {a: s1.compute_alpha_metrics(str(files[a]), judged, sel[a], a) for a in ALPHAS}
    selection = select_stage1_candidates(metrics)
    stats = {a: {**ex.retrieval_stats(runs[a]), **anchor_stats(runs[a])} for a in ALPHAS}
    prev = previous_result()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"stamp": stamp, "judge_model": config.model, "leakage_mode": leakage.mode,
            "c_retrieval_version": "v4", "anchor_scan_depth": 400,
            "duration_sec": round(time.time() - t0, 1), "new_items_judged": len(todo),
            "estimated_cost_usd": plan["est_cost_usd"], "actual_cost_usd": round(actual_cost, 5),
            "n_excluded_q_all_1": all(r["leakage"]["n_excluded_q"] == 1 for recs in runs.values() for r in recs)}
    lines = ["# Stage 1 recheck with answerable-only anchors (Amendment 17; C v4; dev set, ctxrel-v2 primary)", "",
             f"- Time: {stamp} | leakage `{leakage.mode}` | judge {config.model} | new items judged {len(todo)} | "
             f"actual cost ${meta['actual_cost_usd']} (estimate ${plan['est_cost_usd']})",
             f"- Previous stage 1 (C v3, plain anchors): {prev.get('report', 'not found')}", "",
             "| alpha | % q >=1 RELEVANT (v3 anchors -> v4 anchors) | mean score (v3 -> v4) | % accepted (v3 -> v4) |",
             "|---|---|---|---|"]
    for a in ALPHAS:
        o, n = prev.get("metrics", {}).get(a, {}), metrics[a]
        lines.append(f"| {a} | {o.get('pct_questions_with_relevant')} -> {n['pct_questions_with_relevant']} | "
                     f"{o.get('mean_relevance_score')} -> {n['mean_relevance_score']} | "
                     f"{o.get('pct_accepted_selected')} -> {n['pct_accepted_selected']} |")
    lines += ["", f"- Selection with v3 anchors: top_two {prev.get('selection', {}).get('top_two')}",
              f"- Selection with v4 anchors: top_two {selection['top_two']} (keep_current {selection['keep_current']})",
              f"- Reasoning: {selection['reasoning']}", "", "## Retrieval per alpha", ""]
    for a in ALPHAS:
        s = stats[a]
        lines.append(f"- alpha={a}: final_k total {s['final_k_total']} (<5: {s['final_k_lt_5']}, 0: {s['final_k_0']}); "
                     f"questions with <3 seeds {s['questions_with_fewer_than_3_seeds']}; "
                     f"expansion items in top-5 {s['expansion_items_in_top5_total']}; latency p95 {s['latency_p95']} s")
    md = "\n".join(lines) + "\n"
    (OUT_DIR / f"stage1_anchors_report_{stamp}.md").write_text(md)
    (OUT_DIR / f"stage1_anchors_report_{stamp}.json").write_text(json.dumps({
        "meta": meta, "plan": plan, "previous": prev, "metrics": metrics, "selection": selection,
        "retrieval_stats": {str(a): v for a, v in stats.items()}}, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
