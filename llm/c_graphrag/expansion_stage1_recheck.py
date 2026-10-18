"""
expansion_stage1_recheck.py
=====================================
Stage 1 again on the dev set with the answerable-only semantic expansion
(Amendment 9, docs/DECISION_C_SCORING.md), for alpha in {0, 0.25, 0.5, 0.75,
1}. Same pattern as logscore_stage1_recheck.py; protocol fixed in the
Amendment 9 pre-registration before any result.

WHAT IT DOES
------------------------------------------------------------
1. Dev set (positions 385-434, n = 50, seed 42) exactly as stage1_sweep.py.
2. Retrieval-only C v3 per alpha through c_graphrag.process_sample() with the
   default leakage mode (Amendment 8; no dev effect) twice:
   EXPANSION_REQUIRE_ANSWER=False ("old") and True ("new").
3. Reproduction check: the old top-5 must be IDENTICAL to the stage-1 result
   of record (Amendment 6: results/logscore_check/stage1_logscore_alpha*.jsonl);
   otherwise STOP -- the comparison would not be valid.
4. Judge labels are reused from the cache only when question_id, answer_id,
   text_hash, judge_id/model and prompt_version (ctxrel-v2) are identical.
   Without --run-judge: prints the new items and their EXACT cost estimate
   (cached token means x registry prices), then stops. With --run-judge:
   judges them (temperature 0) only if the estimate <= --max-cost-usd.
5. Stage-1 metrics (stage1_sweep.compute_alpha_metrics) and the unchanged
   selection rule (ctxrel_sweep.select_stage1_candidates) for old and new.
   The old metrics must equal the recorded Amendment 6 metrics.
   The result is accepted whatever it is (Amendment 9).
6. Expansion / retrieval comparison old vs new (empty expansion slots,
   context items from expansion, final_k, changed questions, latency p95).

OUTPUT: llm/c_graphrag/results/expansion_check/

    python3 llm/c_graphrag/expansion_stage1_recheck.py               # plan only, 0 API calls
    python3 llm/c_graphrag/expansion_stage1_recheck.py --run-judge   # judge new items (<= $0.50)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
import logscore_stage1_recheck as lsr  # noqa: E402
import reputation_stage1_recheck as rsr  # noqa: E402

ALPHAS = rsr.ALPHAS
CTXREL_VERSION = rsr.CTXREL_VERSION
JUDGE_ID = rsr.JUDGE_ID
N_DEV = 50
OUT_DIR = rsr.RESULTS / "expansion_check"
NEW_JUDGMENTS = OUT_DIR / "ctxrel_v2_expansion_primary.jsonl"
JUDGMENT_FILES = (*lsr.JUDGMENT_FILES, NEW_JUDGMENTS)
RECORD_DIR = lsr.OUT_DIR  # Amendment 6 stage-1 result of record
RECORD_REPORT = "stage1_logscore_report_*.json"


def record_file(alpha: float) -> Path:
    return RECORD_DIR / f"stage1_logscore_alpha{str(alpha).replace('.', '-')}.jsonl"


def recorded_metrics() -> dict:
    reports = sorted(RECORD_DIR.glob(RECORD_REPORT))
    if len(reports) != 1:
        raise FileNotFoundError(f"need exactly 1 {RECORD_REPORT} in {RECORD_DIR}, found {len(reports)}")
    d = json.loads(reports[0].read_text())
    return {float(a): m for a, m in d["v3"].items()}, d["selection_v3"]


def cost_estimate(n_items: int, judged: dict, config) -> dict:
    """Exact-as-possible estimate: mean prompt/completion tokens of the cached
    labels of this judge x the registry prices."""
    pt = [r.get("prompt_tokens") for r in judged.values() if r.get("prompt_tokens")]
    ct = [r.get("completion_tokens") for r in judged.values() if r.get("completion_tokens")]
    mean_in = float(np.mean(pt)) if pt else config.default_input_tokens_per_item
    mean_out = float(np.mean(ct)) if ct else config.default_output_tokens_per_item
    usd = n_items * (mean_in * config.price_per_m_input + mean_out * config.price_per_m_output) / 1e6
    return {"n_items": n_items, "mean_prompt_tokens": round(mean_in, 1), "mean_completion_tokens": round(mean_out, 1),
            "price_per_m_input": config.price_per_m_input, "price_per_m_output": config.price_per_m_output,
            "est_cost_usd": round(usd, 4)}


def retrieval_stats(records: list[dict]) -> dict:
    exp_q = sum(len(r.get("expansion_qids", [])) for r in records)
    exp_empty = sum(len(r.get("expansion_qids_empty", [])) for r in records)
    exp_items = [sum(1 for c in r["retrieved_context"] if c.get("source_stage") == "semantic_expansion") for r in records]
    ks = [len(r["retrieved_context"]) for r in records]
    lat = [r["retrieval_latency_sec"] for r in records]
    return {"expansion_q": exp_q, "expansion_q_empty": exp_empty,
            "pct_expansion_empty": round(100 * exp_empty / exp_q, 1) if exp_q else None,
            "expansion_items_in_top5_total": sum(exp_items),
            "expansion_items_in_top5_per_q": round(float(np.mean(exp_items)), 3),
            "final_k_total": sum(ks), "final_k_lt_5": sum(k < 5 for k in ks), "final_k_0": sum(k == 0 for k in ks),
            "latency_mean": round(float(np.mean(lat)), 3), "latency_p95": round(float(np.percentile(lat, 95)), 3)}


def build_markdown(old_m, new_m, sel_old, sel_new, stats, changed, plan, meta) -> str:
    lines = [
        "# Stage 1 recheck with answerable-only semantic expansion (Amendment 9; dev set, ctxrel-v2 primary)", "",
        f"- Time: {meta['stamp']}  |  git commit: `{meta['git_commit']}`  |  duration {meta['duration_sec']} s",
        f"- Reproduction (switch off) vs stage-1 result of record (Amendment 6): identical "
        f"({N_DEV} x {len(ALPHAS)}); recomputed metrics identical to the recorded ones",
        f"- New items judged: {plan['n_items']}  |  judge model: {meta['judge_model']}  |  "
        f"estimated cost ${plan['est_cost_usd']}", "",
        "| alpha | % q >=1 RELEVANT (old -> new) | mean score (old -> new) | % accepted (old -> new) | questions changed |",
        "|---|---|---|---|---|",
    ]
    for a in ALPHAS:
        o, n = old_m[a], new_m[a]
        lines.append(f"| {a} | {o['pct_questions_with_relevant']} -> {n['pct_questions_with_relevant']} | "
                     f"{o['mean_relevance_score']} -> {n['mean_relevance_score']} | "
                     f"{o['pct_accepted_selected']} -> {n['pct_accepted_selected']} | {changed[a]} / {N_DEV} |")
    lines += ["", f"- Selection old: top_two {sel_old['top_two']} (keep_current {sel_old['keep_current']})",
              f"- Selection new: top_two {sel_new['top_two']} (keep_current {sel_new['keep_current']})",
              f"- New reasoning: {sel_new['reasoning']}", "", "## Retrieval per alpha (old -> new)", ""]
    for a in ALPHAS:
        o, n = stats["old"][a], stats["new"][a]
        lines.append(f"- alpha={a}: empty expansion {o['pct_expansion_empty']}% -> {n['pct_expansion_empty']}%; "
                     f"expansion items in top-5 {o['expansion_items_in_top5_total']} -> {n['expansion_items_in_top5_total']}; "
                     f"final_k total {o['final_k_total']} -> {n['final_k_total']} (<5: {o['final_k_lt_5']} -> "
                     f"{n['final_k_lt_5']}); latency p95 {o['latency_p95']} -> {n['latency_p95']} s")
    return "\n".join(lines) + "\n"


def main() -> int:
    import tempfile

    import llm_judge_context_relevance_v1 as ctxrel
    import stage1_sweep as s1
    from ctxrel_sweep import select_stage1_candidates
    from embedding_cache import EmbeddingCache, default_cache_path
    from judge_clients import JUDGE_REGISTRY

    from llm.common.leakage import LeakageContext, default_mode

    p = argparse.ArgumentParser(description="Stage-1 recheck with answerable-only expansion (Amendment 9).")
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv("KG_WORKSPACE_DIR"))
    p.add_argument("--embedding-cache-path", default=str(default_cache_path()))
    p.add_argument("--run-judge", action="store_true", help="judge new items (paid) if under --max-cost-usd")
    p.add_argument("--max-cost-usd", type=float, default=0.50)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if ctxrel.REQUEST_TEMPERATURE != 0.0:
        print(f"[STOP] ctxrel judge temperature is {ctxrel.REQUEST_TEMPERATURE}, not 0.")
        return 1
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record_sel = {a: rsr.selected_ids(record_file(a)) for a in ALPHAS}
    rec_metrics, rec_selection = recorded_metrics()

    print(f"[1/6] Dev set (n={N_DEV}, positions 385-434, seed 42)...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, N_DEV, 42, 385)

    print("[2/6] Neo4j (read-only) + FAISS + embedding model (CPU)...")
    driver, database = cg.connect_neo4j(print)
    faiss_index, faiss_ids, faiss_emb, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cache = EmbeddingCache(args.embedding_cache_path)
    leakage = LeakageContext.from_mode(default_mode())
    cg.log = lambda *a, **k: None

    runs = {"old": {}, "new": {}}
    try:
        print(f"[3/6] Retrieval-only per alpha, expansion switch off (old) and on (new), leakage {leakage.mode}...")
        for label, flag in (("old", False), ("new", True)):
            for a in ALPHAS:
                with tempfile.TemporaryDirectory() as tmp:
                    recs, _ = cg.process_sample(
                        sample_df, None, None, embed_model, driver, database, faiss_index, faiss_ids, faiss_emb,
                        id_to_row, all_answer_ids_map, 5, 3, 3, 400, "none", Path(tmp) / "o.jsonl",
                        retrieval_only=True, c_retrieval_version="v3", alpha=a, embedding_cache=cache,
                        leakage=leakage, expansion_require_answer=flag)
                runs[label][a] = recs
    finally:
        cache.close()
        driver.close()

    mismatches = [(r["question_id"], a) for a in ALPHAS for r in runs["old"][a]
                  if [int(c["answer_id"]) for c in r["retrieved_context"]] != record_sel[a].get(r["question_id"])]
    if mismatches:
        print(f"[STOP] Switch-off top-5 differs from the stage-1 result of record at {len(mismatches)} (qid, alpha), "
              f"e.g. {mismatches[:5]}. The comparison is not valid.")
        return 1
    print(f"      Reproduction OK: switch-off top-5 identical to the result of record ({N_DEV} x {len(ALPHAS)}).")

    files = {}
    for label in runs:
        for a, recs in runs[label].items():
            path = OUT_DIR / f"stage1_expansion_{label}_alpha{str(a).replace('.', '-')}.jsonl"
            with open(path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r, default=str) + "\n")
            files[(label, a)] = path

    print(f"[4/6] Matching against cached labels ({CTXREL_VERSION}, {JUDGE_ID})...")
    config = JUDGE_REGISTRY[JUDGE_ID]
    _, cached_models = lsr.load_judged(JUDGMENT_FILES)
    if cached_models - {config.model}:
        print(f"[STOP] Cached labels from {sorted(map(str, cached_models))} but the '{JUDGE_ID}' judge is "
              f"{config.model}. Labels of different models must not be mixed.")
        return 1
    judged, _ = lsr.load_judged(JUDGMENT_FILES, judge_model=config.model)
    todo = rsr.unjudged_items(runs["new"], set(judged), ctxrel._text_hash)
    plan = cost_estimate(len(todo), judged, config)
    per_alpha = lsr.new_items_per_alpha(runs["new"], set(judged), ctxrel._text_hash)
    print(f"      cached labels: {len(judged)}; new unique items: {len(todo)} "
          f"(per alpha {dict((str(a), n) for a, n in per_alpha.items())})")
    print(f"      estimate: {plan}")
    (OUT_DIR / "judge_plan.json").write_text(json.dumps({**plan, "per_alpha": {str(a): n for a, n in per_alpha.items()},
                                                         "judge_model": config.model}, indent=2))

    if todo and not args.run_judge:
        print("[PLAN] Nothing judged (0 API calls). Re-run with --run-judge.")
        return 0
    if todo and plan["est_cost_usd"] > args.max_cost_usd:
        print(f"[STOP] Estimated ${plan['est_cost_usd']} > budget ${args.max_cost_usd}.")
        return 1
    if todo:
        print(f"[5/6] Judging {len(todo)} item(s) with {CTXREL_VERSION} / {JUDGE_ID} (temperature 0)...")
        from judge_clients import get_judge_client
        client, jconfig = get_judge_client(JUDGE_ID)
        bodies = s1._load_question_bodies_text(args.questions_parquet, sorted({i["question_id"] for i in todo}))

        def _one(item):
            judge_item = {**item, "tags": ctxrel.format_tags(item["tags"]),
                          "source_files": [str(files[("new", item["alpha_first_seen"])])]}
            return ctxrel.judge_one(judge_item, JUDGE_ID, client, jconfig,
                                    bodies.get(item["question_id"], ""), prompt_version=CTXREL_VERSION)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rec in pool.map(_one, todo):
                ctxrel.write_record(NEW_JUDGMENTS, rec)
        judged, _ = lsr.load_judged(JUDGMENT_FILES, judge_model=config.model)
        still = rsr.unjudged_items(runs["new"], set(judged), ctxrel._text_hash)
        if still:
            print(f"[STOP] {len(still)} item(s) could not be judged; re-run --run-judge (resumes).")
            return 1

    print("[6/6] Stage-1 metrics and selection (unchanged code)...")
    sel = {label: {a: {r["question_id"]: {int(c["answer_id"]) for c in r["retrieved_context"]} for r in recs}
                   for a, recs in runs[label].items()} for label in runs}
    old_m = {a: s1.compute_alpha_metrics(str(files[("old", a)]), judged, sel["old"], a) for a in ALPHAS}
    new_m = {a: s1.compute_alpha_metrics(str(files[("new", a)]), judged, sel["new"], a) for a in ALPHAS}
    off = [a for a in ALPHAS if (old_m[a]["pct_questions_with_relevant"], old_m[a]["mean_relevance_score"]) !=
           (rec_metrics[a]["pct_questions_with_relevant"], rec_metrics[a]["mean_relevance_score"])]
    if off:
        print(f"[STOP] Recomputed switch-off metrics differ from the recorded Amendment 6 metrics at alpha {off}.")
        return 1
    selection_old = select_stage1_candidates(old_m)
    selection_new = select_stage1_candidates(new_m)
    if selection_old["top_two"] != rec_selection["top_two"]:
        print(f"[STOP] Switch-off selection {selection_old['top_two']} != recorded {rec_selection['top_two']}.")
        return 1
    changed = {a: sum(1 for o, n in zip(runs["old"][a], runs["new"][a])
                      if [c["answer_id"] for c in o["retrieved_context"]] != [c["answer_id"] for c in n["retrieved_context"]])
               for a in ALPHAS}
    stats = {label: {a: retrieval_stats(runs[label][a]) for a in ALPHAS} for label in runs}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import reputation_ablation_check as rac
    meta = {"stamp": stamp, "git_commit": rac._git_commit(Path(__file__).resolve().parents[2]),
            "judge_model": config.model,
            "duration_sec": round(time.time() - t0, 1), "leakage_mode": leakage.mode,
            "expansion_overfetch": cg.EXPANSION_OVERFETCH}
    md = build_markdown(old_m, new_m, selection_old, selection_new, stats, changed, plan, meta)
    (OUT_DIR / f"stage1_expansion_report_{stamp}.md").write_text(md)
    (OUT_DIR / f"stage1_expansion_report_{stamp}.json").write_text(json.dumps({
        "meta": meta, "plan": plan, "old": old_m, "new": new_m, "selection_old": selection_old,
        "selection_new": selection_new, "questions_changed": {str(a): n for a, n in changed.items()},
        "retrieval_stats": {label: {str(a): v for a, v in d.items()} for label, d in stats.items()},
    }, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
