"""
logscore_stage1_recheck.py
=====================================
Stage 1 again on the dev set with trustScore "v3-logscore" (Amendment 6,
docs/DECISION_C_SCORING.md), for alpha in {0, 0.25, 0.5, 0.75, 1}. Same
pattern as reputation_stage1_recheck.py.

WHAT IT DOES
------------------------------------------------------------
1. Dev set, retrieval and fuse_and_rank() v3 exactly as stage1_sweep.py.
   Candidates are retrieved once per question and ranked per alpha under
   trust v2 ("v2-reputation") and v3 ("v3-logscore"); both come from
   logscore_top5_check.resolve_trust, so this works before and after
   14_apply_logscore_trust.py --apply (after: pass --backup).
2. Reproduction check: the v2 top-5 must be IDENTICAL to the stage-1 result
   of record (Amendment 5: results/reputation_check/stage1_repfix_alpha*.jsonl).
   Otherwise the script stops; the comparison would not be valid.
3. Judge labels are reused from the cache. Cache key: question_id,
   answer_id, text_hash (SHA-256 of the context text shown to the judge),
   judge_id = primary, judge_model, prompt_version = ctxrel-v2. Only v3
   items without a usable cached label are new.
4. WITHOUT --run-judge (default): prints the number of new items per alpha
   and the number of judge calls needed, then stops. 0 API calls.
   WITH --run-judge: judges the new items (ctxrel-v2, primary; paid).
5. Stage-1 metrics (stage1_sweep.compute_alpha_metrics) and the unchanged
   stage-1 selection rule (ctxrel_sweep.select_stage1_candidates) are
   computed for v2 and v3 with the same code as stage 1.

OUTPUT (llm/c_graphrag/results/logscore_check/)
------------------------------------------------------------
  stage1_logscore_alpha<a>.jsonl              v3 retrieval per alpha
  ctxrel_v2_logscore_primary.jsonl            labels of new items (safe to resume)
  stage1_logscore_report_<ts>.md / .json      comparison v2 -> v3 + selection

USAGE (from the repo root)
------------------------------------------------------------
    python3 llm/c_graphrag/logscore_stage1_recheck.py
    python3 llm/c_graphrag/logscore_stage1_recheck.py --run-judge
    (after --apply, add: --backup 01_data_cleaning/_kg_workspace/<backup>.parquet)
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
import logscore_top5_check as ltc  # noqa: E402
import reputation_ablation_check as rac  # noqa: E402
import reputation_stage1_recheck as rsr  # noqa: E402

ALPHAS = rsr.ALPHAS
CTXREL_VERSION = rsr.CTXREL_VERSION
JUDGE_ID = rsr.JUDGE_ID
N_DEV = 50
OUT_DIR = rsr.RESULTS / "logscore_check"
NEW_JUDGMENTS = OUT_DIR / "ctxrel_v2_logscore_primary.jsonl"
JUDGMENT_FILES = (rsr.STAGE1_JUDGMENTS, rsr.NEW_JUDGMENTS, NEW_JUDGMENTS)
# Stage-1 result of record (Amendment 5, repfix): retrieval files and report.
V2_DIR = rsr.OUT_DIR
V2_REPORT_GLOB = "stage1_repfix_report_*.json"


# ---------------------------------------------------------------------
# Pure logic -- tested in tests/test_logscore_trust.py
# ---------------------------------------------------------------------

def v2_file_for_alpha(alpha: float, v2_dir: Path = V2_DIR) -> Path:
    return Path(v2_dir) / f"stage1_repfix_alpha{str(alpha).replace('.', '-')}.jsonl"


def load_judged(paths, judge_id: str = JUDGE_ID, prompt_version: str = CTXREL_VERSION,
                judge_model: str | None = None) -> tuple[dict, set]:
    """({(question_id, answer_id, text_hash): record}, judge models seen).

    Only records of this judge_id and prompt_version with a usable label count as cached;
    with `judge_model`, records of any other model are ignored as well. A later usable
    record for the same key replaces an earlier one."""
    by_key, models = {}, set()
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("judge_id") != judge_id or r.get("prompt_version") != prompt_version:
                    continue
                if not r.get("label"):
                    continue
                models.add(r.get("judge_model"))
                if judge_model is not None and r.get("judge_model") != judge_model:
                    continue
                by_key[(r.get("question_id"), r.get("answer_id"), r.get("text_hash"))] = r
    return by_key, models


def new_items_per_alpha(records_by_alpha: dict[float, list[dict]], judged_keys: set, text_hash) -> dict[float, int]:
    """Number of selected context items without a cached label, per alpha (an item selected
    at several alphas is counted at each of them; the judge is called once per unique item)."""
    out = {}
    for a, recs in records_by_alpha.items():
        out[a] = sum(1 for rec in recs for c in rec["retrieved_context"]
                     if (rec["question_id"], int(c["answer_id"]), text_hash(c["chunk_text"])) not in judged_keys)
    return out


def compare_metrics(v2: dict, v3: dict, alphas=ALPHAS) -> list[dict]:
    """Descriptive v2 -> v3 deltas per alpha (no decision is attached to them)."""
    return [{"alpha": a,
             "delta_pct_questions_with_relevant": round(
                 v3[a]["pct_questions_with_relevant"] - v2[a]["pct_questions_with_relevant"], 2),
             "delta_mean_relevance_score": round(v3[a]["mean_relevance_score"] - v2[a]["mean_relevance_score"], 4)}
            for a in alphas]


def recorded_v2_metrics(v2_dir: Path = V2_DIR) -> dict[float, tuple[float, float]]:
    """{alpha: (pct_questions_with_relevant, mean_relevance_score)} of the stage-1 result of record."""
    reports = sorted(Path(v2_dir).glob(V2_REPORT_GLOB))
    if len(reports) != 1:
        raise FileNotFoundError(f"need exactly 1 {V2_REPORT_GLOB} in {v2_dir}, found {len(reports)}")
    repfix = json.loads(reports[0].read_text())["repfix"]
    return {float(a): (m["pct_questions_with_relevant"], m["mean_relevance_score"]) for a, m in repfix.items()}


def build_markdown(v2_m: dict, v3_m: dict, sel_v2: dict, sel_v3: dict, deltas: list[dict], meta: dict) -> str:
    lines = [
        "# Stage 1 recheck with trustScore v3-logscore (dev set, ctxrel-v2 primary)", "",
        f"- Time: {meta['stamp']}  |  git commit: `{meta['git_commit']}`",
        f"- Reproduction v2 vs stage-1 result of record (Amendment 5): identical (50 x {len(ALPHAS)})",
        f"- New items judged: {meta['n_new']}  |  judge model: {meta['judge_model']}  |  "
        f"P99 Answer score: {meta['p99_answer']}  |  duration {meta['duration_sec']} s", "",
        "| alpha | % q >=1 RELEVANT (v2 -> v3) | mean score (v2 -> v3) | % accepted (v2 -> v3) |",
        "|---|---|---|---|",
    ]
    for a in ALPHAS:
        b, r = v2_m[a], v3_m[a]
        lines.append(f"| {a} | {b['pct_questions_with_relevant']} -> {r['pct_questions_with_relevant']} | "
                     f"{b['mean_relevance_score']} -> {r['mean_relevance_score']} | "
                     f"{b['pct_accepted_selected']} -> {r['pct_accepted_selected']} |")
    lines += [
        "", f"- Selection helper v2: top_two {sel_v2['top_two']} (keep_current {sel_v2['keep_current']})",
        f"- Selection helper v3: top_two {sel_v3['top_two']} (keep_current {sel_v3['keep_current']})",
        f"- v3 reasoning: {sel_v3['reasoning']}", "",
        "## v2 -> v3 per alpha (descriptive)", "",
    ]
    for d in deltas:
        lines.append(f"- alpha={d['alpha']}: delta % q = {d['delta_pct_questions_with_relevant']}, "
                     f"delta score = {d['delta_mean_relevance_score']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> int:
    import llm_judge_context_relevance_v1 as ctxrel
    import stage1_sweep as s1
    from ctxrel_sweep import select_stage1_candidates
    from embedding_cache import EmbeddingCache, default_cache_path
    from judge_clients import JUDGE_REGISTRY

    p = argparse.ArgumentParser(description="Stage-1 recheck with trustScore v3-logscore (Amendment 6).")
    p.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    p.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    p.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    p.add_argument("--embedding-cache-path", default=str(default_cache_path()))
    p.add_argument("--backup", default=None,
                   help="Backup parquet from 14_apply_logscore_trust.py --apply (needed once the graph is v3).")
    p.add_argument("--run-judge", action="store_true",
                   help="Judge the new items with ctxrel-v2 primary (paid API calls).")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] QUESTIONS_PARQUET/ANSWERS_PARQUET are required (flag or .env).")
        return 1

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_files = {a: v2_file_for_alpha(a) for a in ALPHAS}
    v2_selected = {a: rsr.selected_ids(f) for a, f in v2_files.items()}

    print(f"[1/6] Dev set (n={N_DEV}, positions 385-434, seed 42)...")
    sample_df, all_answer_ids_map, _, _ = s1._build_dev_sample(
        args.questions_parquet, args.answers_parquet, N_DEV, 42, 385)

    print("[2/6] Neo4j (read-only) + P99 + FAISS + embedding model (CPU)...")
    driver, database = cg.connect_neo4j(print)
    p99_answer = ltc.answer_score_p99(driver, database)
    backup_trust = ltc.load_backup_trust(args.backup) if args.backup else None
    print(f"      P99_A = {p99_answer}; backup: {args.backup or '-'}")
    faiss_index, faiss_ids, faiss_emb, id_to_row = cg.load_faiss_cache(Path(args.kg_workspace_dir), print)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.embed_model, device="cpu")
    cache = EmbeddingCache(args.embedding_cache_path)

    v3_records = {a: [] for a in ALPHAS}
    mismatches = []
    try:
        print(f"[3/6] One retrieval per question; ranking v2 vs v3 at {len(ALPHAS)} alpha...")
        for _, row in sample_df.iterrows():
            qid = int(row["Id"])
            q_emb, graph_c, exp_c = rac.retrieve_candidates(
                row, faiss_index, faiss_ids, faiss_emb, id_to_row, all_answer_ids_map,
                driver, database, embed_model, 3, 3)
            aids = sorted({int(c["answer_id"]) for c in graph_c + exp_c})
            trust_v2, trust_v3 = ltc.resolve_trust(ltc.fetch_answer_inputs(driver, database, aids),
                                                   p99_answer, backup_trust)
            for a in ALPHAS:
                top_v2 = ltc.rank_with_trust(a, graph_c, exp_c, trust_v2, embed_model, q_emb, cache)
                if [int(c["answer_id"]) for c in top_v2] != v2_selected[a].get(qid):
                    mismatches.append((qid, a))
                top_v3 = ltc.rank_with_trust(a, graph_c, exp_c, trust_v3, embed_model, q_emb, cache)
                ltc.compare_alpha(qid, a, top_v2, top_v3)
                v3_records[a].append({"question_id": qid, "title": row["Title"], "tags": row["Tags"],
                                      "retrieved_context": top_v3})
    except (ValueError, RuntimeError) as e:
        print(f"[STOP] {e}")
        return 1
    finally:
        cache.close()
        driver.close()

    if mismatches:
        print(f"[STOP] The v2 top-5 differs from the stage-1 result of record at {len(mismatches)} (qid, alpha), "
              f"e.g. {mismatches[:5]}. Retrieval no longer reproduces it; the comparison is not valid.")
        return 1
    print(f"      Reproduction OK: v2 top-5 identical to the result of record for {N_DEV} questions x "
          f"{len(ALPHAS)} alpha.")

    v3_files = {}
    for a, recs in v3_records.items():
        path = OUT_DIR / f"stage1_logscore_alpha{str(a).replace('.', '-')}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        v3_files[a] = path

    print(f"[4/6] Matching against cached labels ({CTXREL_VERSION}, {JUDGE_ID})...")
    judge_model = JUDGE_REGISTRY[JUDGE_ID].model
    _, cached_models = load_judged(JUDGMENT_FILES)
    if cached_models - {judge_model}:
        print(f"[STOP] Cached labels come from {sorted(map(str, cached_models))}, but the registry's "
              f"'{JUDGE_ID}' judge is {judge_model}. Labels of different models must not be mixed.")
        return 1
    judged, _ = load_judged(JUDGMENT_FILES, judge_model=judge_model)
    per_alpha = new_items_per_alpha(v3_records, set(judged), ctxrel._text_hash)
    todo = rsr.unjudged_items(v3_records, set(judged), ctxrel._text_hash)
    print(f"      Judge model: {judge_model}; cached labels: {len(judged)}")
    for a in ALPHAS:
        print(f"      alpha={a}: {per_alpha[a]} new item(s) of {sum(len(r['retrieved_context']) for r in v3_records[a])}")
    print(f"      Unique new items = judge calls needed: {len(todo)}")

    if todo and not args.run_judge:
        print(f"[PLAN] Nothing was judged. Re-run with --run-judge to judge {len(todo)} item(s) "
              f"({JUDGE_ID}, {judge_model}). Amendment 6 must be committed first.")
        return 0

    if todo:
        print(f"[5/6] Judging {len(todo)} item(s) with {CTXREL_VERSION} / {JUDGE_ID}...")
        from judge_clients import get_judge_client
        client, config = get_judge_client(JUDGE_ID)
        bodies = s1._load_question_bodies_text(args.questions_parquet, sorted({i["question_id"] for i in todo}))

        def _one(item):
            judge_item = {**item, "tags": ctxrel.format_tags(item["tags"]),
                          "source_files": [str(v3_files[item["alpha_first_seen"]])]}
            return ctxrel.judge_one(judge_item, JUDGE_ID, client, config,
                                    bodies.get(item["question_id"], ""), prompt_version=CTXREL_VERSION)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for rec in pool.map(_one, todo):
                ctxrel.write_record(NEW_JUDGMENTS, rec)
        judged, _ = load_judged(JUDGMENT_FILES, judge_model=judge_model)
        still = rsr.unjudged_items(v3_records, set(judged), ctxrel._text_hash)
        if still:
            print(f"[STOP] {len(still)} item(s) could not be judged; re-run --run-judge (safe, resumes).")
            return 1

    print("[6/6] Stage-1 metrics and selection...")
    sel_v2 = {a: {q: set(v) for q, v in v2_selected[a].items()} for a in ALPHAS}
    sel_v3 = {a: {r["question_id"]: {int(c["answer_id"]) for c in r["retrieved_context"]}
                  for r in recs} for a, recs in v3_records.items()}
    v2_m = {a: s1.compute_alpha_metrics(str(v2_files[a]), judged, sel_v2, a) for a in ALPHAS}
    v3_m = {a: s1.compute_alpha_metrics(str(v3_files[a]), judged, sel_v3, a) for a in ALPHAS}
    recorded = recorded_v2_metrics()
    off = [a for a in ALPHAS if recorded.get(a) != (v2_m[a]["pct_questions_with_relevant"],
                                                    v2_m[a]["mean_relevance_score"])]
    if off:
        print(f"[STOP] Recomputed v2 metrics differ from the stage-1 result of record at alpha {off}.")
        return 1
    print("      v2 metrics identical to the stage-1 result of record (Amendment 5).")
    selection_v2 = select_stage1_candidates(v2_m)
    selection_v3 = select_stage1_candidates(v3_m)
    deltas = compare_metrics(v2_m, v3_m)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"stamp": stamp, "git_commit": rac._git_commit(Path(__file__).resolve().parents[2]),
            "n_new": len(todo), "judge_model": judge_model, "p99_answer": p99_answer,
            "duration_sec": round(time.time() - t0, 1)}
    md = build_markdown(v2_m, v3_m, selection_v2, selection_v3, deltas, meta)
    (OUT_DIR / f"stage1_logscore_report_{stamp}.md").write_text(md)
    (OUT_DIR / f"stage1_logscore_report_{stamp}.json").write_text(json.dumps({
        "meta": meta, "v2": v2_m, "v3": v3_m, "selection_v2": selection_v2, "selection_v3": selection_v3,
        "deltas": deltas, "new_items_per_alpha": {str(a): n for a, n in per_alpha.items()},
        "n_new_judgments": len(todo), "backup": args.backup,
    }, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
