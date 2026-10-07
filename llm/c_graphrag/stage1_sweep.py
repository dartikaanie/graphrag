"""
stage1_sweep.py
=====================================
End-to-end stage-1 CLI runner for the C retrieval v3 α sweep (Step 4 of
docs/agent_prompt_c_retrieval_v3_devset.md, decision procedure in
docs/DECISION_C_SCORING.md). For the dev split and α in
DEFAULT_ALPHA_VALUES:

  (a) runs the ctxrel-v1 positive/negative controls and STOPS if they fail;
  (b) runs retrieval-only C v3 for each α (no LLM calls -- free, config-
      hashed + manifested like every other run);
  (c) runs ctxrel-v1 (primary judge) on every resulting context, deduped
      across α (via llm_judge_context_relevance_v1.load_items(), which
      already dedups on (question_id, answer_id, text_hash));
  (d) writes the stage-1 markdown report (per-α relevance metrics, hop/
      edge-type breakdown, Jaccard vs α=0 and α=1);
  (e) prints the selection helper's output with reasoning.

Retrieval (b) has NO API cost, so it runs first to get REAL dedup counts
-- the plan shown before confirmation is then exact, not an estimate of
an estimate. The only real-money step is ctxrel-v1 judging (controls +
main sweep), which is gated behind an explicit "show plan -> confirm ->
run" flow, same shape as the dashboard's judge-v1 plan/launch.

CARA PAKAI (run from the repo root; QUESTIONS_PARQUET/ANSWERS_PARQUET can
come from .env instead of flags, same as c_graphrag.py's own CLI)
----------------------------------------------------------------------------
    # (a) controls only -- cheap sanity check before committing to the sweep
    python3 llm/c_graphrag/stage1_sweep.py --controls-only

    # (b) the full stage-1 sweep: builds the dev sample, runs retrieval-only
    # for every alpha (free), shows the plan, asks for y/N confirmation,
    # then (if confirmed) runs the controls and the ctxrel-v1 judging and
    # writes the report. Add --yes to skip the confirmation prompt, or
    # --plan-only to stop right after the plan (no API call at all).
    python3 llm/c_graphrag/stage1_sweep.py
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

DEFAULT_ALPHA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
RESULTS_DIR = Path(__file__).resolve().parent / "results"  # absolute -- stable regardless of invocation cwd


# ---------------------------------------------------------------------
# (b) Retrieval-only runs per alpha -- no API cost.
# ---------------------------------------------------------------------

def run_retrieval_only_for_alpha(
    sample_df, alpha: float, driver, database, faiss_index, faiss_ids, faiss_embeddings,
    id_to_row, all_answer_ids_map, embed_model, embedding_cache, questions_parquet: str,
    output_dir: str, seed: int, n_sample: int, provider: str = "openai", model: str = "gpt-4o-mini",
    top_k: int = 5, n_anchor: int = 3, n_semantic_expansion: int = 3, token_chunk_limit: int = 400,
    oversample_pool: int = 1536,
) -> dict:
    """Runs C retrieval-only v3 for one alpha on the dev split, writes a
    config-hashed output file + manifest + run_history entry (so
    _run_metadata.resolve_run_metadata() -- which ctxrel-v1's load_items()
    requires -- can resolve it), and returns {"output_path", "config_hash",
    "n_questions"}."""
    import c_graphrag as cg
    from llm.manifest import compute_config_hash, write_manifest

    fusion_mode = "uniform" if alpha == 0 else "trust_weighted"
    output_path = cg.build_output_path(
        output_dir, provider, model, n_sample, seed, fusion_mode,
        oversample_pool=oversample_pool, top_k=top_k, n_anchor=n_anchor,
        n_semantic_expansion=n_semantic_expansion, c_retrieval_version="v3", alpha=alpha,
        sample_split="dev",
    )
    config = cg.build_config(
        provider, model, n_sample, seed, oversample_pool, top_k, n_anchor, n_semantic_expansion,
        fusion_mode, cg.DEFAULT_FUSION_W_PATH_TRUST, cg.DEFAULT_FUSION_W_ANSWER_INTRINSIC_TRUST,
        cg.DEFAULT_SEMANTIC_EXPANSION_TRUST_CAP, True, True,
        c_retrieval_version="v3", alpha=alpha, sample_split="dev",
    )
    config_hash = compute_config_hash(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()  # sweep runs are meant to be fresh/reproducible, not resumed mid-sweep

    run_started_at = datetime.now(timezone.utc)
    results, stats = cg.process_sample(
        sample_df, None, None, embed_model, driver, database,
        faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
        top_k, n_anchor, n_semantic_expansion, token_chunk_limit, model, output_path,
        c_retrieval_version="v3", alpha=alpha, embedding_cache=embedding_cache, retrieval_only=True,
        config=config, config_hash=config_hash,
    )
    finished_at = datetime.now(timezone.utc)

    history_path = cg.append_run_history({
        "prompt_version": "retrieval-only", "c_retrieval_version": "v3", "alpha": alpha,
        "sample_split": "dev", "run_started_at": run_started_at.isoformat(), "condition": "C",
        "status": "success", "provider": provider, "model": model, "n_sample_target": n_sample,
        "n_processed": len(results), "seed": seed, "oversample_pool": oversample_pool,
        "fusion_mode": fusion_mode, "output_path": str(output_path), "config_hash": config_hash,
        "duration_sec": round((finished_at - run_started_at).total_seconds(), 1),
        "retrieval_only": True,
    })
    write_manifest(
        output_path, run_label=None, status="completed",
        config=config, config_hash=config_hash,
        started_at_utc=run_started_at.isoformat(), finished_at_utc=finished_at.isoformat(),
        item_counts={"attempted": len(sample_df), "succeeded": len(results), "failed": 0},
        prompt_version="retrieval-only",
    )
    return {"output_path": str(output_path), "config_hash": config_hash, "n_questions": len(results),
            "history_path": str(history_path)}


# ---------------------------------------------------------------------
# (a) Controls -- own accepted answer must be RELEVANT, an unrelated
# item must be IRRELEVANT. Smoke-test data, written to its OWN file,
# never mixed with real sweep results.
# ---------------------------------------------------------------------

def build_control_items(sample_df) -> list[dict]:
    """Two synthetic (question, context item) pairs built from REAL dev
    questions already in `sample_df` (needs AcceptedAnswerBody/Title/
    Tags columns, same as process_sample()'s input):
      - positive: question A's OWN accepted answer as its context item
        -> expected RELEVANT.
      - negative: question A paired with question B's accepted answer,
        where B shares NO tags with A -> expected IRRELEVANT.
    Falls back gracefully (fewer controls) if the dev sample is too
    small/homogeneous to find a true negative -- callers must check
    len(result) before treating missing controls as a failure.
    """
    import re

    def tagset(raw):
        return set(re.findall(r"<([^>]+)>", str(raw or "")))

    items = []
    if len(sample_df) == 0:
        return items

    row_a = sample_df.iloc[0]
    tags_a = tagset(row_a["Tags"])
    items.append({
        "question_id": int(row_a["Id"]), "answer_id": int(row_a["AcceptedAnswerId"]),
        "chunk_text": f"Q: {row_a['Title']}\nA: {row_a['AcceptedAnswerBody']}",
        "title": row_a["Title"], "tags": row_a["Tags"], "expected_label": "RELEVANT",
        "control_type": "positive",
    })

    row_b = None
    for _, candidate in sample_df.iloc[1:].iterrows():
        if not (tagset(candidate["Tags"]) & tags_a):
            row_b = candidate
            break
    if row_b is not None:
        items.append({
            "question_id": int(row_a["Id"]), "answer_id": int(row_b["AcceptedAnswerId"]),
            "chunk_text": f"Q: {row_b['Title']}\nA: {row_b['AcceptedAnswerBody']}",
            "title": row_a["Title"], "tags": row_a["Tags"], "expected_label": "IRRELEVANT",
            "control_type": "negative",
        })
    return items


def run_controls(sample_df, judge_id: str, questions_parquet: str, workers: int = 2) -> dict:
    """Runs the controls through the SAME ctxrel-v1 judge machinery
    (judge_one/run_batch), writing to results/ctxrel_v1_controls_<judge_id>.jsonl
    -- a clearly-named, separate file, never mixed with real sweep output.
    Returns {"passed": bool, "results": [{"control_type", "expected",
    "actual", "match"}, ...]}."""
    import llm_judge_context_relevance_v1 as ctxrel
    from judge_clients import get_judge_client

    control_items = build_control_items(sample_df)
    if len(control_items) < 2:
        return {"passed": False, "results": [],
                "error": "could not build both positive and negative control items from this dev sample"}

    client, config = get_judge_client(judge_id)
    results = []
    for item in control_items:
        messages = ctxrel.build_messages(item["title"], "", ctxrel.format_tags(item["tags"]), item["chunk_text"])
        payload, attempts = ctxrel.call_judge_with_retry(client, config, messages)
        if payload is None:
            results.append({"control_type": item["control_type"], "expected": item["expected_label"],
                             "actual": None, "match": False, "error": attempts[-1]["error"] if attempts else None})
            continue
        parsed = ctxrel.parse_judgment(payload["raw"])
        actual = parsed["label"] if not parsed["parse_error"] else None
        results.append({
            "control_type": item["control_type"], "expected": item["expected_label"],
            "actual": actual, "match": actual == item["expected_label"],
        })

    out_path = RESULTS_DIR / f"ctxrel_v1_controls_{judge_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "judge_id": judge_id,
            "results": results,
        }) + "\n")

    passed = all(r["match"] for r in results)
    return {"passed": passed, "results": results, "output_path": str(out_path)}


# ---------------------------------------------------------------------
# Token/latency estimate for the plan -- same fallback-chain shape as
# backend/app/services/engine_service.py's _ctxrel_token_latency_stats,
# duplicated here (not imported) since backend/ depends on llm/, never
# the reverse.
# ---------------------------------------------------------------------

def _token_latency_stats(judge_model: str, results_dir: Path | None = None) -> tuple:
    results_dir = results_dir if results_dir is not None else RESULTS_DIR  # read at call time (monkeypatch-friendly)
    input_vals, output_vals, latency_vals = [], [], []
    if results_dir.exists():
        for path in sorted(results_dir.glob("ctxrel_v1_*.jsonl")):
            if "controls" in path.name or "smoke_test" in path.name:
                continue
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("judge_model") != judge_model or r.get("call_failed"):
                            continue
                        if isinstance(r.get("prompt_tokens"), (int, float)):
                            input_vals.append(r["prompt_tokens"])
                        if isinstance(r.get("completion_tokens"), (int, float)):
                            output_vals.append(r["completion_tokens"])
                        if isinstance(r.get("latency_s"), (int, float)):
                            latency_vals.append(r["latency_s"])
            except OSError:
                continue
    if input_vals and output_vals:
        return (sum(input_vals) / len(input_vals), sum(output_vals) / len(output_vals),
                sum(latency_vals) / len(latency_vals) if latency_vals else None, "from history")
    return None, None, None, None


def build_plan(unique_items: int, n_controls: int, judge_id: str, workers: int = 4) -> dict:
    from judge_clients import JUDGE_REGISTRY

    config = JUDGE_REGISTRY.get(judge_id)
    total_items = unique_items + n_controls
    if config is None:
        return {"items": total_items, "unique_items": unique_items, "n_controls": n_controls,
                "judge_id": judge_id, "cost_unknown": True, "est_cost_usd": None, "est_time_sec": None}

    avg_in, avg_out, avg_lat, source = _token_latency_stats(config.model)
    if avg_in is None:
        avg_in, avg_out = config.default_input_tokens_per_item, config.default_output_tokens_per_item
        source = "default estimate"
    if avg_lat is None:
        avg_lat = config.default_latency_sec_per_item

    est_tokens_in = round(avg_in * total_items)
    est_tokens_out = round(avg_out * total_items)
    est_cost = round(est_tokens_in / 1_000_000 * config.price_per_m_input
                      + est_tokens_out / 1_000_000 * config.price_per_m_output, 4)
    w = max(1, workers)
    est_time_sec = round(total_items / w * avg_lat)

    return {
        "items": total_items, "unique_items": unique_items, "n_controls": n_controls,
        "judge_id": judge_id, "judge_model": config.model, "cost_unknown": False,
        "est_tokens_in": est_tokens_in, "est_tokens_out": est_tokens_out,
        "est_cost_usd": est_cost, "est_time_sec": est_time_sec, "token_source": source,
    }


def format_plan(plan: dict) -> str:
    lines = [
        "=== STAGE 1 SWEEP -- PLAN (no API calls made yet) ===",
        f"Controls: {plan['n_controls']} item(s)",
        f"Unique (question, context item) pairs to judge (deduped across all alpha values): {plan['unique_items']}",
        f"Total items: {plan['items']}  (judge_id={plan['judge_id']})",
    ]
    if plan["cost_unknown"]:
        lines.append("Estimated cost: unknown (judge_id not in registry)")
    else:
        lines.append(f"Estimated tokens (in/out): {plan['est_tokens_in']} / {plan['est_tokens_out']} "
                      f"(source: {plan['token_source']})")
        lines.append(f"Estimated cost: ${plan['est_cost_usd']:.4f}")
        lines.append(f"Estimated time: ~{plan['est_time_sec']}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# (d) Stage-1 markdown report
# ---------------------------------------------------------------------

def compute_alpha_metrics(retrieval_output_path: str, ctxrel_by_key: dict, selected_answer_ids_by_alpha: dict,
                           this_alpha: float) -> dict:
    """Per-alpha stage-1 metrics for ONE retrieval-only output file,
    joined against the ctxrel-v1 judged records (keyed by (question_id,
    answer_id, text_hash), content-based, never by run_id -- same join
    pattern as llm_judge_context_relevance_v1.compute_run_context_
    relevance()). `selected_answer_ids_by_alpha` is {alpha: {qid:
    set(answer_ids)}}, used for the Jaccard overlap vs alpha=0/alpha=1."""
    import llm_judge_context_relevance_v1 as ctxrel_mod

    LABEL_SCORE = {"RELEVANT": 1.0, "PARTIAL": 0.5, "IRRELEVANT": 0.0}
    n_judged = 0
    label_counts: dict[str, int] = {}
    scores = []
    sims_selected = []
    trusts_selected = []
    n_accepted_selected = 0
    n_total_selected = 0
    questions_with_relevant: set = set()
    all_questions: set = set()
    by_hop: dict = {}
    by_edge_type: dict = {}

    with open(retrieval_output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["question_id"]
            all_questions.add(qid)
            for item in record.get("retrieved_context") or []:
                n_total_selected += 1
                if item.get("is_accepted"):
                    n_accepted_selected += 1
                sims_selected.append(item.get("sim"))
                trusts_selected.append(item.get("trust"))

                key = (qid, item.get("answer_id"), ctxrel_mod._text_hash(item.get("chunk_text", "") or ""))
                judged = ctxrel_by_key.get(key)
                if judged is None or judged.get("label") not in LABEL_SCORE:
                    continue
                n_judged += 1
                label = judged["label"]
                label_counts[label] = label_counts.get(label, 0) + 1
                scores.append(LABEL_SCORE[label])
                if label == "RELEVANT":
                    questions_with_relevant.add(qid)

                hop = item.get("hop")
                by_hop.setdefault(hop, {"n": 0, "n_relevant": 0, "n_partial": 0})
                by_hop[hop]["n"] += 1
                if label == "RELEVANT":
                    by_hop[hop]["n_relevant"] += 1
                elif label == "PARTIAL":
                    by_hop[hop]["n_partial"] += 1

                rel_type = item.get("rel_type")
                by_edge_type.setdefault(rel_type, {"n": 0, "n_relevant": 0, "n_partial": 0})
                by_edge_type[rel_type]["n"] += 1
                if label == "RELEVANT":
                    by_edge_type[rel_type]["n_relevant"] += 1
                elif label == "PARTIAL":
                    by_edge_type[rel_type]["n_partial"] += 1

    from ctxrel_sweep import jaccard_overlap

    this_selected = selected_answer_ids_by_alpha.get(this_alpha, {})

    def _overlap_vs(other_alpha):
        other_selected = selected_answer_ids_by_alpha.get(other_alpha, {})
        if not this_selected:
            return None
        overlaps = []
        for qid, aids in this_selected.items():
            other_aids = other_selected.get(qid, set())
            overlaps.append(jaccard_overlap(aids, other_aids))
        return round(sum(overlaps) / len(overlaps), 4) if overlaps else None

    return {
        "n_questions": len(all_questions),
        "n_questions_with_relevant": len(questions_with_relevant),
        "pct_questions_with_relevant": round(100 * len(questions_with_relevant) / len(all_questions), 2) if all_questions else None,
        "pct_relevant": round(100 * label_counts.get("RELEVANT", 0) / n_judged, 2) if n_judged else None,
        "pct_partial": round(100 * label_counts.get("PARTIAL", 0) / n_judged, 2) if n_judged else None,
        "pct_irrelevant": round(100 * label_counts.get("IRRELEVANT", 0) / n_judged, 2) if n_judged else None,
        "mean_relevance_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_sim_selected": round(sum(s for s in sims_selected if s is not None) / len(sims_selected), 4) if sims_selected else None,
        "mean_trust_selected": round(sum(t for t in trusts_selected if t is not None) / len(trusts_selected), 4) if trusts_selected else None,
        "pct_accepted_selected": round(100 * n_accepted_selected / n_total_selected, 2) if n_total_selected else None,
        "jaccard_vs_alpha1": _overlap_vs(1.0),
        "jaccard_vs_alpha0": _overlap_vs(0.0),
        "by_hop": by_hop, "by_edge_type": by_edge_type,
    }


def build_stage1_markdown(per_alpha_metrics: dict[float, dict], selection: dict) -> str:
    lines = [
        "# Stage 1 Report -- C retrieval v3 alpha sweep", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "| alpha | % q w/ >=1 RELEVANT | % RELEVANT | % PARTIAL | % IRRELEVANT | mean score | mean sim (sel) | mean trust (sel) | % accepted (sel) | Jaccard vs a=0 | Jaccard vs a=1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for alpha in sorted(per_alpha_metrics):
        m = per_alpha_metrics[alpha]
        lines.append(
            f"| {alpha} | {m['pct_questions_with_relevant']} | {m['pct_relevant']} | {m['pct_partial']} | "
            f"{m['pct_irrelevant']} | {m['mean_relevance_score']} | {m['mean_sim_selected']} | "
            f"{m['mean_trust_selected']} | {m['pct_accepted_selected']} | {m['jaccard_vs_alpha0']} | {m['jaccard_vs_alpha1']} |"
        )
    lines.append("")
    lines.append("## Relevance by hop and edge type (per alpha)")
    lines.append("")
    for alpha in sorted(per_alpha_metrics):
        m = per_alpha_metrics[alpha]
        lines.append(f"**alpha={alpha}**")
        lines.append("")
        lines.append("| hop | n | % RELEVANT | % PARTIAL |")
        lines.append("|---|---|---|---|")
        for hop, h in sorted(m["by_hop"].items(), key=lambda kv: str(kv[0])):
            pct_rel = round(100 * h["n_relevant"] / h["n"], 2) if h["n"] else None
            pct_part = round(100 * h["n_partial"] / h["n"], 2) if h["n"] else None
            lines.append(f"| {hop} | {h['n']} | {pct_rel} | {pct_part} |")
        lines.append("")
        lines.append("| edge type | n | % RELEVANT | % PARTIAL |")
        lines.append("|---|---|---|---|")
        for et, e in sorted(m["by_edge_type"].items(), key=lambda kv: str(kv[0])):
            pct_rel = round(100 * e["n_relevant"] / e["n"], 2) if e["n"] else None
            pct_part = round(100 * e["n_partial"] / e["n"], 2) if e["n"] else None
            lines.append(f"| {et} | {e['n']} | {pct_rel} | {pct_part} |")
        lines.append("")

    lines.append("## Selection helper output")
    lines.append("")
    lines.append(f"keep_current: {selection['keep_current']}")
    if not selection["keep_current"]:
        lines.append(f"top_two (stage 2 candidates): {selection['top_two']}")
    lines.append(f"reasoning: {selection['reasoning']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI orchestration -- wires (a)-(e) together. Setup mirrors
# c_graphrag.py::main()'s own setup steps (same functions, same
# defaults) so the dev-split sample/candidate pool/FAISS cache/embed
# model are built exactly the way every other C run builds them.
# ---------------------------------------------------------------------

def _build_dev_sample(questions_parquet: str, answers_parquet: str, n_dev: int, seed: int, dev_offset: int):
    import duckdb

    import c_graphrag as cg

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    oversample_pool = 384 * 4  # MAX_PLANNED_N_SAMPLE * 4, same pool as every other condition
    candidates = cg.get_candidate_questions(con, questions_parquet, answers_parquet, oversample_pool, seed)
    candidates = cg.filter_by_token_limit(candidates)
    sample_df = cg.sample_questions_split(candidates, n_dev, seed, split="dev", dev_offset=dev_offset)

    accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
    answers_df = cg.get_accepted_answers(con, answers_parquet, accepted_ids)
    sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
    sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)

    eval_question_ids = sample_df["Id"].astype(int).tolist()
    all_answer_ids_map = cg.get_all_answer_ids_for_questions(con, answers_parquet, eval_question_ids)
    return sample_df, all_answer_ids_map, oversample_pool


def _build_selected_answer_ids_by_alpha(retrieval_output_paths: dict[float, str]) -> dict[float, dict[int, set]]:
    result: dict[float, dict[int, set]] = {}
    for alpha, path in retrieval_output_paths.items():
        per_question: dict[int, set] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = record["question_id"]
                per_question[qid] = {item.get("answer_id") for item in record.get("retrieved_context") or []}
        result[alpha] = per_question
    return result


def run_stage1_sweep(
    questions_parquet: str, answers_parquet: str, kg_workspace_dir: str, embed_model_name: str,
    embedding_cache_path: str, n_dev: int, dev_offset: int, seed: int, judge_id: str, workers: int,
    plan_only: bool, yes: bool, controls_only: bool,
) -> int:
    """Returns a process exit code (0 success, 1 controls failed/aborted)."""
    import c_graphrag as cg
    from embedding_cache import EmbeddingCache
    from llm_judge_context_relevance_v1 import load_ctxrel_records, load_items, run_batch

    from ctxrel_sweep import build_stage1_report, select_stage1_candidates

    print(f"[1/6] Building the dev sample (n={n_dev}, dev_offset={dev_offset}, seed={seed})...")
    sample_df, all_answer_ids_map, oversample_pool = _build_dev_sample(
        questions_parquet, answers_parquet, n_dev, seed, dev_offset,
    )
    print(f"      {len(sample_df)} dev questions ready.")

    if controls_only:
        print("[controls-only] Running ctxrel-v1 positive/negative controls on the dev split...")
        controls = run_controls(sample_df, judge_id, questions_parquet, workers=min(workers, 2))
        _print_controls_result(controls)
        return 0 if controls["passed"] else 1

    print("[2/6] Connecting to Neo4j + loading the FAISS cache (shared, read-only)...")
    driver, database = cg.connect_neo4j(print)
    faiss_index, faiss_ids, faiss_embeddings, id_to_row = cg.load_faiss_cache(Path(kg_workspace_dir), print)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(embed_model_name, device="cpu")
    embedding_cache = EmbeddingCache(embedding_cache_path)

    try:
        print(f"[3/6] Retrieval-only C v3 for alpha in {DEFAULT_ALPHA_VALUES} (no API cost)...")
        retrieval_output_paths: dict[float, str] = {}
        for alpha in DEFAULT_ALPHA_VALUES:
            info = run_retrieval_only_for_alpha(
                sample_df, alpha, driver, database, faiss_index, faiss_ids, faiss_embeddings,
                id_to_row, all_answer_ids_map, embed_model, embedding_cache, questions_parquet,
                output_dir=str(RESULTS_DIR), seed=seed, n_sample=n_dev, oversample_pool=oversample_pool,
            )
            retrieval_output_paths[alpha] = info["output_path"]
            print(f"      alpha={alpha} -> {info['output_path']} ({info['n_questions']} questions)")
    finally:
        embedding_cache.close()

    control_items = build_control_items(sample_df)
    dedup_items = load_items(list(retrieval_output_paths.values()), limit=None)
    plan = build_plan(len(dedup_items), len(control_items), judge_id, workers)
    print()
    print(format_plan(plan))
    print()

    if plan_only:
        print("[plan-only] Stopping before any API call, as requested.")
        return 0

    if not yes:
        answer = input("Proceed with the controls + ctxrel-v1 judging above? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted -- no API calls made.")
            return 1

    print("[4/6] Running ctxrel-v1 positive/negative controls...")
    controls = run_controls(sample_df, judge_id, questions_parquet, workers=min(workers, 2))
    _print_controls_result(controls)
    if not controls["passed"]:
        print("[STOP] Controls failed -- fix ctxrel-v1 before running the main sweep judging. "
              "Retrieval-only outputs above are still valid and will be reused on the next attempt.")
        return 1

    print(f"[5/6] Judging {len(dedup_items)} deduplicated context items with judge_id={judge_id}...")
    ctxrel_output_path = RESULTS_DIR / f"ctxrel_v1_stage1_sweep_{judge_id}.jsonl"
    failures_path = RESULTS_DIR / f"ctxrel_v1_stage1_sweep_{judge_id}_failures.jsonl"
    batch_result = run_batch(dedup_items, [judge_id], questions_parquet, ctxrel_output_path, failures_path, workers)
    print(f"      judged={batch_result['judged']} skipped={batch_result['skipped']} "
          f"failed={batch_result['failed']} label_distribution={batch_result['label_distribution']}")

    print("[6/6] Building the stage-1 report...")
    ctxrel_by_key = load_ctxrel_records([ctxrel_output_path])
    selected_by_alpha = _build_selected_answer_ids_by_alpha(retrieval_output_paths)
    per_alpha_metrics = {
        alpha: compute_alpha_metrics(path, ctxrel_by_key, selected_by_alpha, alpha)
        for alpha, path in retrieval_output_paths.items()
    }
    stage1_report = build_stage1_report({
        alpha: {k: v for k, v in m.items() if k not in ("by_hop", "by_edge_type")}
        for alpha, m in per_alpha_metrics.items()
    })
    selection = select_stage1_candidates(stage1_report)

    report_path = RESULTS_DIR / f"stage1_report_{judge_id}_n{n_dev}_seed{seed}.md"
    report_path.write_text(build_stage1_markdown(per_alpha_metrics, selection))
    print(f"      report written -> {report_path}")
    print()
    print(f"keep_current: {selection['keep_current']}")
    if not selection["keep_current"]:
        print(f"top_two (stage 2 candidates): {selection['top_two']}")
    print(f"reasoning: {selection['reasoning']}")
    return 0


def _print_controls_result(controls: dict) -> None:
    if controls.get("error"):
        print(f"      [ERROR] {controls['error']}")
        return
    for r in controls["results"]:
        status = "OK" if r["match"] else "MISMATCH"
        print(f"      [{status}] {r['control_type']}: expected={r['expected']} actual={r['actual']}")
    print(f"      controls passed: {controls['passed']} (output: {controls.get('output_path')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"))
    parser.add_argument("--answers-parquet", default=os.getenv("ANSWERS_PARQUET"))
    parser.add_argument("--kg-workspace-dir", default=os.getenv(
        "KG_WORKSPACE_DIR", str(Path(__file__).resolve().parents[2] / "01_data_cleaning" / "_kg_workspace")))
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--embedding-cache-path", default=os.getenv(
        "C_EMBEDDING_CACHE_PATH", str(Path(__file__).resolve().parent / "embedding_cache.sqlite3")))
    parser.add_argument("--n-dev", type=int, default=50)
    parser.add_argument("--dev-offset", type=int, default=385)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-id", default="primary")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true",
                         help="Build the dev sample and run retrieval-only (free), show the plan, then stop -- "
                              "no API call is made.")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive y/N confirmation.")
    parser.add_argument("--controls-only", action="store_true",
                         help="Run ONLY the ctxrel-v1 positive/negative controls on the dev split and exit -- "
                              "does not touch retrieval or run the main sweep judging.")
    args = parser.parse_args()

    if not args.questions_parquet or not args.answers_parquet:
        print("[ERROR] --questions-parquet/--answers-parquet (or QUESTIONS_PARQUET/ANSWERS_PARQUET in .env) required.")
        sys.exit(1)

    exit_code = run_stage1_sweep(
        args.questions_parquet, args.answers_parquet, args.kg_workspace_dir, args.embed_model,
        args.embedding_cache_path, args.n_dev, args.dev_offset, args.seed, args.judge_id, args.workers,
        args.plan_only, args.yes, args.controls_only,
    )
    sys.exit(exit_code)
