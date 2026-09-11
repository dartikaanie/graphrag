"""Wraps the existing CLI condition scripts (llm/a_pure_llm, llm/b_rag,
llm/c_graphrag) as callable functions the dashboard can invoke directly --
PLAN_UI_UX.md §5.7. Deliberately a *minimal* wrapper: it imports and calls
the already-validated functions in those scripts (get_candidate_questions,
filter_by_token_limit, sample_questions, get_accepted_answers,
process_sample, append_run_history, ...) rather than re-implementing
retrieval/generation logic here.

Condition A, B, and C are all wired -- PLAN_UI_UX.md §9 sequenced building
them in that order ("test with Condition A first, then B, then C"), which is
also the order they were implemented in.
"""

import importlib
import json as _json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import duckdb

from app.config import get_settings
from app.db.duckdb_client import dedup_answers_cte, dedup_questions_cte
from app.services import run_registry, settings_service


def _questions_parquet() -> str:
    override = settings_service.get_raw_settings().get("questions_parquet")
    return override or get_settings().resolved_questions_parquet()


def _answers_parquet() -> str:
    override = settings_service.get_raw_settings().get("answers_parquet")
    return override or get_settings().resolved_answers_parquet()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@lru_cache
def _condition_a_module():
    mod = importlib.import_module("llm.a_pure_llm.a_baseline_replication")
    # The CLI script hardcodes LOG_DIR = Path("logs") assuming its own cwd
    # (true when run as `python a_baseline_replication.py` from inside
    # llm/a_pure_llm/); the backend process has a different cwd, so without
    # this, append_run_history()/setup_logging() would silently write to
    # the wrong place instead of the logs/ that Phase 6 History reads from.
    mod.LOG_DIR = REPO_ROOT / "llm" / "a_pure_llm" / "logs"
    return mod


@lru_cache
def _condition_b_module():
    mod = importlib.import_module("llm.b_rag.b_condition_b_rag")
    mod.LOG_DIR = REPO_ROOT / "llm" / "b_rag" / "logs"
    return mod


@lru_cache
def _condition_c_module():
    mod = importlib.import_module("llm.c_graphrag.c_graphrag")
    mod.LOG_DIR = REPO_ROOT / "llm" / "c_graphrag" / "logs"
    return mod


@lru_cache
def _embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache
def _embed_model_cpu():
    # Condition C explicitly pins device="cpu" (see c_graphrag.py comments on
    # MPS memory creep over long loops) -- kept as a separate cached instance
    # so Condition A/B's default-device model isn't silently reused here.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def _anchor_output_path(output_path: Path, condition_folder: str) -> Path:
    if output_path.is_absolute():
        return output_path
    return (REPO_ROOT / "llm" / condition_folder / output_path).resolve()


def _read_already_done(output_path: Path) -> set[int]:
    already_done: set[int] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    already_done.add(_json.loads(line)["question_id"])
                except Exception:
                    continue
    return already_done


def _make_on_progress(run_id: str):
    def on_progress(event: dict[str, Any]):
        run_registry.append_result(run_id, {
            "question_id": event["question_id"],
            "index": event["index"],
            "total": event["total"],
            "status": event["status"],
            "similarity": event.get("similarity"),
            "error": event.get("error"),
        })
    return on_progress


def _safe_model_name(model: str) -> str:
    return model.replace("/", "-").replace(":", "-").replace(".", "-")


def _build_single_question_df(question_id: int, cond_module):
    """One-row DataFrame shaped like the batch sample_df (Id, Title, Body,
    Tags, AcceptedAnswerId, ViewCount, Score, n_tokens, AcceptedAnswerBody),
    reusing the same dedup CTEs the rest of the backend already relies on
    (app.db.duckdb_client) rather than duplicating that SQL here.
    """
    con = duckdb.connect()
    q_df = con.execute(
        f"""
        SELECT Id, Title, Body, Tags, AcceptedAnswerId, ViewCount, Score
        FROM ({dedup_questions_cte()})
        WHERE Id = ?
        """,
        [question_id],
    ).df()
    if q_df.empty:
        raise ValueError(f"Question {question_id} not found")
    if q_df.loc[0, "AcceptedAnswerId"] in (None, 0) or q_df["AcceptedAnswerId"].isna().any():
        raise ValueError(f"Question {question_id} has no accepted answer")

    # limit=10**9 -- reuse filter_by_token_limit purely to compute n_tokens,
    # not to exclude: a single-question run is explicit user intent, it
    # should never be silently dropped for being over the token limit.
    q_df = cond_module.filter_by_token_limit(q_df, limit=10**9)

    accepted_id = int(q_df.loc[0, "AcceptedAnswerId"])
    a_df = con.execute(
        f"""
        SELECT Id AS AcceptedAnswerId, Body AS AcceptedAnswerBody
        FROM ({dedup_answers_cte()})
        WHERE Id = ?
        """,
        [accepted_id],
    ).df()
    if a_df.empty:
        raise ValueError(f"Accepted answer {accepted_id} for question {question_id} not found in Answers parquet")

    return q_df.merge(a_df, on="AcceptedAnswerId", how="left")


def _summarize(results: list[dict]) -> dict[str, Any]:
    sims = [r["cosine_similarity"] for r in results if r.get("cosine_similarity") is not None]
    if not sims:
        return {"n_processed": len(results), "cosine_similarity_mean": None}
    n_above = sum(1 for s in sims if s > 0.5)
    return {
        "n_processed": len(results),
        "cosine_similarity_mean": round(sum(sims) / len(sims), 4),
        "cosine_similarity_median": round(sorted(sims)[len(sims) // 2], 4),
        "pct_similarity_above_0_5": round(n_above / len(sims) * 100, 1),
    }


def run_condition_a(run_id: str, params: dict[str, Any]) -> None:
    """Batch or single-question run of Condition A, driven by dashboard
    params instead of argparse/CLI. Writes to the same auto-named .jsonl
    output (build_output_path) and the same logs/run_history.jsonl as a CLI
    run, so History (Phase 6) sees dashboard runs and CLI runs identically.
    """
    cond = _condition_a_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(run_id, status="running", started_at=run_started_at.isoformat())

    provider = params["provider"]
    model = params["model"]

    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), cond)
            output_path = Path("results") / f"condition_a_{provider}_{_safe_model_name(model)}_single_{params['question_id']}.jsonl"
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = int(params.get("oversample_pool") or n_sample * 3)
            con = duckdb.connect()
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            output_path = cond.build_output_path("results", provider, model, n_sample, seed)

        output_path = _anchor_output_path(output_path, "a_pure_llm")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path)

        run_registry.update_run(
            run_id,
            output_path=str(output_path),
            progress={"current": 0, "total": len(sample_df)},
        )

        embed_model = _embed_model()

        results = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, model, output_path, already_done,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
        )

        cancelled = run_registry.is_cancelled(run_id)
        summary = _summarize(results)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        cond.append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": params.get("seed"),
            "output_path": str(output_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        run_registry.update_run(
            run_id,
            status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
        )
    except Exception as e:
        run_registry.update_run(
            run_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )


def run_condition_b(run_id: str, params: dict[str, Any]) -> None:
    """Batch or single-question run of Condition B (dense RAG). Same shape as
    run_condition_a, plus building/loading the flat FAISS retrieval corpus
    (llm/b_rag/index_cache/), reusing build_retrieval_corpus/chunk_corpus/
    build_or_load_faiss_index exactly as the CLI does.
    """
    cond = _condition_b_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(run_id, status="running", started_at=run_started_at.isoformat())

    provider = params["provider"]
    model = params["model"]
    top_k = int(params.get("top_k") or 5)
    index_pool = 8000
    token_chunk_limit = 400
    embed_model_name = "all-MiniLM-L6-v2"

    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)
        con = duckdb.connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            output_path = Path("results") / f"condition_b_{provider}_{_safe_model_name(model)}_single_{params['question_id']}.jsonl"
            seed = 42
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = int(params.get("oversample_pool") or n_sample * 4)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            output_path = cond.build_output_path("results", provider, model, n_sample, seed)

        output_path = _anchor_output_path(output_path, "b_rag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        corpus_df = cond.build_retrieval_corpus(
            con, _questions_parquet(), _answers_parquet(),
            index_pool, seed, eval_question_ids, accepted_ids,
        )
        chunk_df = cond.chunk_corpus(corpus_df, token_chunk_limit)
        embed_model = _embed_model()
        index_cache_dir = REPO_ROOT / "llm" / "b_rag" / "index_cache"
        index, meta_df = cond.build_or_load_faiss_index(
            chunk_df, embed_model, index_cache_dir, index_pool, token_chunk_limit, embed_model_name, rebuild=False,
        )

        require_citation = bool(params.get("require_citation", True))
        results = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, index, meta_df, top_k, model, output_path, already_done,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
            require_citation=require_citation,
        )

        cancelled = run_registry.is_cancelled(run_id)
        summary = _summarize(results)
        if require_citation and results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        cond.append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "condition": "B",
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": seed,
            "require_citation": require_citation,
            "index_pool": index_pool,
            "top_k": top_k,
            "output_path": str(output_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        run_registry.update_run(
            run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(), summary=summary,
        )
    except Exception as e:
        run_registry.update_run(
            run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )


def run_condition_c(run_id: str, params: dict[str, Any]) -> None:
    """Batch or single-question run of Condition C (GraphRAG). Same shape as
    run_condition_a/b, plus connecting to Neo4j and loading the pre-built
    FAISS/embedding cache from 01_data_cleaning/_kg_workspace (built by
    11_densify_embedding_similarity.py -- NOT rebuilt here, same as the CLI).
    """
    cond = _condition_c_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(run_id, status="running", started_at=run_started_at.isoformat())

    provider = params["provider"]
    model = params["model"]
    top_k = int(params.get("top_k") or 5)
    n_anchor = int(params.get("n_anchor") or 3)
    n_semantic_expansion = int(params.get("n_semantic_expansion") or 3)
    token_chunk_limit = 400

    driver = None
    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)
        con = duckdb.connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            output_path = Path("results") / f"condition_c_{provider}_{_safe_model_name(model)}_single_{params['question_id']}.jsonl"
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = int(params.get("oversample_pool") or n_sample * 4)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            output_path = cond.build_output_path("results", provider, model, n_sample, seed)

        output_path = _anchor_output_path(output_path, "c_graphrag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        all_answer_ids_map = cond.get_all_answer_ids_for_questions(con, _answers_parquet(), eval_question_ids)

        driver, database = cond.connect_neo4j(print)
        kg_workspace_dir = REPO_ROOT / "01_data_cleaning" / "_kg_workspace"
        faiss_index, faiss_ids, faiss_embeddings, id_to_row = cond.load_faiss_cache(kg_workspace_dir, print)
        embed_model = _embed_model_cpu()

        results, stats = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, driver, database,
            faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
            top_k, n_anchor, n_semantic_expansion, token_chunk_limit, model, output_path, already_done,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
        )

        cancelled = run_registry.is_cancelled(run_id) or stats["interrupted"]
        summary = _summarize(results)
        if results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            latencies = [r["retrieval_latency_sec"] for r in results if r.get("retrieval_latency_sec") is not None]
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
            if latencies:
                summary["avg_retrieval_latency_sec"] = round(sum(latencies) / len(latencies), 3)

        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        cond.append_run_history({
            "run_started_at": run_started_at.isoformat(),
            "condition": "C",
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": params.get("seed"),
            "top_k": top_k,
            "n_anchor": n_anchor,
            "n_semantic_expansion": n_semantic_expansion,
            "output_path": str(output_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        run_registry.update_run(
            run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(), summary=summary,
        )
    except Exception as e:
        run_registry.update_run(
            run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )
    finally:
        if driver is not None:
            driver.close()


RUNNERS: dict[str, Callable[[str, dict[str, Any]], None]] = {
    "A": run_condition_a,
    "B": run_condition_b,
    "C": run_condition_c,
}


def start_run(run_id: str, condition: str, params: dict[str, Any]) -> None:
    runner = RUNNERS.get(condition.upper())
    if runner is None:
        run_registry.update_run(run_id, status="failed", error=f"Unknown condition '{condition}'")
        return
    # Apply Settings-page values (API keys, Neo4j credentials) to the process
    # env right before running, so a saved Settings key actually takes
    # effect for dashboard-triggered runs instead of only ever reading
    # whatever the repo-root .env happened to have at backend startup.
    settings_service.apply_to_environment()
    runner(run_id, params)
