"""Wraps the existing CLI condition scripts (llm/a_pure_llm, llm/b_rag,
llm/c_graphrag) as callable functions the dashboard can invoke directly --
PLAN_UI_UX.md §5.7. Deliberately a *minimal* wrapper: it imports and calls
the already-validated functions in those scripts (get_candidate_questions,
filter_by_token_limit, sample_questions, get_accepted_answers,
process_sample, append_run_history, ...) rather than re-implementing
retrieval/generation logic here.

Condition A, B, C, and D are all wired -- PLAN_UI_UX.md §9 sequenced building
them in that order ("test with Condition A first, then B, then C"), which is
also the order they were implemented in. Condition D (Dual-Level Retrieval,
adaptasi LightRAG) reuses the SAME Neo4j KG & FAISS cache as Condition C --
see llm/d_lightrag/d_lightrag.py.

RESOURCE BUDGET (target machine: 8GB RAM MacBook M2) -- "Run All Conditions"
used to fire A/B/C/D as four fully-parallel background threads, which could
mean 4x embedding model + 2x Neo4j driver + 2x FAISS/embedding-cache memmap
+ 4x DuckDB connection alive AT ONCE, on top of n_sample now defaulting to
384 (see RunAllConditionsPage.tsx). That was enough to exhaust memory and
hang/crash the machine. Three mitigations, all in this file:
  1. `_HEAVY_RUN_LOCK` -- A/B/C/D all run ONE AT A TIME, never concurrently.
     Condition A also runs its own DuckDB sampling query (oversample_pool
     rows, capped by `_duckdb_connect()`'s pragmas below) and must not
     overlap C/D's ~4GB FAISS footprint on an 8GB machine, so it shares
     this lock too rather than being exempted from it.
  2. `_faiss_cache()` -- Condition C and D read the EXACT SAME on-disk
     FAISS index + embedding memmap (see d_lightrag.py's docstring: "SAME
     KG & FAISS cache as Condition C"); loading it twice doubled that
     memory footprint for no reason. Now loaded once and shared.
  3. `_duckdb_connect()` -- every duckdb.connect() in this module goes
     through one helper that applies memory_limit='2GB'/threads=2/
     preserve_insertion_order=false, so DuckDB itself is capped instead of
     defaulting to "use all available resources" on a machine that can't
     spare them, especially with multiple connections alive across A
     overlapping a queued/running B/C/D.
"""

import importlib
import json as _json
import sys
import threading
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

# Only one of A/B/C/D actually running at a time -- see module docstring
# ("RESOURCE BUDGET"). Condition A also runs a DuckDB sampling query and
# must not overlap C/D's ~4GB FAISS footprint on an 8GB machine, so it
# shares this lock with B/C/D rather than being exempted from it.
_HEAVY_RUN_LOCK = threading.Lock()
HEAVY_CONDITIONS = {"A", "B", "C", "D"}

# Judge runs (llm_judge_hallucination/context_relevance/answer_relevance,
# via run_judge_batch_dashboard()) do NOT share _HEAVY_RUN_LOCK's mutual
# exclusion by default -- they never touch the two actual causes of the
# original crash (the ~4GB FAISS/embedding memmap, the Neo4j driver). A
# judge run is LLM-API-call-bound (mostly waiting on network) plus, for the
# context-relevance judge, ONE batched DuckDB query already capped by the
# same 2GB/2-thread pragma as everywhere else (llm/evaluation/
# _judge_common.py's load_question_bodies()). Forcing a judge to queue
# behind a 384-question A/B/C/D run (which can take tens of minutes and
# shares none of its resources) would be a real UX regression for no
# memory-safety benefit in the common case. A small SEPARATE semaphore
# bounds how many judges run at once -- 2 concurrent judges means at most
# ~4GB of DuckDB headroom used (2 x 2GB), leaving room for whatever else is
# running, without serializing judges behind generators.
#
# That said, on an 8GB machine a judge run's own DuckDB usage stacking on
# top of a live A/B/C/D run's FAISS/Neo4j footprint is still real headroom
# pressure. The "judges_wait_for_heavy_run" Settings toggle (default ON,
# see settings_service.DEFAULTS) lets a judge run ALSO wait for
# _HEAVY_RUN_LOCK to be free before proceeding, on top of -- not instead of
# -- _JUDGE_SEMAPHORE's own limit of 2 concurrent judges. See
# start_judge_run() below.
_JUDGE_CONCURRENCY_LIMIT = 2
_JUDGE_SEMAPHORE = threading.Semaphore(_JUDGE_CONCURRENCY_LIMIT)

# Best-effort FIFO queue-position bookkeeping -- threading.Lock/Semaphore
# don't expose "how many are waiting," so this list is maintained alongside
# them purely for the "queued (position N)" UI. Not itself used for mutual
# exclusion (the Lock/Semaphore above still do that); "best-effort" because
# Python doesn't guarantee FIFO wake order for blocked acquire() calls, so
# the displayed position can occasionally be off by one -- it's informational,
# not a scheduling guarantee.
_HEAVY_QUEUE_LOCK = threading.Lock()
_HEAVY_QUEUE: list[str] = []
_JUDGE_QUEUE_LOCK = threading.Lock()
_JUDGE_QUEUE: list[str] = []


def _enter_queue(queue_lock: threading.Lock, queue: list[str], run_id: str) -> None:
    with queue_lock:
        queue.append(run_id)
        snapshot = list(queue)
    for i, rid in enumerate(snapshot, start=1):
        run_registry.update_run(rid, queue_position=i)


def _leave_queue(queue_lock: threading.Lock, queue: list[str], run_id: str) -> None:
    with queue_lock:
        if run_id in queue:
            queue.remove(run_id)
        snapshot = list(queue)
    run_registry.update_run(run_id, queue_position=None)
    for i, rid in enumerate(snapshot, start=1):
        run_registry.update_run(rid, queue_position=i)

# Single source of truth for the "auto" oversample_pool fallback -- MUST
# equal MAX_PLANNED_N_SAMPLE * 4, where MAX_PLANNED_N_SAMPLE (384) is the
# CLI scripts' own constant (a_baseline_replication.py, c_graphrag.py,
# d_lightrag.py all define it locally as the size of the official n=384
# thesis run). Nested sampling -- the property that an n=10 pilot sample is
# an exact PREFIX of the n=384 sample -- requires the SAME oversample_pool
# (and seed) on every run regardless of n_sample, which is why this is one
# constant reused by all four conditions instead of each condition computing
# its own multiple of n_sample (previously A used n_sample*3, B/C/D used
# n_sample*4 -- neither matches the CLI's own MAX_PLANNED_N_SAMPLE*4
# convention, and neither keeps smaller pilot runs nested inside the n=384
# sample the way a fixed pool does). Exposed to the frontend via
# GET /api/config/defaults (routers/config.py) so 1536 isn't hard-coded in
# multiple places in the UI.
MAX_PLANNED_N_SAMPLE = 384
DEFAULT_OVERSAMPLE_POOL = MAX_PLANNED_N_SAMPLE * 4


def resolve_oversample_pool(params: dict[str, Any]) -> int:
    """The ONE place all four run_condition_x() functions get their
    effective oversample_pool from -- pulled out as its own pure function
    (rather than the same inline expression repeated 4x) so it's directly
    unit-testable without mocking DuckDB/Neo4j/an LLM client."""
    return int(params.get("oversample_pool") or DEFAULT_OVERSAMPLE_POOL)


def _duckdb_connect() -> duckdb.DuckDBPyConnection:
    """Every duckdb.connect() in this module goes through here so the same
    memory/thread cap applies everywhere -- see module docstring point 3.
    Same pragmas already used by the standalone analysis scripts (e.g.
    analyze_retrieval_quality.py, llm/evaluation/_judge_common.py)."""
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    return con


def _process_memory_mb() -> float | None:
    """Current process RSS in MB via psutil, or None if psutil isn't
    installed -- callers must handle None (log nothing) rather than crash,
    per "using psutil if it is already available; otherwise skip this."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


@lru_cache
def _faiss_cache(kg_workspace_dir_str: str):
    """Shared FAISS index + embedding-cache loader for Condition C AND D --
    see module docstring point 2. Both conditions' own load_faiss_cache()
    read the identical files from the same kg_workspace_dir; this loads
    them ONCE (cached on the workspace-dir string, which is always the same
    path in practice) and both conditions reuse the result instead of each
    memmap-ing the embeddings file and reading the FAISS index a second
    time. Uses Condition C's module to do the actual reading (D's version
    is byte-for-byte the same function) -- an implementation detail, not a
    behavior difference for D.

    Logs process RSS before/after (psutil, if installed) so a memory-tight
    run can see exactly how much this one load step cost -- this is the
    single largest allocation in the whole crash story (module docstring
    "RESOURCE BUDGET": ~4GB for the embeddings memmap + FAISS index).
    release_faiss_cache() below clears this LRU cache on demand (Settings
    page button / admin endpoint) so that memory can be given back without
    restarting the backend process."""
    before = _process_memory_mb()
    if before is not None:
        print(f"      [faiss-cache] RSS before load: {before:.0f} MB")
    cond = _condition_c_module()
    result = cond.load_faiss_cache(Path(kg_workspace_dir_str), print)
    after = _process_memory_mb()
    if after is not None:
        print(f"      [faiss-cache] RSS after load: {after:.0f} MB (+{after - before:.0f} MB)")
    return result


def release_faiss_cache() -> dict[str, Any]:
    """Clears the @lru_cache on _faiss_cache() -- the NEXT Condition C/D run
    re-reads the FAISS index + embedding memmap from disk (a few seconds)
    instead of reusing the in-memory copy. Frees ~4GB of RSS immediately
    (Python's allocator returns large mmap'd/numpy-backed buffers to the OS
    promptly on dereference, unlike small-object heap fragmentation) -- use
    this between a heavy C/D session and something else memory-hungry
    (e.g. a big judge batch) without restarting the whole backend."""
    before = _process_memory_mb()
    _faiss_cache.cache_clear()
    after = _process_memory_mb()
    return {
        "released": True,
        "rss_before_mb": round(before, 1) if before is not None else None,
        "rss_after_mb": round(after, 1) if after is not None else None,
    }


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
def _condition_d_module():
    mod = importlib.import_module("llm.d_lightrag.d_lightrag")
    mod.LOG_DIR = REPO_ROOT / "llm" / "d_lightrag" / "logs"
    return mod


@lru_cache
def _judge_module():
    mod = importlib.import_module("llm.evaluation.llm_judge_hallucination")
    mod.LOG_DIR = REPO_ROOT / "llm" / "evaluation" / "logs"
    mod.RESULTS_DIR = REPO_ROOT / "llm" / "evaluation" / "results"
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


def _make_judge_on_progress(run_id: str):
    """run_judge_batch()'s on_progress events carry {question_id, status,
    ...} -- no index/total (it streams straight off the input file, it
    doesn't know the total up front the way process_sample() does). The
    "already_complete" event (question_id=None) is a one-off signal, not a
    per-question result, so it updates status directly instead of being
    appended to the results list."""
    def on_progress(event: dict[str, Any]):
        if event["status"] == "already_complete":
            return
        run_registry.append_result(run_id, {
            "question_id": event["question_id"],
            "status": event["status"],
            "hallucination_label": event.get("hallucination_label"),
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
    con = _duckdb_connect()
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

    # Recorded into run_history regardless of mode -- None for single-question
    # mode, where there's no candidate pool to speak of. oversample_pool is
    # the EFFECTIVE value actually passed to get_candidate_questions() (not
    # just whatever the request happened to say), so a paired comparison can
    # trust it; n_candidates_after_token_filter is a cheap sanity signal that
    # the pool was actually large enough to draw n_sample from without
    # running dry (see DEFAULT_OVERSAMPLE_POOL in Phase 2).
    oversample_pool: int | None = None
    n_candidates_after_token_filter: int | None = None

    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), cond)
            output_path = Path("results") / f"condition_a_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = resolve_oversample_pool(params)
            con = _duckdb_connect()
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            n_candidates_after_token_filter = len(candidates)
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
        summary["duration_sec"] = duration
        cond.append_run_history({
            "prompt_version": cond.PROMPT_VERSION,
            "run_started_at": run_started_at.isoformat(),
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": params.get("seed"),
            "oversample_pool": oversample_pool,
            "n_candidates_after_token_filter": n_candidates_after_token_filter,
            # Condition A never retrieves anything, so log_full_candidates has
            # no effect here -- recorded anyway (honestly reflecting whatever
            # was submitted) purely so History/Compare shows the SAME field
            # across all four conditions rather than omitting it for A alone.
            "log_full_candidates": bool(params.get("log_full_candidates")),
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
    log_full_candidates = bool(params.get("log_full_candidates"))
    oversample_pool: int | None = None
    n_candidates_after_token_filter: int | None = None

    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            output_path = Path("results") / f"condition_b_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
            seed = 42
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = resolve_oversample_pool(params)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            n_candidates_after_token_filter = len(candidates)
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
            require_citation=require_citation, log_full_candidates=log_full_candidates,
        )

        cancelled = run_registry.is_cancelled(run_id)
        summary = _summarize(results)
        if require_citation and results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
        if results:
            latencies = [r["retrieval_latency_sec"] for r in results if r.get("retrieval_latency_sec") is not None]
            if latencies:
                summary["avg_retrieval_latency_sec"] = round(sum(latencies) / len(latencies), 3)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        cond.append_run_history({
            "prompt_version": cond.PROMPT_VERSION,
            "run_started_at": run_started_at.isoformat(),
            "condition": "B",
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": seed,
            "oversample_pool": oversample_pool,
            "n_candidates_after_token_filter": n_candidates_after_token_filter,
            "require_citation": require_citation,
            "log_full_candidates": log_full_candidates,
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
    fusion_mode = params.get("fusion_mode") or "trust_weighted"
    fusion_w_path_trust = float(params.get("fusion_w_path_trust") if params.get("fusion_w_path_trust") is not None else 0.7)
    fusion_w_intrinsic = float(params.get("fusion_w_intrinsic") if params.get("fusion_w_intrinsic") is not None else 0.3)
    semantic_expansion_trust_cap = float(params.get("semantic_expansion_trust_cap") if params.get("semantic_expansion_trust_cap") is not None else 0.4)
    require_grounding = params.get("require_grounding")
    require_grounding = True if require_grounding is None else bool(require_grounding)
    enable_semantic_expansion = params.get("enable_semantic_expansion")
    enable_semantic_expansion = True if enable_semantic_expansion is None else bool(enable_semantic_expansion)
    log_full_candidates = bool(params.get("log_full_candidates"))
    token_chunk_limit = 400
    oversample_pool: int | None = None
    n_candidates_after_token_filter: int | None = None

    driver = None
    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            output_path = Path("results") / f"condition_c_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = resolve_oversample_pool(params)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            n_candidates_after_token_filter = len(candidates)
            sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            output_path = cond.build_output_path(
                "results", provider, model, n_sample, seed,
                fusion_mode=fusion_mode, fusion_w_path_trust=fusion_w_path_trust, fusion_w_intrinsic=fusion_w_intrinsic,
                require_grounding=require_grounding, enable_semantic_expansion=enable_semantic_expansion,
            )

        output_path = _anchor_output_path(output_path, "c_graphrag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        all_answer_ids_map = cond.get_all_answer_ids_for_questions(con, _answers_parquet(), eval_question_ids)

        driver, database = cond.connect_neo4j(print)
        kg_workspace_dir = REPO_ROOT / "01_data_cleaning" / "_kg_workspace"
        faiss_index, faiss_ids, faiss_embeddings, id_to_row = _faiss_cache(str(kg_workspace_dir))
        embed_model = _embed_model_cpu()

        results, stats = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, driver, database,
            faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
            top_k, n_anchor, n_semantic_expansion, token_chunk_limit, model, output_path, already_done,
            fusion_mode=fusion_mode, fusion_w_path_trust=fusion_w_path_trust, fusion_w_intrinsic=fusion_w_intrinsic,
            semantic_expansion_trust_cap=semantic_expansion_trust_cap,
            require_grounding=require_grounding, enable_semantic_expansion=enable_semantic_expansion,
            log_full_candidates=log_full_candidates,
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
        summary["duration_sec"] = duration
        summary["require_grounding"] = require_grounding
        summary["enable_semantic_expansion"] = enable_semantic_expansion
        cond.append_run_history({
            "prompt_version": cond.PROMPT_VERSION,
            "c_retrieval_version": cond.C_RETRIEVAL_VERSION,
            "run_started_at": run_started_at.isoformat(),
            "condition": "C",
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": params.get("seed"),
            "oversample_pool": oversample_pool,
            "n_candidates_after_token_filter": n_candidates_after_token_filter,
            "top_k": top_k,
            "n_anchor": n_anchor,
            "n_semantic_expansion": n_semantic_expansion,
            "fusion_mode": fusion_mode,
            "fusion_w_path_trust": fusion_w_path_trust,
            "fusion_w_answer_intrinsic_trust": fusion_w_intrinsic,
            "semantic_expansion_trust_cap": semantic_expansion_trust_cap,
            "require_grounding": require_grounding,
            "enable_semantic_expansion": enable_semantic_expansion,
            "log_full_candidates": log_full_candidates,
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


def run_condition_d(run_id: str, params: dict[str, Any]) -> None:
    """Batch or single-question run of Condition D (Dual-Level Retrieval,
    adaptasi LightRAG). Struktur SAMA PERSIS run_condition_c() -- sampling
    1-4 identik, connect_neo4j, load_faiss_cache -- tapi memakai KG & FAISS
    cache Kondisi C APA ADANYA (TIDAK membangun apa pun baru) dan memanggil
    process_sample() Kondisi D (dual-level retrieval, bukan trust-weighted).
    """
    cond = _condition_d_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(run_id, status="running", started_at=run_started_at.isoformat())

    provider = params["provider"]
    model = params["model"]
    top_k = int(params.get("top_k") or 5)
    n_low_level = int(params.get("n_low_level") or 3)
    n_high_level = int(params.get("n_high_level") or 3)
    require_grounding = params.get("require_grounding")
    require_grounding = True if require_grounding is None else bool(require_grounding)
    token_chunk_limit = 400
    oversample_pool: int | None = None
    n_candidates_after_token_filter: int | None = None

    driver = None
    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=print)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            output_path = Path("results") / f"condition_d_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = resolve_oversample_pool(params)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            n_candidates_after_token_filter = len(candidates)
            sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            output_path = cond.build_output_path(
                "results", provider, model, n_sample, seed, require_grounding=require_grounding,
            )

        output_path = _anchor_output_path(output_path, "d_lightrag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        all_answer_ids_map = cond.get_all_answer_ids_for_questions(con, _answers_parquet(), eval_question_ids)

        driver, database = cond.connect_neo4j(print)
        kg_workspace_dir = REPO_ROOT / "01_data_cleaning" / "_kg_workspace"
        faiss_index, faiss_ids, faiss_embeddings, id_to_row = _faiss_cache(str(kg_workspace_dir))
        embed_model = _embed_model_cpu()

        results, stats = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, driver, database,
            faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
            top_k, n_low_level, n_high_level, token_chunk_limit, model, output_path, already_done,
            require_grounding=require_grounding,
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
        summary["duration_sec"] = duration
        summary["require_grounding"] = require_grounding
        cond.append_run_history({
            "prompt_version": cond.PROMPT_VERSION,
            "d_retrieval_version": cond.D_RETRIEVAL_VERSION,
            "run_started_at": run_started_at.isoformat(),
            "condition": "D",
            "status": "cancelled" if cancelled else ("success" if results else "no_results"),
            "provider": provider,
            "model": model,
            "n_sample_target": params.get("n_sample", 1),
            "n_processed": len(results),
            "seed": params.get("seed"),
            "oversample_pool": oversample_pool,
            "n_candidates_after_token_filter": n_candidates_after_token_filter,
            "top_k": top_k,
            "n_low_level": n_low_level,
            "n_high_level": n_high_level,
            "require_grounding": require_grounding,
            # Condition D's process_sample() doesn't accept/use
            # log_full_candidates at all (only B/C do) -- recorded anyway
            # (honestly reflecting whatever was submitted) for the same
            # cross-condition-consistency reason as Condition A above.
            "log_full_candidates": bool(params.get("log_full_candidates")),
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


CONDITION_RESULTS_DIR = {
    "A": REPO_ROOT / "llm" / "a_pure_llm" / "results",
    "B": REPO_ROOT / "llm" / "b_rag" / "results",
    "C": REPO_ROOT / "llm" / "c_graphrag" / "results",
    "D": REPO_ROOT / "llm" / "d_lightrag" / "results",
}


def run_judge_batch_dashboard(run_id: str, params: dict[str, Any]) -> None:
    """LLM-as-judge batch run triggered from the dashboard
    (backend/app/routers/judge.py). Pola struktur SAMA seperti
    run_condition_c(): resolve config dari params dengan fallback ke
    settings_service (pola sama _questions_parquet()/_answers_parquet()),
    import llm.evaluation.llm_judge_hallucination via importlib
    (_judge_module(), LOG_DIR/RESULTS_DIR di-set ke path absolut sama
    seperti _condition_c_module() dst.), panggil run_judge_batch() atau
    run_kappa_validation() dari modul itu sesuai params.kappa_validation.
    Output path SELALU dihitung otomatis oleh modul itu sendiri
    (build_judge_output_path) -- backend tidak pernah menentukannya.
    """
    cond = _judge_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(run_id, status="running", started_at=run_started_at.isoformat())

    settings_service.apply_to_environment()
    settings = settings_service.get_raw_settings()

    condition = params["condition"].upper()
    input_path = Path(params["input_path"])
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    judge_provider = params.get("judge_provider") or settings["judge_provider"]
    judge_model = params.get("judge_model") or settings["judge_model"]
    judge_temperature = params.get("judge_temperature")
    judge_temperature = float(judge_temperature) if judge_temperature is not None else float(settings["judge_temperature"])
    majority_rounds = params.get("majority_rounds")
    majority_rounds = int(majority_rounds) if majority_rounds is not None else int(settings["judge_majority_rounds"])
    force = bool(params.get("force"))
    kappa_validation = bool(params.get("kappa_validation"))
    secondary_judge_provider = params.get("secondary_judge_provider") or settings["secondary_judge_provider"]
    secondary_judge_model = params.get("secondary_judge_model") or settings["secondary_judge_model"]
    kappa_sample_size = params.get("kappa_sample_size")
    kappa_sample_size = int(kappa_sample_size) if kappa_sample_size is not None else int(settings["kappa_sample_size"])

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"input_path tidak ditemukan: {input_path}")

        with open(input_path) as f:
            n_lines = sum(1 for line in f if line.strip())
        run_registry.update_run(run_id, progress={"current": 0, "total": n_lines})

        if kappa_validation:
            kappa_result = cond.run_kappa_validation(
                input_path, judge_provider, judge_model, judge_temperature,
                secondary_judge_provider, secondary_judge_model, judge_temperature,
                majority_rounds, condition, kappa_sample_size, params.get("seed") or 42,
                force=force,
            )
            summary = dict(kappa_result["primary_summary"])
            summary["kappa_value"] = kappa_result["kappa_value"]
            summary["kappa_interpretation"] = kappa_result["interpretation"]
            summary["judge_models_identical"] = kappa_result["judge_models_identical"]
        else:
            kappa_result = None
            summary = cond.run_judge_batch(
                input_path, judge_provider, judge_model, judge_temperature, majority_rounds, condition,
                force=force, on_progress=_make_judge_on_progress(run_id),
                check_cancel=lambda: run_registry.is_cancelled(run_id),
            )

        output_path = summary["output_path"]
        cancelled = run_registry.is_cancelled(run_id)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        run_registry.update_run(run_id, output_path=output_path)

        cond.append_judge_run_history(
            condition, str(input_path), output_path, judge_provider, judge_model, judge_temperature,
            majority_rounds, summary, kappa_result=kappa_result,
        )
        run_registry.update_run(
            run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(), summary=summary,
        )
    except Exception as e:
        run_registry.update_run(
            run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )


def start_judge_run(run_id: str, params: dict[str, Any]) -> None:
    """Entry point routers/judge.py's background thread calls (instead of
    run_judge_batch_dashboard() directly) -- bounds concurrent judge runs to
    _JUDGE_CONCURRENCY_LIMIT via _JUDGE_SEMAPHORE, independent of
    _HEAVY_RUN_LOCK/A/B/C/D by default (see the semaphore's own comment for
    why). A judge run waiting for a free slot shows status "queued" +
    `queue_position`, same as a queued A/B/C/D run, and can be cancelled
    while still queued.

    When the "judges_wait_for_heavy_run" setting is on (default), this ALSO
    blocks until _HEAVY_RUN_LOCK is free before even trying for a semaphore
    slot -- a plain acquire-then-immediately-release gate, not a claim held
    for the judge run's whole duration, since a heavy run may start again
    right after the gate opens. That's an accepted, inherent race in "wait
    until no heavy run is executing" as stated (it bounds the *chance* of
    overlap, not a hard guarantee), and still layers under the semaphore:
    the 2-judges-at-once cap applies regardless of this setting.
    """
    settings_service.apply_to_environment()
    wait_for_heavy_run = bool(settings_service.get_raw_settings().get("judges_wait_for_heavy_run", True))
    run_registry.update_run(run_id, status="queued")
    _enter_queue(_JUDGE_QUEUE_LOCK, _JUDGE_QUEUE, run_id)
    try:
        if wait_for_heavy_run:
            with _HEAVY_RUN_LOCK:
                pass
            if run_registry.is_cancelled(run_id):
                run_registry.update_run(
                    run_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
        _JUDGE_SEMAPHORE.acquire()
        try:
            _leave_queue(_JUDGE_QUEUE_LOCK, _JUDGE_QUEUE, run_id)
            if run_registry.is_cancelled(run_id):
                run_registry.update_run(
                    run_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            run_judge_batch_dashboard(run_id, params)
        finally:
            _JUDGE_SEMAPHORE.release()
    finally:
        _leave_queue(_JUDGE_QUEUE_LOCK, _JUDGE_QUEUE, run_id)


RUNNERS: dict[str, Callable[[str, dict[str, Any]], None]] = {
    "A": run_condition_a,
    "B": run_condition_b,
    "C": run_condition_c,
    "D": run_condition_d,
}


def start_run(run_id: str, condition: str, params: dict[str, Any]) -> None:
    condition = condition.upper()
    runner = RUNNERS.get(condition)
    if runner is None:
        run_registry.update_run(run_id, status="failed", error=f"Unknown condition '{condition}'")
        return
    # Apply Settings-page values (API keys, Neo4j credentials) to the process
    # env right before running, so a saved Settings key actually takes
    # effect for dashboard-triggered runs instead of only ever reading
    # whatever the repo-root .env happened to have at backend startup.
    settings_service.apply_to_environment()

    # All of A/B/C/D are HEAVY_CONDITIONS now (see module docstring
    # "RESOURCE BUDGET") and run ONE AT A TIME. Each is called from its own
    # background thread (routers/runs.py), so with "Run All Conditions"
    # firing A, B, C, D nearly simultaneously, whichever thread acquires
    # _HEAVY_RUN_LOCK first runs to completion while the others block here
    # -- genuinely queued, not just visually queued. The "queued" status is
    # set BEFORE blocking on the lock so a still-waiting run is visibly
    # distinct from "pending" (not yet picked up by any thread at all) in
    # the UI. `runner is None` above already rejects any condition outside
    # RUNNERS/HEAVY_CONDITIONS, so every reachable condition takes this path.
    run_registry.update_run(run_id, status="queued")
    _enter_queue(_HEAVY_QUEUE_LOCK, _HEAVY_QUEUE, run_id)
    try:
        with _HEAVY_RUN_LOCK:
            _leave_queue(_HEAVY_QUEUE_LOCK, _HEAVY_QUEUE, run_id)
            if run_registry.is_cancelled(run_id):
                run_registry.update_run(
                    run_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            runner(run_id, params)
    finally:
        _leave_queue(_HEAVY_QUEUE_LOCK, _HEAVY_QUEUE, run_id)
