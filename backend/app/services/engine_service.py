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


def compute_run_config_hash(condition: str, params: dict[str, Any]) -> str:
    """The config_hash a run with these dashboard params WOULD get, without
    actually launching it -- used by the Factorial Batch page's "already
    completed" skip-detection (POST /api/runs/check-completed) so it can
    compare against existing run_history.jsonl entries' config_hash using
    the EXACT SAME formula run_condition_a/b/c/d use, rather than a
    separate, driftable field-by-field comparison (the pre-config_hash
    skip-detection bug class: a client-side check that doesn't fully
    mirror the resume gate can both false-positive AND false-negative).
    Single-question mode isn't meaningfully "resumable" (each single run
    gets a unique run_id-suffixed filename already), so this only applies
    to batch mode -- callers should treat single-mode params as never
    already-completed.
    """
    condition = condition.upper()
    oversample_pool = resolve_oversample_pool(params)
    n_sample = int(params.get("n_sample") or 1)
    seed = int(params.get("seed") or 42)
    provider = params["provider"]
    model = params["model"]

    if condition == "A":
        cond = _condition_a_module()
        config = cond.build_config(provider, model, n_sample, seed, oversample_pool)
    elif condition == "B":
        cond = _condition_b_module()
        top_k = int(params.get("top_k") or 5)
        require_citation = bool(params.get("require_citation", True))
        require_grounding = bool(params.get("require_grounding", False))
        config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k,
                                    require_citation, require_grounding)
    elif condition == "C":
        cond = _condition_c_module()
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
        config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k, n_anchor,
                                    n_semantic_expansion, fusion_mode, fusion_w_path_trust, fusion_w_intrinsic,
                                    semantic_expansion_trust_cap, require_grounding, enable_semantic_expansion)
    elif condition == "D":
        cond = _condition_d_module()
        top_k = int(params.get("top_k") or 5)
        n_low_level = int(params.get("n_low_level") or 3)
        n_high_level = int(params.get("n_high_level") or 3)
        require_grounding = params.get("require_grounding")
        require_grounding = True if require_grounding is None else bool(require_grounding)
        config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k,
                                    n_low_level, n_high_level, require_grounding)
    else:
        raise ValueError(f"Unknown condition '{condition}'")

    return _compute_config_hash(config)


def is_run_already_completed(condition: str, params: dict[str, Any]) -> tuple[bool, str]:
    """(already_completed, config_hash) -- `already_completed` is True iff
    some run_history.jsonl entry for this condition has a MATCHING
    config_hash AND status "success". An older entry with no config_hash
    at all (pre-this-fix) never matches -- it genuinely isn't verifiable
    as the SAME config, so it must not count as "already done" (matches
    read_already_done()'s resume-gate semantics, see llm/manifest.py)."""
    from app.services import history_service

    config_hash = compute_run_config_hash(condition, params)
    records = history_service._load_condition_history(condition.upper())
    completed = any(
        r.get("config_hash") == config_hash and r.get("status") == "success"
        for r in records
    )
    return completed, config_hash


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
def _judge_v1_module():
    """llm_judge_hallucination_v1.py -- a SEPARATE module from the legacy
    judge's _judge_module() above, never touched/depended on by it. Its
    own top-level `sys.path.insert(0, str(Path(__file__).resolve()
    .parent))` self-heals the bare sibling imports (_judge_common,
    _run_metadata, judge_prompt_v1, judge_clients) it needs, the same way
    importing it directly as a script would. append_manifest() (imported
    from _judge_common) reads THAT module's own LOG_DIR global at call
    time -- same as every other condition module's LOG_DIR override
    pattern, this must be set to an absolute path or it'd try to create
    a bare "logs/" relative to the backend process's cwd."""
    mod = importlib.import_module("llm.evaluation.llm_judge_hallucination_v1")
    judge_common = importlib.import_module("_judge_common")
    judge_common.LOG_DIR = REPO_ROOT / "llm" / "evaluation" / "logs"
    judge_common.RESULTS_DIR = REPO_ROOT / "llm" / "evaluation" / "results"
    return mod


@lru_cache
def _judge_v2_module():
    """llm_judge_hallucination_v2.py -- a SEPARATE module from
    _judge_v1_module() above, never touched/depended on by it (judge-v1
    stays frozen for reproducibility). Same self-healing sys.path/
    LOG_DIR/RESULTS_DIR override pattern."""
    mod = importlib.import_module("llm.evaluation.llm_judge_hallucination_v2")
    judge_common = importlib.import_module("_judge_common")
    judge_common.LOG_DIR = REPO_ROOT / "llm" / "evaluation" / "logs"
    judge_common.RESULTS_DIR = REPO_ROOT / "llm" / "evaluation" / "results"
    return mod


@lru_cache
def _context_relevance_v1_module():
    """llm_judge_context_relevance_v1.py -- a SEPARATE module from BOTH
    the legacy judge AND judge-v1, and from the pre-existing
    llm_judge_context_relevance.py (no suffix, untouched by this dashboard
    integration). Same self-healing sys.path / LOG_DIR/RESULTS_DIR
    override pattern as _judge_v1_module() above."""
    mod = importlib.import_module("llm.evaluation.llm_judge_context_relevance_v1")
    judge_common = importlib.import_module("_judge_common")
    judge_common.LOG_DIR = REPO_ROOT / "llm" / "evaluation" / "logs"
    judge_common.RESULTS_DIR = REPO_ROOT / "llm" / "evaluation" / "results"
    return mod


@lru_cache
def _judge_agreement_module():
    return importlib.import_module("llm.evaluation.judge_agreement")


@lru_cache
def _judge_clients_module():
    return importlib.import_module("llm.evaluation.judge_clients")


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


# _read_already_done is llm.manifest.read_already_done, used directly at
# every call site below (prompt_version-gated -- see that function's
# docstring for why: a stale output file from an OLDER prompt_version must
# never be silently treated as "this run already finished").
from llm.citations import compute_citation_report as _compute_citation_report  # noqa: E402
from llm.manifest import compute_config_hash as _compute_config_hash  # noqa: E402
from llm.manifest import read_already_done as _read_already_done  # noqa: E402
from llm.manifest import sum_usage_tokens as _sum_usage_tokens  # noqa: E402
from llm.manifest import write_manifest as _write_manifest  # noqa: E402


def _manifest_status(cancelled: bool, results: list) -> str:
    """Same 3-value vocabulary as the CLI scripts' write_manifest() calls
    (llm/manifest.py) -- "completed" only when generation actually ran to
    completion with >=1 new record; a cancelled run is "interrupted"; an
    empty `results` list (dashboard's "no_results" status) is "failed",
    since nothing usable came out of it."""
    if cancelled:
        return "interrupted"
    return "completed" if results else "failed"


def _run_log_path(condition_folder: str, run_id: str) -> Path:
    """Per-run log file for a dashboard-launched run -- previously this
    content only ever went to the server's console (log=print), so a run
    launched from the UI had no durable record of what happened during
    generation, unlike a CLI run (which always gets a timestamped .log
    file via setup_logging()). Stored at logs/dashboard_<run_id>.log next
    to that condition's run_history.jsonl, and referenced by path in the
    manifest (write_manifest(..., log_path=...))."""
    return REPO_ROOT / "llm" / condition_folder / "logs" / f"dashboard_{run_id}.log"


def _make_run_logger(log_path: Path) -> Callable[[Any], None]:
    """Returns a log(msg) function that both prints (console, same as
    before) AND appends to `log_path` -- passed anywhere this module
    previously passed bare `print`/`cond.log`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: Any = "") -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    return _log


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

    run_log_path = _run_log_path("a_pure_llm", run_id)
    log = _make_run_logger(run_log_path)
    original_cond_log = cond.log
    cond.log = log

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
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=log, with_meta=True)

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), cond)
            output_path = Path("results") / f"condition_a_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
            config = cond.build_config(provider, model, 1, 42, None)
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
            config = cond.build_config(provider, model, n_sample, seed, oversample_pool)
            output_path = cond.build_output_path("results", provider, model, n_sample, seed,
                                                  oversample_pool=oversample_pool)

        config_hash = _compute_config_hash(config)
        output_path = _anchor_output_path(output_path, "a_pure_llm")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path, cond.PROMPT_VERSION, config_hash)

        run_registry.update_run(
            run_id,
            output_path=str(output_path),
            progress={"current": 0, "total": len(sample_df)},
        )

        embed_model = _embed_model()

        results = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, model, output_path, already_done,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
            config=config, config_hash=config_hash,
        )

        cancelled = run_registry.is_cancelled(run_id)
        summary = _summarize(results)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        cond.append_run_history({
            "batch_id": params.get("batch_id"), "batch_launched_at": params.get("batch_launched_at"),
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
            "config_hash": config_hash,
            # Condition A never retrieves anything, so log_full_candidates has
            # no effect here -- recorded anyway (honestly reflecting whatever
            # was submitted) purely so History/Compare shows the SAME field
            # across all four conditions rather than omitting it for A alone.
            "log_full_candidates": bool(params.get("log_full_candidates")),
            "output_path": str(output_path),
            "log_path": str(run_log_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        _write_manifest(
            output_path, run_label="A", status=_manifest_status(cancelled, results),
            config=config, config_hash=config_hash, log_path=str(run_log_path),
            started_at_utc=run_started_at.isoformat(), finished_at_utc=datetime.now(timezone.utc).isoformat(),
            item_counts={"attempted": params.get("n_sample", 1), "succeeded": len(results),
                         "failed": params.get("n_sample", 1) - len(results)},
            prompt_version=cond.PROMPT_VERSION, total_tokens=_sum_usage_tokens(results),
        )
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
    finally:
        cond.log = original_cond_log


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

    # Computed up front (not just near the process_sample() call below) so
    # build_output_path() can fold grounding into the filename -- without
    # this, B-plain and B-grounded (both require_citation=True, only
    # require_grounding differs) compute the IDENTICAL filename and the
    # second one silently sees the first's output as "already done" (half
    # of the 2026-10-05 pilot no-op bug; the other half, a stale file from
    # an OLDER prompt_version, is fixed by the PROMPT_VERSION gate on
    # _read_already_done() below).
    require_citation = bool(params.get("require_citation", True))
    require_grounding = bool(params.get("require_grounding", False))

    run_log_path = _run_log_path("b_rag", run_id)
    log = _make_run_logger(run_log_path)
    original_cond_log = cond.log
    cond.log = log

    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=log, with_meta=True)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            output_path = Path("results") / f"condition_b_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
            seed = 42
            config = cond.build_config(provider, model, 1, seed, None, top_k, require_citation, require_grounding)
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
            config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k,
                                        require_citation, require_grounding)
            output_path = cond.build_output_path(
                "results", provider, model, n_sample, seed,
                require_citation=require_citation, require_grounding=require_grounding,
                oversample_pool=oversample_pool, top_k=top_k,
            )

        config_hash = _compute_config_hash(config)
        output_path = _anchor_output_path(output_path, "b_rag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path, cond.PROMPT_VERSION, config_hash)

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

        results = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, index, meta_df, top_k, model, output_path, already_done,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
            require_citation=require_citation, require_grounding=require_grounding,
            log_full_candidates=log_full_candidates,
            config=config, config_hash=config_hash,
        )

        cancelled = run_registry.is_cancelled(run_id)
        summary = _summarize(results)
        if require_citation and results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
            # 3-way citation split (NoCit%/InvOnly%/Fabric%/Precis in History
            # Compare) -- the CLI's main() always computed this via
            # compute_citation_report(), but this dashboard entry point
            # never did, so every dashboard-launched run's run_history entry
            # was missing these 4 fields (History Compare showed "—" for
            # all of them). get_history_detail() ALSO now recomputes this
            # on read straight from the output file as a fallback, so
            # already-written entries (like the 2026-10-05 pilot re-run)
            # show correctly without needing this fix or a re-run.
            citation_report = _compute_citation_report(results)
            summary["pct_citation_valid"] = citation_report["pct_valid"]
            summary["pct_citation_no_citation"] = citation_report["pct_no_citation"]
            summary["pct_citation_invalid_only"] = citation_report["pct_invalid_only"]
            summary["fabricated_citation_rate"] = citation_report["fabricated_citation_rate"]
            summary["citation_precision"] = citation_report["citation_precision"]
        if results:
            latencies = [r["retrieval_latency_sec"] for r in results if r.get("retrieval_latency_sec") is not None]
            if latencies:
                summary["avg_retrieval_latency_sec"] = round(sum(latencies) / len(latencies), 3)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        cond.append_run_history({
            "batch_id": params.get("batch_id"), "batch_launched_at": params.get("batch_launched_at"),
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
            "require_grounding": require_grounding,
            "grounding": (("on" if require_grounding else "off") if require_citation else None),
            "config_hash": config_hash,
            "log_full_candidates": log_full_candidates,
            "index_pool": index_pool,
            "top_k": top_k,
            "output_path": str(output_path),
            "log_path": str(run_log_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        run_label = ("B-grounded" if require_grounding else "B-plain") if require_citation else "B"
        _write_manifest(
            output_path, run_label=run_label, status=_manifest_status(cancelled, results),
            config=config, config_hash=config_hash, log_path=str(run_log_path),
            started_at_utc=run_started_at.isoformat(), finished_at_utc=datetime.now(timezone.utc).isoformat(),
            item_counts={"attempted": params.get("n_sample", 1), "succeeded": len(results),
                         "failed": params.get("n_sample", 1) - len(results)},
            prompt_version=cond.PROMPT_VERSION, total_tokens=_sum_usage_tokens(results),
        )
        run_registry.update_run(
            run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(), summary=summary,
        )
    except Exception as e:
        run_registry.update_run(
            run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )
    finally:
        cond.log = original_cond_log


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
    c_retrieval_version = params.get("c_retrieval_version") or "v2"
    alpha = params.get("alpha")
    sample_split = params.get("sample_split") or "test"
    max_hops = int(params.get("max_hops") or 2)
    edge_types = params.get("edge_types") or None
    use_author_trust = bool(params.get("use_author_trust"))
    accepted_only = bool(params.get("accepted_only"))
    token_chunk_limit = 400
    oversample_pool: int | None = None
    n_candidates_after_token_filter: int | None = None

    driver = None
    run_log_path = _run_log_path("c_graphrag", run_id)
    log = _make_run_logger(run_log_path)
    original_cond_log = cond.log
    cond.log = log
    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=log, with_meta=True)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            output_path = Path("results") / f"condition_c_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
            config = cond.build_config(provider, model, 1, 42, None, top_k, n_anchor, n_semantic_expansion,
                                        fusion_mode, fusion_w_path_trust, fusion_w_intrinsic,
                                        semantic_expansion_trust_cap, require_grounding, enable_semantic_expansion,
                                        c_retrieval_version=c_retrieval_version, alpha=alpha,
                                        sample_split=sample_split, max_hops=max_hops, edge_types=edge_types,
                                        use_author_trust=use_author_trust, accepted_only=accepted_only)
        else:
            n_sample = int(params["n_sample"])
            seed = int(params["seed"])
            oversample_pool = resolve_oversample_pool(params)
            candidates = cond.get_candidate_questions(
                con, _questions_parquet(), _answers_parquet(), oversample_pool, seed
            )
            candidates = cond.filter_by_token_limit(candidates)
            n_candidates_after_token_filter = len(candidates)
            if sample_split == "dev":
                sample_df = cond.sample_questions_split(candidates, n_sample, seed, split="dev")
            else:
                sample_df = cond.sample_questions(candidates, n_sample, seed)
            accepted_ids = sample_df["AcceptedAnswerId"].dropna().unique().tolist()
            answers_df = cond.get_accepted_answers(con, _answers_parquet(), accepted_ids)
            sample_df = sample_df.merge(answers_df, on="AcceptedAnswerId", how="left")
            sample_df = sample_df.dropna(subset=["AcceptedAnswerBody"]).reset_index(drop=True)
            config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k, n_anchor,
                                        n_semantic_expansion, fusion_mode, fusion_w_path_trust, fusion_w_intrinsic,
                                        semantic_expansion_trust_cap, require_grounding, enable_semantic_expansion,
                                        c_retrieval_version=c_retrieval_version, alpha=alpha,
                                        sample_split=sample_split, max_hops=max_hops, edge_types=edge_types,
                                        use_author_trust=use_author_trust, accepted_only=accepted_only)
            output_path = cond.build_output_path(
                "results", provider, model, n_sample, seed,
                fusion_mode=fusion_mode, fusion_w_path_trust=fusion_w_path_trust, fusion_w_intrinsic=fusion_w_intrinsic,
                require_grounding=require_grounding, enable_semantic_expansion=enable_semantic_expansion,
                oversample_pool=oversample_pool, top_k=top_k, n_anchor=n_anchor,
                n_semantic_expansion=n_semantic_expansion,
                semantic_expansion_trust_cap=semantic_expansion_trust_cap,
                c_retrieval_version=c_retrieval_version, alpha=alpha, sample_split=sample_split,
                max_hops=max_hops, edge_types=edge_types, use_author_trust=use_author_trust,
                accepted_only=accepted_only,
            )

        config_hash = _compute_config_hash(config)
        output_path = _anchor_output_path(output_path, "c_graphrag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path, cond.PROMPT_VERSION, config_hash)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        all_answer_ids_map = cond.get_all_answer_ids_for_questions(con, _answers_parquet(), eval_question_ids)

        driver, database = cond.connect_neo4j(log)
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
            config=config, config_hash=config_hash,
            c_retrieval_version=c_retrieval_version, alpha=alpha, max_hops=max_hops, edge_types=edge_types,
            use_author_trust=use_author_trust, accepted_only=accepted_only,
        )

        cancelled = run_registry.is_cancelled(run_id) or stats["interrupted"]
        summary = _summarize(results)
        if results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            latencies = [r["retrieval_latency_sec"] for r in results if r.get("retrieval_latency_sec") is not None]
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
            citation_report = _compute_citation_report(results)
            summary["pct_citation_valid"] = citation_report["pct_valid"]
            summary["pct_citation_no_citation"] = citation_report["pct_no_citation"]
            summary["pct_citation_invalid_only"] = citation_report["pct_invalid_only"]
            summary["fabricated_citation_rate"] = citation_report["fabricated_citation_rate"]
            summary["citation_precision"] = citation_report["citation_precision"]
            if latencies:
                summary["avg_retrieval_latency_sec"] = round(sum(latencies) / len(latencies), 3)

        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        summary["require_grounding"] = require_grounding
        summary["enable_semantic_expansion"] = enable_semantic_expansion
        cond.append_run_history({
            "batch_id": params.get("batch_id"), "batch_launched_at": params.get("batch_launched_at"),
            "prompt_version": cond.PROMPT_VERSION,
            "c_retrieval_version": c_retrieval_version,
            "alpha": alpha,
            "sample_split": sample_split,
            "max_hops": max_hops,
            "edge_types": list(edge_types) if edge_types else None,
            "use_author_trust": use_author_trust,
            "accepted_only": accepted_only,
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
            "grounding": "on" if require_grounding else "off",
            "enable_semantic_expansion": enable_semantic_expansion,
            "config_hash": config_hash,
            "log_full_candidates": log_full_candidates,
            "output_path": str(output_path),
            "log_path": str(run_log_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        from llm.evaluation._run_metadata import derive_run_label
        run_label = derive_run_label("C", {
            "fusion_mode": fusion_mode, "grounding": "on" if require_grounding else "off",
        })["run_label"]
        _write_manifest(
            output_path, run_label=run_label, status=_manifest_status(cancelled, results),
            config=config, config_hash=config_hash, log_path=str(run_log_path),
            started_at_utc=run_started_at.isoformat(), finished_at_utc=datetime.now(timezone.utc).isoformat(),
            item_counts={"attempted": params.get("n_sample", 1), "succeeded": len(results),
                         "failed": params.get("n_sample", 1) - len(results)},
            prompt_version=cond.PROMPT_VERSION, total_tokens=_sum_usage_tokens(results),
        )
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
        cond.log = original_cond_log


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
    run_log_path = _run_log_path("d_lightrag", run_id)
    log = _make_run_logger(run_log_path)
    original_cond_log = cond.log
    cond.log = log
    try:
        llm_client, call_llm_fn = cond.get_llm_client(provider, model, log=log, with_meta=True)
        con = _duckdb_connect()

        if params["mode"] == "single":
            sample_df = _build_single_question_df(int(params["question_id"]), _condition_a_module())
            output_path = Path("results") / f"condition_d_{provider}_{_safe_model_name(model)}_single_{params['question_id']}_{run_id}.jsonl"
            config = cond.build_config(provider, model, 1, 42, None, top_k, n_low_level, n_high_level,
                                        require_grounding)
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
            config = cond.build_config(provider, model, n_sample, seed, oversample_pool, top_k,
                                        n_low_level, n_high_level, require_grounding)
            output_path = cond.build_output_path(
                "results", provider, model, n_sample, seed, require_grounding=require_grounding,
                oversample_pool=oversample_pool, top_k=top_k, n_low_level=n_low_level, n_high_level=n_high_level,
            )

        config_hash = _compute_config_hash(config)
        output_path = _anchor_output_path(output_path, "d_lightrag")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        already_done = _read_already_done(output_path, cond.PROMPT_VERSION, config_hash)

        run_registry.update_run(run_id, output_path=str(output_path), progress={"current": 0, "total": len(sample_df)})

        eval_question_ids = sample_df["Id"].astype(int).tolist()
        all_answer_ids_map = cond.get_all_answer_ids_for_questions(con, _answers_parquet(), eval_question_ids)

        driver, database = cond.connect_neo4j(log)
        kg_workspace_dir = REPO_ROOT / "01_data_cleaning" / "_kg_workspace"
        faiss_index, faiss_ids, faiss_embeddings, id_to_row = _faiss_cache(str(kg_workspace_dir))
        embed_model = _embed_model_cpu()

        results, stats = cond.process_sample(
            sample_df, llm_client, call_llm_fn, embed_model, driver, database,
            faiss_index, faiss_ids, faiss_embeddings, id_to_row, all_answer_ids_map,
            top_k, n_low_level, n_high_level, token_chunk_limit, model, output_path, already_done,
            require_grounding=require_grounding,
            on_progress=_make_on_progress(run_id), check_cancel=lambda: run_registry.is_cancelled(run_id),
            config=config, config_hash=config_hash,
        )

        cancelled = run_registry.is_cancelled(run_id) or stats["interrupted"]
        summary = _summarize(results)
        if results:
            n_citation = sum(1 for r in results if r.get("has_citation"))
            n_valid_citation = sum(1 for r in results if r.get("has_valid_citation"))
            latencies = [r["retrieval_latency_sec"] for r in results if r.get("retrieval_latency_sec") is not None]
            summary["pct_with_citation"] = round(n_citation / len(results) * 100, 1)
            summary["pct_with_valid_citation"] = round(n_valid_citation / len(results) * 100, 1)
            citation_report = _compute_citation_report(results)
            summary["pct_citation_valid"] = citation_report["pct_valid"]
            summary["pct_citation_no_citation"] = citation_report["pct_no_citation"]
            summary["pct_citation_invalid_only"] = citation_report["pct_invalid_only"]
            summary["fabricated_citation_rate"] = citation_report["fabricated_citation_rate"]
            summary["citation_precision"] = citation_report["citation_precision"]
            if latencies:
                summary["avg_retrieval_latency_sec"] = round(sum(latencies) / len(latencies), 3)

        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        summary["duration_sec"] = duration
        summary["require_grounding"] = require_grounding
        cond.append_run_history({
            "batch_id": params.get("batch_id"), "batch_launched_at": params.get("batch_launched_at"),
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
            "grounding": "on" if require_grounding else "off",
            "config_hash": config_hash,
            # Condition D's process_sample() doesn't accept/use
            # log_full_candidates at all (only B/C do) -- recorded anyway
            # (honestly reflecting whatever was submitted) for the same
            # cross-condition-consistency reason as Condition A above.
            "log_full_candidates": bool(params.get("log_full_candidates")),
            "output_path": str(output_path),
            "log_path": str(run_log_path),
            "duration_sec": duration,
            "source": "dashboard",
            **summary,
        })
        run_label = "D-grounded" if require_grounding else "D-plain"
        _write_manifest(
            output_path, run_label=run_label, status=_manifest_status(cancelled, results),
            config=config, config_hash=config_hash, log_path=str(run_log_path),
            started_at_utc=run_started_at.isoformat(), finished_at_utc=datetime.now(timezone.utc).isoformat(),
            item_counts={"attempted": params.get("n_sample", 1), "succeeded": len(results),
                         "failed": params.get("n_sample", 1) - len(results)},
            prompt_version=cond.PROMPT_VERSION, total_tokens=_sum_usage_tokens(results),
        )
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
        cond.log = original_cond_log


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


# ---------------------------------------------------------------------
# Judge-v1 (hallucination, reference-based) -- Phase 2 dashboard UI.
# Fully separate from the legacy judge's _JUDGE_SEMAPHORE/_JUDGE_QUEUE
# above -- judge-v1 jobs may run WHILE a generation run OR a legacy
# judge run is executing (API-only, not memory/DuckDB-heavy), but only
# ONE judge-v1 job at a time (per docs/agent_prompt_phase2_judge_ui.md
# Step 1.4) -- hence Semaphore(1), not a concurrency limit >1.
# ---------------------------------------------------------------------
_JUDGE_V1_SEMAPHORE = threading.Semaphore(1)
_JUDGE_V1_QUEUE_LOCK = threading.Lock()
_JUDGE_V1_QUEUE: list[str] = []

# Context-relevance-v1 -- its own semaphore/queue, independent of
# judge-v1's and the legacy judge's, same "only ONE job of this kind at
# a time" rule as judge-v1.
_CTXREL_V1_SEMAPHORE = threading.Semaphore(1)
_CTXREL_V1_QUEUE_LOCK = threading.Lock()
_CTXREL_V1_QUEUE: list[str] = []

# One canonical output file PER judge_id, shared across every dashboard-
# launched judge-v1 job regardless of which runs were selected -- makes
# resume (llm_judge_hallucination_v1.load_already_done) correct no matter
# how run selections vary between launches, since it's keyed on
# (run_id, question_id, judge_id, prompt_version), not on "which file".
def _judge_output_paths_generic(mod, filename_prefix: str, judge_id: str) -> tuple[Path, Path, Path]:
    """Shared by judge_v1_output_paths()/judge_v2_output_paths() -- the
    canonical dashboard output file for this judge_id ALSO encodes the
    judge prompt version (e.g. "judge-v1"/"judge-v2") in its filename,
    so a run under one version can never silently append into the
    other's file. Readers additionally refuse a file containing more
    than one prompt_version/blinding_version (see judge_agreement.
    load_judge_records()) as defense in depth for a manually-overridden
    --out path."""
    prompt_version = mod.PROMPT_VERSION
    base = REPO_ROOT / "llm" / "evaluation" / "results" / f"{filename_prefix}_{judge_id}_{prompt_version}.jsonl"
    failures = base.with_name(f"{filename_prefix}_{judge_id}_{prompt_version}_failures.jsonl")
    resolved = base.with_name(f"{filename_prefix}_{judge_id}_{prompt_version}_failures_resolved.jsonl")
    return base, failures, resolved


def judge_v1_output_paths(judge_id: str) -> tuple[Path, Path, Path]:
    return _judge_output_paths_generic(_judge_v1_module(), "jv1_dashboard", judge_id)


def judge_v1_run_log_path(job_run_id: str) -> Path:
    return REPO_ROOT / "llm" / "evaluation" / "logs" / f"dashboard_judge_v1_{job_run_id}.log"


def _judge_refuse_reason(run_id: str) -> str | None:
    """None if the run is judgeable; otherwise a human-readable reason to
    show in the plan/launch response. Checked BEFORE calling load_items()
    so a mixed-config-hash file produces a clean per-run plan-row error
    instead of crashing the whole plan/launch call. Version-independent
    -- shared by judge-v1 and judge-v2 (neither the invalid/superseded
    checks nor the mixed-config-hash guard depend on prompt_version).
    _judge_v1_refuse_reason is kept as an alias so existing call sites/
    tests referencing that name are unaffected."""
    from app.services import history_service, invalid_runs_service, superseded_runs_service
    from llm.manifest import MixedConfigHashError, assert_single_config_hash

    if run_id in invalid_runs_service.load_invalid_run_ids():
        return "run is marked invalid (logs/invalid_runs.jsonl) -- refused"
    if run_id in superseded_runs_service.load_superseded_run_ids():
        return "run is marked superseded (logs/superseded_runs.jsonl) -- refused"
    try:
        detail = history_service.get_history_detail(run_id)
    except MixedConfigHashError as e:
        # get_history_detail() itself already refuses to summarize a
        # mixed-config-hash output file (raises, doesn't return) -- catch
        # it here too so THIS caller gets a clean per-run plan/launch
        # row instead of a 500.
        return str(e)
    if not detail or not detail.get("output_path"):
        return "run not found or has no output file"
    output_path = Path(detail["output_path"])
    if not output_path.exists():
        return f"output file missing: {output_path}"
    try:
        records = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
        assert_single_config_hash(records, str(output_path))
    except MixedConfigHashError as e:
        return str(e)
    return None


_judge_v1_refuse_reason = _judge_refuse_reason


def _token_latency_stats_from_glob(
    glob_pattern: str, judge_model: str, exclude_basenames: set[str] | None = None,
) -> tuple[float | None, float | None, float | None, str | None]:
    """(avg_input_tokens, avg_output_tokens, avg_latency_sec, source) for
    this judge_model, scanning every llm/evaluation/results/<glob_pattern>
    file -- a BOUNDED, estimation-only glob (not an authoritative data
    read), shared by plan_judge_v1()/plan_judge_v2()/
    plan_context_relevance_v1() (each passes its own pattern, e.g.
    "jv1_*.jsonl"/"jv2_*.jsonl"/"ctxrel_v1_*.jsonl", so the three never
    cross-contaminate each other's token estimates). `exclude_basenames`
    skips files registered as known-bad (e.g. a smoke-test --out path)
    -- see invalid_context_relevance_outputs_service.py.

    Fallback chain used by every caller: this real-history scan first,
    then (if it finds nothing) the registry's own default_input_tokens_
    per_item/default_output_tokens_per_item/default_latency_sec_per_item,
    with source="default estimate" in that case.
    """
    exclude_basenames = exclude_basenames or set()
    results_dir = REPO_ROOT / "llm" / "evaluation" / "results"
    input_vals: list[float] = []
    output_vals: list[float] = []
    latency_vals: list[float] = []
    if results_dir.exists():
        for path in sorted(results_dir.glob(glob_pattern)):
            if path.name in exclude_basenames:
                continue
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = _json.loads(line)
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
        avg_in = sum(input_vals) / len(input_vals)
        avg_out = sum(output_vals) / len(output_vals)
        avg_lat = sum(latency_vals) / len(latency_vals) if latency_vals else None
        return avg_in, avg_out, avg_lat, "from history"
    return None, None, None, None


def _judge_token_latency_stats(judge_model: str) -> tuple[float | None, float | None, float | None, str | None]:
    """plan_judge_v1()'s token/cost/time estimate -- see
    _token_latency_stats_from_glob()'s docstring for the fallback chain."""
    return _token_latency_stats_from_glob("jv1_*.jsonl", judge_model)


def _plan_judge_generic(
    run_ids: list[str], judge_ids: list[str], workers: dict[str, int], mod, output_paths_fn, token_stats_fn,
) -> dict[str, Any]:
    """Shared by plan_judge_v1()/plan_judge_v2() -- dry-run: per (run,
    judge) -- items to judge (resume-aware), est. tokens/cost/time.
    Refused runs (invalid/superseded/mixed-hash/missing) get a
    `refused_reason` and no per-judge rows. `mod` is the judge module
    (llm_judge_hallucination_v1/v2), `output_paths_fn`/`token_stats_fn`
    are that version's own output-path/token-stats functions -- the
    only two things that actually differ between versions."""
    from app.services import history_service

    judge_registry = _judge_clients_module().JUDGE_REGISTRY
    rows: list[dict[str, Any]] = []
    # Cache per judge_id -- the fallback scan is the same for every run in
    # this plan() call, no need to re-glob results/ once per row.
    stats_cache: dict[str, tuple[float | None, float | None, float | None, str | None]] = {}

    for run_id in run_ids:
        reason = _judge_refuse_reason(run_id)
        if reason:
            rows.append({"run_id": run_id, "refused_reason": reason})
            continue

        detail = history_service.get_history_detail(run_id)
        output_path = Path(detail["output_path"])
        with open(output_path) as f:
            records = [_json.loads(line) for line in f if line.strip()]
        n_items = len(records)

        for judge_id in judge_ids:
            out_path, _, _ = output_paths_fn(judge_id)
            done_keys = mod.load_already_done(out_path) if out_path.exists() else set()
            config = judge_registry.get(judge_id)
            judge_model = config.model if config else None
            # parse_version/max_tokens (2026-10-07 follow-up) -- only
            # judge-v2's module exposes these (judge-v1 has neither
            # concept), via getattr so this stays generic across both
            # `mod` modules; must match _judge_common.judge_resume_key()'s
            # 9-field shape exactly or every record looks "not done".
            already_judged = sum(
                1 for r in records
                if (run_id, r.get("config_hash"), r.get("question_id"), judge_id,
                    judge_model, mod.PROMPT_VERSION, mod.BLINDING_VERSION,
                    getattr(mod, "PARSE_VERSION", None), getattr(mod, "REQUEST_MAX_TOKENS", None)) in done_keys
            )
            to_judge = n_items - already_judged

            # Token/latency estimate -- real history (any output file
            # for this judge_model, THIS version only) first, the
            # registry's own defaults otherwise. `token_source` is
            # surfaced in the row so the UI can label which one was
            # used ("from history" / "default estimate") -- never
            # silently guessed without saying so.
            avg_in = avg_out = avg_lat = None
            token_source = None
            if config is not None:
                if judge_model not in stats_cache:
                    stats_cache[judge_model] = token_stats_fn(judge_model)
                avg_in, avg_out, avg_lat, token_source = stats_cache[judge_model]
                if avg_in is None:
                    avg_in, avg_out = config.default_input_tokens_per_item, config.default_output_tokens_per_item
                    token_source = "default estimate"
                if avg_lat is None:
                    avg_lat = config.default_latency_sec_per_item

            est_tokens_in = round(avg_in * to_judge) if avg_in is not None else None
            est_tokens_out = round(avg_out * to_judge) if avg_out is not None else None
            # "Unknown" (never $0) only for a judge_id missing from the
            # registry entirely -- with the default-estimate fallback
            # above, a REGISTERED judge_id always has SOME token estimate,
            # so cost is only ever genuinely $0 when to_judge is 0 (nothing
            # left to judge), never a silent stand-in for "we don't know".
            est_cost = None
            if config is not None and est_tokens_in is not None and est_tokens_out is not None:
                est_cost = round(
                    est_tokens_in / 1_000_000 * config.price_per_m_input
                    + est_tokens_out / 1_000_000 * config.price_per_m_output,
                    4,
                )

            w = max(1, workers.get(judge_id, 4))
            est_time_sec = round(to_judge / w * avg_lat) if (to_judge and avg_lat is not None) else 0

            rows.append({
                "run_id": run_id, "run_label": detail.get("run_label"), "judge_id": judge_id,
                "role": config.role if config else None,
                "items_total": n_items, "already_judged": already_judged, "items_to_judge": to_judge,
                "est_tokens_in": est_tokens_in, "est_tokens_out": est_tokens_out,
                "est_cost_usd": est_cost, "cost_unknown": config is None,
                "est_time_sec": est_time_sec, "token_source": token_source,
            })

    valid_rows = [r for r in rows if "refused_reason" not in r]
    per_judge: dict[str, dict[str, Any]] = {}
    for judge_id in judge_ids:
        judge_rows = [r for r in valid_rows if r["judge_id"] == judge_id]
        known_cost_rows = [r for r in judge_rows if not r["cost_unknown"]]
        w = max(1, workers.get(judge_id, 4))
        total_items_to_judge = sum(r["items_to_judge"] for r in judge_rows)

        config = judge_registry.get(judge_id)
        avg_lat = config.default_latency_sec_per_item if config else 30.0
        if config is not None:
            _, _, hist_lat, _ = stats_cache.get(config.model, (None, None, None, None))
            if hist_lat is not None:
                avg_lat = hist_lat

        # Sequential per judge (one run after another; workers only
        # parallelize WITHIN a run) -- a judge's total wall-clock is its
        # own items-to-judge summed across every selected run, divided by
        # its worker count, times its average per-item latency. NOT a
        # max() of single-row times, which badly understates a multi-run
        # batch's real wall-clock (the bug this fix addresses).
        total_time_sec = round(total_items_to_judge / w * avg_lat) if total_items_to_judge else 0

        per_judge[judge_id] = {
            "items_to_judge": total_items_to_judge,
            "est_tokens_in": sum(r["est_tokens_in"] or 0 for r in judge_rows) or None,
            "est_tokens_out": sum(r["est_tokens_out"] or 0 for r in judge_rows) or None,
            "est_cost_usd": round(sum(r["est_cost_usd"] or 0 for r in known_cost_rows), 4) if known_cost_rows else None,
            "cost_unknown_rows": sum(1 for r in judge_rows if r["cost_unknown"]),
            "est_time_sec": total_time_sec,
        }

    totals = {
        "items_to_judge": sum(p["items_to_judge"] for p in per_judge.values()),
        "est_tokens_in": sum(p["est_tokens_in"] or 0 for p in per_judge.values()) or None,
        "est_tokens_out": sum(p["est_tokens_out"] or 0 for p in per_judge.values()) or None,
        "est_cost_usd": round(sum(p["est_cost_usd"] or 0 for p in per_judge.values()), 4)
        if any(p["est_cost_usd"] is not None for p in per_judge.values()) else None,
        "cost_unknown_rows": sum(p["cost_unknown_rows"] for p in per_judge.values()),
        # Judges run in PARALLEL (each is its own sequential queue of
        # runs) -- the whole job's wall-clock is the SLOWEST judge, not a
        # sum across judges.
        "est_time_sec": max((p["est_time_sec"] for p in per_judge.values()), default=0),
        "per_judge": per_judge,
    }
    return {"rows": rows, "totals": totals}


def plan_judge_v1(run_ids: list[str], judge_ids: list[str], workers: dict[str, int]) -> dict[str, Any]:
    return _plan_judge_generic(run_ids, judge_ids, workers, _judge_v1_module(), judge_v1_output_paths, _judge_token_latency_stats)


def _build_judge_items_generic(run_ids: list[str], mod, output_paths_fn, limit: int | None = None) -> list[dict[str, Any]]:
    """Shared by _build_judge_v1_items()/_build_judge_v2_items() --
    load_items() over each selected run's own output_path -- re-checks
    the same refusal conditions (defense in depth: a run could become
    invalid/superseded between plan and launch). `limit`: per-file item
    cap, same meaning as load_items()'s own --limit -- used for a cheap
    smoke-test launch (e.g. 2 items) through this SAME path, not a
    separate code path. `output_paths_fn` is unused here but accepted
    for symmetry with the other generics (kept for a future version
    that might need to inspect existing outputs before loading)."""
    from app.services import history_service

    paths = []
    for run_id in run_ids:
        reason = _judge_refuse_reason(run_id)
        if reason:
            raise ValueError(f"{run_id}: {reason}")
        detail = history_service.get_history_detail(run_id)
        paths.append(detail["output_path"])
    return mod.load_items(paths, limit=limit)


def _build_judge_v1_items(run_ids: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    return _build_judge_items_generic(run_ids, _judge_v1_module(), judge_v1_output_paths, limit=limit)


def _run_judge_batch_dashboard_generic(
    job_run_id: str, params: dict[str, Any], mod, output_paths_fn, log_path_fn, manifest_name: str,
    build_items_fn,
) -> None:
    """Shared by run_judge_v1_batch_dashboard()/run_judge_v2_batch_
    dashboard() -- one run_batch() call per judge_id, sequential
    (run_batch() itself already parallelizes across items via its own
    `workers`). Every version-specific bit is passed in: the judge
    module, its output-path/log-path functions, its own manifest
    filename (judge_v1_run_history.jsonl vs judge_v2_run_history.jsonl
    -- never the same file), and its items-builder."""
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(job_run_id, status="running", started_at=run_started_at.isoformat())

    run_ids = params["run_ids"]
    judge_ids = params["judge_ids"]
    workers = params.get("workers") or {}
    log_path = log_path_fn(job_run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: Any = "") -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    try:
        from app.services import history_service

        items = build_items_fn(run_ids, limit=params.get("limit"))
        # Actual output FILE PATHS, not run_ids -- judge_v1_lookup_service's
        # run_label/is_other_config resolution reads these via
        # _run_metadata.resolve_run_metadata(), which needs a path matching
        # a run_history.jsonl output_path, not a dashboard history_id.
        run_file_paths = [history_service.get_history_detail(rid)["output_path"] for rid in run_ids]
        questions_parquet = str(_questions_parquet())
        per_judge_summary: dict[str, Any] = {}
        total_done = 0
        total_target = len(items) * len(judge_ids)
        run_registry.update_run(job_run_id, progress={"current": 0, "total": total_target})

        for judge_id in judge_ids:
            if run_registry.is_cancelled(job_run_id):
                break
            out_path, failures_path, resolved_path = output_paths_fn(judge_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            w = max(1, int(workers.get(judge_id, 4)))

            def on_progress(event: dict[str, Any]) -> None:
                nonlocal total_done
                total_done += 1
                run_registry.update_run(job_run_id, progress={"current": total_done, "total": total_target})
                run_registry.append_result(job_run_id, {
                    "question_id": event.get("question_id"), "judge_id": judge_id,
                    "status": event.get("status"),
                })

            summary = mod.run_batch(
                items, [judge_id], questions_parquet, out_path, failures_path, workers=w,
                resolved_path=resolved_path, on_progress=on_progress,
                check_cancel=lambda: run_registry.is_cancelled(job_run_id),
            )
            per_judge_summary[judge_id] = summary
            log(f"[{judge_id}] judged={summary['judged']} skipped={summary['skipped']} failed={summary['failed']}")

            from llm.manifest import sum_usage_tokens, write_manifest
            finished_at = datetime.now(timezone.utc)
            with open(out_path) as f:
                out_records = [_json.loads(line) for line in f if line.strip()]
            write_manifest(
                out_path, run_label=None, status="interrupted" if run_registry.is_cancelled(job_run_id) else "completed",
                config={"run_ids": run_ids, "judge_id": judge_id, "workers": w},
                started_at_utc=run_started_at.isoformat(), finished_at_utc=finished_at.isoformat(),
                item_counts={"attempted": len(items), "succeeded": summary["judged"], "failed": summary["failed"]},
                prompt_version=mod.PROMPT_VERSION, judge_version=mod.PROMPT_VERSION,
                blinding_version=getattr(mod, "BLINDING_VERSION", None),
                total_tokens=sum_usage_tokens(out_records), extra_output_files=[failures_path],
                log_path=str(log_path),
            )

            mod.append_manifest(manifest_name, {
                "run_started_at": run_started_at.isoformat(),
                "run_files": run_file_paths, "judge": judge_id, "workers": w,
                "out": str(out_path), "prompt_version": mod.PROMPT_VERSION, "source": "dashboard",
                **summary,
            })

        cancelled = run_registry.is_cancelled(job_run_id)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        run_registry.update_run(
            job_run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            summary={"per_judge": per_judge_summary, "duration_sec": duration},
        )
    except Exception as e:
        run_registry.update_run(
            job_run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )


def run_judge_v1_batch_dashboard(job_run_id: str, params: dict[str, Any]) -> None:
    _run_judge_batch_dashboard_generic(
        job_run_id, params, _judge_v1_module(), judge_v1_output_paths, judge_v1_run_log_path,
        "judge_v1_run_history.jsonl", _build_judge_v1_items,
    )


def _start_judge_run_generic(
    job_run_id: str, params: dict[str, Any], semaphore: threading.Semaphore,
    queue_lock: threading.Lock, queue: list[str], dashboard_fn,
) -> None:
    """Shared by start_judge_v1_run()/start_judge_v2_run() -- only ONE
    job of a given version at a time (its own Semaphore(1), never
    gating on _HEAVY_RUN_LOCK or the legacy judge's _JUDGE_SEMAPHORE:
    these jobs are API-only and may run alongside either, and v1/v2
    jobs may run alongside EACH OTHER too since they have separate
    semaphores)."""
    settings_service.apply_to_environment()
    run_registry.update_run(job_run_id, status="queued")
    _enter_queue(queue_lock, queue, job_run_id)
    try:
        semaphore.acquire()
        try:
            _leave_queue(queue_lock, queue, job_run_id)
            if run_registry.is_cancelled(job_run_id):
                run_registry.update_run(
                    job_run_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            dashboard_fn(job_run_id, params)
        finally:
            semaphore.release()
    finally:
        _leave_queue(queue_lock, queue, job_run_id)


def start_judge_v1_run(job_run_id: str, params: dict[str, Any]) -> None:
    _start_judge_run_generic(
        job_run_id, params, _JUDGE_V1_SEMAPHORE, _JUDGE_V1_QUEUE_LOCK, _JUDGE_V1_QUEUE, run_judge_v1_batch_dashboard,
    )


# ---------------------------------------------------------------------
# Judge-v2 -- a fix to judge-v1's prompt (see docs/JUDGE_V2_CHANGES.md).
# judge-v1 is NEVER touched by anything below -- these are v2's OWN
# output paths/manifest/semaphore, reusing the version-independent
# generics above (_plan_judge_generic/_build_judge_items_generic/
# _run_judge_batch_dashboard_generic/_start_judge_run_generic) rather
# than duplicating them.
# ---------------------------------------------------------------------
_JUDGE_V2_SEMAPHORE = threading.Semaphore(1)
_JUDGE_V2_QUEUE_LOCK = threading.Lock()
_JUDGE_V2_QUEUE: list[str] = []


def judge_v2_output_paths(judge_id: str) -> tuple[Path, Path, Path]:
    return _judge_output_paths_generic(_judge_v2_module(), "jv2_dashboard", judge_id)


def judge_v2_run_log_path(job_run_id: str) -> Path:
    return REPO_ROOT / "llm" / "evaluation" / "logs" / f"dashboard_judge_v2_{job_run_id}.log"


def _judge_v2_token_latency_stats(judge_model: str) -> tuple[float | None, float | None, float | None, str | None]:
    """plan_judge_v2()'s token/cost/time estimate -- separate glob
    pattern ("jv2_*.jsonl") from judge-v1's, so the two never cross-
    contaminate each other's estimates."""
    return _token_latency_stats_from_glob("jv2_*.jsonl", judge_model)


def plan_judge_v2(run_ids: list[str], judge_ids: list[str], workers: dict[str, int]) -> dict[str, Any]:
    return _plan_judge_generic(run_ids, judge_ids, workers, _judge_v2_module(), judge_v2_output_paths, _judge_v2_token_latency_stats)


def _build_judge_v2_items(run_ids: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    return _build_judge_items_generic(run_ids, _judge_v2_module(), judge_v2_output_paths, limit=limit)


def run_judge_v2_batch_dashboard(job_run_id: str, params: dict[str, Any]) -> None:
    _run_judge_batch_dashboard_generic(
        job_run_id, params, _judge_v2_module(), judge_v2_output_paths, judge_v2_run_log_path,
        "judge_v2_run_history.jsonl", _build_judge_v2_items,
    )


def start_judge_v2_run(job_run_id: str, params: dict[str, Any]) -> None:
    _start_judge_run_generic(
        job_run_id, params, _JUDGE_V2_SEMAPHORE, _JUDGE_V2_QUEUE_LOCK, _JUDGE_V2_QUEUE, run_judge_v2_batch_dashboard,
    )


def _existing_output_path(output_paths_fn, judge_id: str) -> str | None:
    out_path, _, _ = output_paths_fn(judge_id)
    return str(out_path) if out_path.exists() else None


def build_version_comparison(run_ids: list[str], judge_id_a: str = "primary", judge_id_b: str = "secondary") -> dict[str, Any]:
    """Step 5: a label TRANSITION matrix (v1 label -> v2 label, per
    run_label, for judge_id_a) plus kappa(judge_id_a, judge_id_b) under
    v1 and under v2, SIDE BY SIDE -- never mixed into one read (each
    version's records are loaded with a SEPARATE load_judge_records()
    call, per the mixed-version guard)."""
    agreement_mod = _judge_agreement_module()

    v1_paths = [p for p in (_existing_output_path(judge_v1_output_paths, judge_id_a),
                            _existing_output_path(judge_v1_output_paths, judge_id_b)) if p]
    v2_paths = [p for p in (_existing_output_path(judge_v2_output_paths, judge_id_a),
                            _existing_output_path(judge_v2_output_paths, judge_id_b)) if p]

    v1_records = agreement_mod.load_judge_records(v1_paths) if v1_paths else []
    v2_records = agreement_mod.load_judge_records(v2_paths) if v2_paths else []

    run_id_set = set(run_ids)
    v1_by_key = {(r["run_id"], r["question_id"]): r for r in v1_records
                 if r.get("judge_id") == judge_id_a and r.get("run_id") in run_id_set}
    v2_by_key = {(r["run_id"], r["question_id"]): r for r in v2_records
                 if r.get("judge_id") == judge_id_a and r.get("run_id") in run_id_set}

    transition: dict[str, dict[str, dict[str, int]]] = {}
    for key in set(v1_by_key) & set(v2_by_key):
        v1_rec, v2_rec = v1_by_key[key], v2_by_key[key]
        run_label = v1_rec.get("run_label") or "unknown"
        v1_label = v1_rec.get("label") or "PARSE_ERROR"
        v2_label = v2_rec.get("label") or "PARSE_ERROR"
        transition.setdefault(run_label, {}).setdefault(v1_label, {})
        transition[run_label][v1_label][v2_label] = transition[run_label][v1_label].get(v2_label, 0) + 1

    def _kappas_for(records: list[dict]) -> dict[str, Any]:
        pairs = agreement_mod.pair_two_judges(
            [r for r in records if r.get("run_id") in run_id_set], judge_id_a, judge_id_b,
        )
        if not pairs:
            return {"n_pairs": 0, "kappas": None, "insufficient_data": True}
        kappas = agreement_mod.compute_kappas(pairs)
        return {"n_pairs": len(pairs), "kappas": kappas, "insufficient_data": kappas["weighted"] is None}

    return {
        "judge_id_a": judge_id_a, "judge_id_b": judge_id_b,
        "transition_matrix": transition,
        "kappa_v1": _kappas_for(v1_records),
        "kappa_v2": _kappas_for(v2_records),
    }


def reference_conflicts(run_ids: list[str], judge_ids: list[str]) -> list[dict[str, Any]]:
    """Step 5: a "Reference conflicts" list (question, run_label, claim,
    both judges) for human review -- every v2 claim flagged
    reference_conflict=True (the judge was confident the reference
    answer is outdated/wrong), across the given judge_ids, restricted
    to the given run_ids."""
    run_id_set = set(run_ids)
    rows: list[dict[str, Any]] = []
    for judge_id in judge_ids:
        out_path, _, _ = judge_v2_output_paths(judge_id)
        if not out_path.exists():
            continue
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = _json.loads(line)
                if record.get("run_id") not in run_id_set:
                    continue
                for claim in record.get("claims") or []:
                    if claim.get("reference_conflict"):
                        rows.append({
                            "question_id": record.get("question_id"), "run_label": record.get("run_label"),
                            "judge_id": judge_id, "claim": claim.get("claim"),
                            "evidence": claim.get("evidence"), "verdict": claim.get("verdict"),
                        })
    return rows


# ---------------------------------------------------------------------
# Context relevance v1 -- Step 3 of docs/agent_prompt_judge_results_
# export.md. Dedup is CROSS-RUN (same (question_id, answer_id,
# text_hash) across every selected run's retrieved_context), so this
# produces ONE canonical output file per judge_id, shared across every
# dashboard launch regardless of which runs were selected -- same
# reasoning as judge_v1_output_paths() above.
# ---------------------------------------------------------------------

def ctxrel_v1_output_paths(judge_id: str) -> tuple[Path, Path, Path]:
    prompt_version = _context_relevance_v1_module().PROMPT_VERSION
    base = REPO_ROOT / "llm" / "evaluation" / "results" / f"ctxrel_v1_{judge_id}_{prompt_version}.jsonl"
    failures = base.with_name(f"ctxrel_v1_{judge_id}_{prompt_version}_failures.jsonl")
    resolved = base.with_name(f"ctxrel_v1_{judge_id}_{prompt_version}_failures_resolved.jsonl")
    return base, failures, resolved


def _ctxrel_v1_refuse_reason(run_id: str) -> str | None:
    """Same checks as _judge_v1_refuse_reason(), PLUS: condition A has no
    retrieved_context at all, so it's excluded from context relevance
    entirely (Step 3.6)."""
    from app.services import history_service

    reason = _judge_v1_refuse_reason(run_id)
    if reason:
        return reason
    detail = history_service.get_history_detail(run_id)
    if detail.get("condition") == "A":
        return "condition A has no retrieved_context -- excluded from context relevance"
    return None


def _ctxrel_token_latency_stats(judge_model: str) -> tuple[float | None, float | None, float | None, str | None]:
    """plan_context_relevance_v1()'s token/cost/time estimate. Skips any
    file registered in logs/invalid_context_relevance_outputs.jsonl
    (e.g. a manual smoke-test --out path) -- dedup/resume/results-
    summary never glob at all (they only ever read the canonical per-
    judge_id file), so this is the one place a smoke-test output could
    otherwise quietly bias a real plan-table estimate."""
    from app.services import invalid_context_relevance_outputs_service

    invalid_basenames = invalid_context_relevance_outputs_service.load_invalid_output_basenames()
    return _token_latency_stats_from_glob("ctxrel_v1_*.jsonl", judge_model, exclude_basenames=invalid_basenames)


def _build_ctxrel_v1_items(run_ids: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    """Deduplicated (question, context item) rows across every selected
    run (via llm_judge_context_relevance_v1.load_items()) -- re-checks
    refusal conditions, same defense-in-depth reasoning as
    _build_judge_v1_items()."""
    from app.services import history_service

    mod = _context_relevance_v1_module()
    paths = []
    for run_id in run_ids:
        reason = _ctxrel_v1_refuse_reason(run_id)
        if reason:
            raise ValueError(f"{run_id}: {reason}")
        detail = history_service.get_history_detail(run_id)
        paths.append(detail["output_path"])
    return mod.load_items(paths, limit=limit) if paths else []


def plan_context_relevance_v1(run_ids: list[str], judge_ids: list[str], workers: dict[str, int]) -> dict[str, Any]:
    """Dry-run for context relevance: per-run context-item counts (for
    visibility), plus the GLOBAL deduplicated item count actually judged
    (shared across every selected run, incl. plain/grounded pairs of the
    same retrieval method) -- est. tokens/cost/time are computed on that
    deduplicated count, not on the sum of each run's own context items."""
    from app.services import history_service

    mod = _context_relevance_v1_module()
    judge_registry = _judge_clients_module().JUDGE_REGISTRY

    rows: list[dict[str, Any]] = []
    valid_run_ids: list[str] = []
    for run_id in run_ids:
        reason = _ctxrel_v1_refuse_reason(run_id)
        if reason:
            rows.append({"run_id": run_id, "refused_reason": reason})
            continue
        detail = history_service.get_history_detail(run_id)
        output_path = Path(detail["output_path"])
        with open(output_path) as f:
            records = [_json.loads(line) for line in f if line.strip()]
        n_context_items = sum(len(r.get("retrieved_context") or []) for r in records)
        rows.append({"run_id": run_id, "run_label": detail.get("run_label"), "n_context_items": n_context_items})
        valid_run_ids.append(run_id)

    paths = [history_service.get_history_detail(rid)["output_path"] for rid in valid_run_ids]
    items = mod.load_items(paths, limit=None) if paths else []
    unique_items = len(items)

    stats_cache: dict[str, tuple[float | None, float | None, float | None, str | None]] = {}
    per_judge: dict[str, dict[str, Any]] = {}
    for judge_id in judge_ids:
        out_path, _, _ = ctxrel_v1_output_paths(judge_id)
        done_keys = mod.load_already_done(out_path) if out_path.exists() else set()
        config = judge_registry.get(judge_id)
        judge_model = config.model if config else None
        already_judged = sum(
            1 for it in items
            if (it["question_id"], it["answer_id"], it["text_hash"], judge_id, judge_model, mod.PROMPT_VERSION) in done_keys
        )
        to_judge = unique_items - already_judged

        avg_in = avg_out = avg_lat = None
        token_source = None
        if config is not None:
            if judge_model not in stats_cache:
                stats_cache[judge_model] = _ctxrel_token_latency_stats(judge_model)
            avg_in, avg_out, avg_lat, token_source = stats_cache[judge_model]
            if avg_in is None:
                avg_in, avg_out = config.default_input_tokens_per_item, config.default_output_tokens_per_item
                token_source = "default estimate"
            if avg_lat is None:
                avg_lat = config.default_latency_sec_per_item

        est_tokens_in = round(avg_in * to_judge) if avg_in is not None else None
        est_tokens_out = round(avg_out * to_judge) if avg_out is not None else None
        est_cost = None
        if config is not None and est_tokens_in is not None and est_tokens_out is not None:
            est_cost = round(
                est_tokens_in / 1_000_000 * config.price_per_m_input
                + est_tokens_out / 1_000_000 * config.price_per_m_output, 4,
            )
        w = max(1, workers.get(judge_id, 4))
        est_time_sec = round(to_judge / w * avg_lat) if (to_judge and avg_lat is not None) else 0

        per_judge[judge_id] = {
            "role": config.role if config else None,
            "already_judged": already_judged, "items_to_judge": to_judge,
            "est_tokens_in": est_tokens_in, "est_tokens_out": est_tokens_out,
            "est_cost_usd": est_cost, "cost_unknown": config is None,
            "est_time_sec": est_time_sec, "token_source": token_source,
        }

    totals = {
        "unique_items": unique_items,
        "items_to_judge": sum(p["items_to_judge"] for p in per_judge.values()),
        "est_tokens_in": sum(p["est_tokens_in"] or 0 for p in per_judge.values()) or None,
        "est_tokens_out": sum(p["est_tokens_out"] or 0 for p in per_judge.values()) or None,
        "est_cost_usd": round(sum(p["est_cost_usd"] or 0 for p in per_judge.values()), 4)
        if any(p["est_cost_usd"] is not None for p in per_judge.values()) else None,
        "cost_unknown_rows": sum(1 for p in per_judge.values() if p["cost_unknown"]),
        "est_time_sec": max((p["est_time_sec"] for p in per_judge.values()), default=0),
        "per_judge": per_judge,
    }
    return {"rows": rows, "totals": totals}


def ctxrel_v1_run_log_path(job_run_id: str) -> Path:
    return REPO_ROOT / "llm" / "evaluation" / "logs" / f"dashboard_ctxrel_v1_{job_run_id}.log"


def run_context_relevance_v1_batch_dashboard(job_run_id: str, params: dict[str, Any]) -> None:
    """Mirrors run_judge_v1_batch_dashboard()'s shape -- one run_batch()
    call per judge_id, driven over the GLOBALLY deduplicated item list
    (not per-run), since a context item shared by several selected runs
    (e.g. a plain/grounded pair) is judged only once regardless of how
    many runs reference it."""
    mod = _context_relevance_v1_module()
    run_started_at = datetime.now(timezone.utc)
    run_registry.update_run(job_run_id, status="running", started_at=run_started_at.isoformat())

    run_ids = params["run_ids"]
    judge_ids = params["judge_ids"]
    workers = params.get("workers") or {}
    log_path = ctxrel_v1_run_log_path(job_run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: Any = "") -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    try:
        from app.services import history_service

        items = _build_ctxrel_v1_items(run_ids, limit=params.get("limit"))
        run_file_paths = [history_service.get_history_detail(rid)["output_path"] for rid in run_ids]
        questions_parquet = str(_questions_parquet())
        per_judge_summary: dict[str, Any] = {}
        total_done = 0
        total_target = len(items) * len(judge_ids)
        run_registry.update_run(job_run_id, progress={"current": 0, "total": total_target})

        for judge_id in judge_ids:
            if run_registry.is_cancelled(job_run_id):
                break
            out_path, failures_path, resolved_path = ctxrel_v1_output_paths(judge_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            w = max(1, int(workers.get(judge_id, 4)))

            def on_progress(event: dict[str, Any]) -> None:
                nonlocal total_done
                total_done += 1
                run_registry.update_run(job_run_id, progress={"current": total_done, "total": total_target})
                run_registry.append_result(job_run_id, {
                    "question_id": event.get("question_id"), "judge_id": judge_id,
                    "status": event.get("status"),
                })

            summary = mod.run_batch(
                items, [judge_id], questions_parquet, out_path, failures_path, workers=w,
                resolved_path=resolved_path, on_progress=on_progress,
                check_cancel=lambda: run_registry.is_cancelled(job_run_id),
            )
            per_judge_summary[judge_id] = summary
            log(f"[{judge_id}] judged={summary['judged']} skipped={summary['skipped']} failed={summary['failed']}")

            from llm.manifest import sum_usage_tokens, write_manifest
            finished_at = datetime.now(timezone.utc)
            with open(out_path) as f:
                out_records = [_json.loads(line) for line in f if line.strip()]
            write_manifest(
                out_path, run_label=None, status="interrupted" if run_registry.is_cancelled(job_run_id) else "completed",
                config={"run_ids": run_ids, "judge_id": judge_id, "workers": w},
                started_at_utc=run_started_at.isoformat(), finished_at_utc=finished_at.isoformat(),
                item_counts={"attempted": len(items), "succeeded": summary["judged"], "failed": summary["failed"]},
                prompt_version=mod.PROMPT_VERSION,
                total_tokens=sum_usage_tokens(out_records), extra_output_files=[failures_path],
                log_path=str(log_path),
            )

            mod.append_manifest("context_relevance_v1_run_history.jsonl", {
                "run_started_at": run_started_at.isoformat(),
                "run_files": run_file_paths, "judge": judge_id, "workers": w,
                "out": str(out_path), "prompt_version": mod.PROMPT_VERSION, "source": "dashboard",
                **summary,
            })

        cancelled = run_registry.is_cancelled(job_run_id)
        duration = round((datetime.now(timezone.utc) - run_started_at).total_seconds(), 1)
        run_registry.update_run(
            job_run_id, status="cancelled" if cancelled else "completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            summary={"per_judge": per_judge_summary, "duration_sec": duration},
        )
    except Exception as e:
        run_registry.update_run(
            job_run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(e),
        )


def start_context_relevance_v1_run(job_run_id: str, params: dict[str, Any]) -> None:
    settings_service.apply_to_environment()
    run_registry.update_run(job_run_id, status="queued")
    _enter_queue(_CTXREL_V1_QUEUE_LOCK, _CTXREL_V1_QUEUE, job_run_id)
    try:
        _CTXREL_V1_SEMAPHORE.acquire()
        try:
            _leave_queue(_CTXREL_V1_QUEUE_LOCK, _CTXREL_V1_QUEUE, job_run_id)
            if run_registry.is_cancelled(job_run_id):
                run_registry.update_run(
                    job_run_id, status="cancelled", finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            run_context_relevance_v1_batch_dashboard(job_run_id, params)
        finally:
            _CTXREL_V1_SEMAPHORE.release()
    finally:
        _leave_queue(_CTXREL_V1_QUEUE_LOCK, _CTXREL_V1_QUEUE, job_run_id)


# ---------------------------------------------------------------------
# Per-question detail export (Step 2.3 of docs/agent_prompt_judge_
# results_export.md) -- assembles question/body/tags, reference answer,
# per-context-item relevance label, candidate raw+blinded, citation
# outcome, and each registered judge's label/claims/reasoning/
# served_model for one (run_id, question_id) pair. Read-only, reuses
# existing canonical output files -- never duplicates judging logic.
# ---------------------------------------------------------------------

_TRUNCATE_NOTE = " … [truncated]"


def _truncate(text: str | None, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATE_NOTE


def build_question_detail(run_id: str, question_ids: list[int]) -> list[dict[str, Any]]:
    from app.services import history_service

    detail = history_service.get_history_detail(run_id)
    if not detail or not detail.get("output_path"):
        raise ValueError(f"{run_id}: not found or has no output file")
    output_path = Path(detail["output_path"])

    with open(output_path) as f:
        records_by_qid = {r["question_id"]: r for r in (_json.loads(line) for line in f if line.strip())}

    ctxrel_mod = _context_relevance_v1_module()  # self-heals sys.path for _judge_common/judge_prompt_v1 below
    judge_common = importlib.import_module("_judge_common")
    questions_parquet = str(_questions_parquet())
    judge_registry = _judge_clients_module().JUDGE_REGISTRY
    ctxrel_by_judge = {}
    for judge_id in judge_registry:
        out_path, _, _ = ctxrel_v1_output_paths(judge_id)
        if out_path.exists():
            ctxrel_by_judge[judge_id] = ctxrel_mod.load_ctxrel_records([out_path])

    jv1_mod = _judge_agreement_module()
    jv1_by_judge: dict[str, dict[int, dict]] = {}
    for judge_id in judge_registry:
        out_path, _, _ = judge_v1_output_paths(judge_id)
        if not out_path.exists():
            continue
        records = jv1_mod.load_judge_records([str(out_path)])
        jv1_by_judge[judge_id] = {r["question_id"]: r for r in records if r.get("run_id") == run_id}

    try:
        judge_prompt_v1 = importlib.import_module("judge_prompt_v1")
        blind_candidate = judge_prompt_v1.blind_candidate
    except Exception:
        blind_candidate = lambda text: text  # noqa: E731

    bodies_html = judge_common.load_question_bodies(questions_parquet, question_ids)

    out = []
    for qid in question_ids:
        record = records_by_qid.get(qid)
        if record is None:
            out.append({"question_id": qid, "error": "question_id not found in this run's output file"})
            continue

        body_html = bodies_html.get(qid)
        candidate_raw = record.get("llm_answer", "") or ""

        context_items = []
        for rank, item in enumerate(record.get("retrieved_context") or [], start=1):
            chunk_text = item.get("chunk_text", "") or ""
            text_hash = ctxrel_mod._text_hash(chunk_text)
            ctxrel_entries = {}
            for judge_id, by_key in ctxrel_by_judge.items():
                judged = by_key.get((qid, item.get("answer_id"), text_hash))
                if judged:
                    ctxrel_entries[judge_id] = {"label": judged.get("label"), "reason": judged.get("reason")}
            context_items.append({
                "rank": rank, "so_question_id": item.get("question_id"), "answer_id": item.get("answer_id"),
                "is_accepted": item.get("is_accepted"), "trust_weight": item.get("trust_weight"),
                "hop": item.get("hop"), "source_stage": item.get("source_stage"),
                "combined_score": item.get("combined_score"), "score": item.get("score"),
                "chunk_text": _truncate(chunk_text, 600),
                "context_relevance": ctxrel_entries,
            })

        judges_out = {}
        for judge_id in judge_registry:
            r = jv1_by_judge.get(judge_id, {}).get(qid)
            if r is None:
                continue
            judges_out[judge_id] = {
                "label": r.get("label"), "derived_label": r.get("derived_label"),
                "consistent": r.get("consistent"), "claims": r.get("claims", []),
                "reasoning": r.get("reasoning"), "served_model": r.get("response_model"),
                "parse_error": r.get("parse_error"),
            }

        out.append({
            "question_id": qid,
            "title": record.get("title", ""),
            "body": _truncate(judge_common.html_to_text(body_html, 100000) if body_html else "", 1500),
            "tags": record.get("tags", ""),
            "reference_answer": _truncate(judge_common.html_to_text(record.get("ground_truth_answer", ""), 100000), 1500),
            "context_items": context_items,
            "candidate_raw": candidate_raw,
            "candidate_blinded": blind_candidate(candidate_raw),
            "citation_outcome": _compute_citation_outcome(record),
            "judges": judges_out,
        })
    return out


def _compute_citation_outcome(record: dict[str, Any]) -> str:
    from llm.citations import classify_citation_outcome

    return classify_citation_outcome(bool(record.get("has_citation")), bool(record.get("has_valid_citation")))


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
