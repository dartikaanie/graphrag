"""Pydantic response models for Phase 1 (master data + stats)."""

from typing import Any

from pydantic import BaseModel


class PageMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class QuestionListItem(BaseModel):
    id: int
    title: str
    domain_tag: str | None = None
    score: int | None = None
    view_count: int | None = None
    tags: str | None = None
    answer_count: int | None = None


class QuestionListResponse(BaseModel):
    items: list[QuestionListItem]
    meta: PageMeta


class QuestionDetail(BaseModel):
    id: int
    attributes: dict[str, Any]
    graph_meta: dict[str, Any] | None = None


class AnswerListItem(BaseModel):
    id: int
    body_preview: str
    score: int | None = None
    is_accepted: bool | None = None
    question_id: int | None = None


class AnswerListResponse(BaseModel):
    items: list[AnswerListItem]
    meta: PageMeta


class AnswerDetail(BaseModel):
    id: int
    attributes: dict[str, Any]
    graph_meta: dict[str, Any] | None = None


class TagListItem(BaseModel):
    name: str
    question_count: int | None = None


class TagListResponse(BaseModel):
    items: list[TagListItem]
    meta: PageMeta


class TagDetail(BaseModel):
    name: str
    question_count: int | None = None
    questions: list[QuestionListItem]


class StatsSummary(BaseModel):
    total_questions: int
    total_answers: int
    total_tags: int
    total_edges: int


class RunCreateRequest(BaseModel):
    condition: str  # "A" | "B" | "C" | "D"
    mode: str  # "batch" | "single"
    # batch mode
    n_sample: int = 30
    seed: int = 42
    oversample_pool: int | None = None
    top_k: int = 5
    n_anchor: int = 3
    n_semantic_expansion: int = 3
    # Condition B only -- default True to match CONDITION_B_REQUIRE_CITATION
    # in .env; lets NF2 (citation compliance) be compared B vs C.
    require_citation: bool = True
    # Condition C only -- ablation study fusion mode, see
    # llm/c_graphrag/c_graphrag.py fuse_and_rank(). Defaults match
    # FUSION_MODE/FUSION_W_PATH_TRUST/FUSION_W_ANSWER_INTRINSIC_TRUST/
    # SEMANTIC_EXPANSION_TRUST_CAP in .env (the CLI's own defaults).
    fusion_mode: str = "trust_weighted"  # "trust_weighted" | "uniform"
    fusion_w_path_trust: float = 0.7
    fusion_w_intrinsic: float = 0.3
    semantic_expansion_trust_cap: float = 0.4
    # Condition C only -- ablation switch to skip the semantic expansion
    # stage entirely (retrieval = anchor + graph traversal only).
    enable_semantic_expansion: bool | None = True
    # Condition D only -- dual-level retrieval (adaptasi LightRAG), lihat
    # llm/d_lightrag/d_lightrag.py. Runs on the SAME KG/FAISS cache as
    # Condition C (no trust weighting at all).
    n_low_level: int | None = None
    n_high_level: int | None = None
    # Condition B, C, AND D -- grounding constraint toggle (see
    # build_rag_messages()/build_graphrag_messages()/build_lightrag_messages()
    # in llm/prompts.py; prompt-parity v3, docs/PROMPT_PARITY_V3.md). True:
    # dual-constraint grounding+citation, BYTE-IDENTICAL text across B/C/D.
    # False: citation instruction only, no "don't introduce facts outside
    # the context" rule. Default is None (not True) so the PER-CONDITION
    # default in engine_service.py applies when the client omits this field:
    # True for C/D (their original behavior, unchanged), False for B
    # ("B-plain" -- B never had a grounding constraint before this field
    # existed, so an old/unaware client must keep getting that, not
    # silently switch to "B-grounded").
    require_grounding: bool | None = None
    # Condition B and C only -- opt-in, see analyze_retrieval_quality.py's
    # docstring ("KENAPA TIDAK ADA RECALL@k") for why this exists: persists
    # all_candidate_question_ids (the full candidate pool BEFORE the top-k
    # cutoff) into each record, enabling real Recall@k on a FUTURE run of
    # that analysis script. Does not affect retrieval/ranking/prompt at all.
    log_full_candidates: bool = False
    # single mode
    question_id: int | None = None
    # Factorial Batch page only -- shared by every run in one "Confirm &
    # Launch" click, generated client-side once per batch (see
    # FactorialBatchPage.tsx). Recorded verbatim into run_history.jsonl so
    # the Hallucination Judge page's run picker can group runs launched
    # together without having to re-infer it (inference is a FALLBACK for
    # older runs that predate this field, see batch_grouping_service.py).
    batch_id: str | None = None
    batch_launched_at: str | None = None
    # shared
    provider: str = "openai"
    model: str = "gpt-4o-mini"


class RunCreateResponse(BaseModel):
    run_id: str


class CheckCompletedRequest(BaseModel):
    """One entry per factorial run the Factorial Batch page is about to
    (maybe) launch -- same shape as RunCreateRequest, so the SAME params
    the page would actually submit are what gets hashed, mode="single"
    always resolves to not-already-completed (see
    engine_service.compute_run_config_hash's docstring)."""
    runs: list[RunCreateRequest]


class CheckCompletedResult(BaseModel):
    config_hash: str
    already_completed: bool


class CheckCompletedResponse(BaseModel):
    results: list[CheckCompletedResult]


class JudgeRunCreateRequest(BaseModel):
    input_path: str
    condition: str  # "A" | "B" | "C" | "D"
    # All optional -- fall back to settings_service.get_raw_settings()
    # (judge_provider/judge_model/judge_temperature/...) when omitted, same
    # precedence pattern as the generator conditions' provider/model.
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_temperature: float | None = None
    majority_rounds: int | None = None
    force: bool = False
    kappa_validation: bool = False
    secondary_judge_provider: str | None = None
    secondary_judge_model: str | None = None
    kappa_sample_size: int | None = None


class JudgeRunCreateResponse(BaseModel):
    run_id: str


class RunStatus(BaseModel):
    run_id: str
    condition: str
    mode: str
    status: str  # pending | running | completed | failed | cancelled
    params: dict[str, Any]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any]
    summary: dict[str, Any] | None = None
    error: str | None = None
    output_path: str | None = None


class RunResultItem(BaseModel):
    question_id: int
    index: int
    status: str
    similarity: float | None = None


class SettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_host: str | None = None
    num_ctx: int | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    questions_parquet: str | None = None
    answers_parquet: str | None = None
    # LLM-as-Judge config -- see settings_service.DEFAULTS.
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_temperature: float | None = None
    secondary_judge_provider: str | None = None
    secondary_judge_model: str | None = None
    kappa_sample_size: int | None = None
    judge_majority_rounds: int | None = None
    judges_wait_for_heavy_run: bool | None = None


class ArchiveRequest(BaseModel):
    dest: str | None = None  # defaults to GRAPHRAG_ARCHIVE_DIR (.env) when omitted
    dry_run: bool = False


class JudgeV1PlanRequest(BaseModel):
    run_ids: list[str]
    judge_ids: list[str]
    workers: dict[str, int] = {}


class JudgeV1LaunchRequest(BaseModel):
    run_ids: list[str]
    judge_ids: list[str]
    workers: dict[str, int] = {}


class JudgeV1ExportHumanCsvRequest(BaseModel):
    run_ids: list[str]
    n: int = 60
    min_per_label: int = 5
    seed: int = 42


class JudgeV1ImportHumanCsvRequest(BaseModel):
    csv_path: str
    mapping_path: str


class TestConnectionRequest(BaseModel):
    target: str  # "neo4j" | "llm"
    provider: str | None = None
    api_key: str | None = None
    model: str | None = None
    ollama_host: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    error: str | None = None
