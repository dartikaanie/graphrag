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
    # Condition C AND D -- grounding constraint toggle (see
    # build_graphrag_messages()/build_lightrag_messages() in llm/prompts.py).
    # True (default): dual-constraint grounding+citation. False: citation
    # instruction only, no "don't introduce facts outside the context" rule.
    require_grounding: bool | None = True
    # single mode
    question_id: int | None = None
    # shared
    provider: str = "openai"
    model: str = "gpt-4o-mini"


class RunCreateResponse(BaseModel):
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
