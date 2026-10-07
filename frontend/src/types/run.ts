import type { JudgeEvaluationSummary, JudgeQuestionResult } from './judge'

export type RunCondition = 'A' | 'B' | 'C' | 'D'
export type RunMode = 'batch' | 'single'
// Live runs use pending/running/completed/failed/cancelled; history entries
// (read straight from run_history.jsonl, written by the CLI's own
// conventions predating the dashboard) can also be success/no_results/
// no_valid_records/interrupted/unknown -- kept as a plain string rather than
// re-deriving/duplicating that whole legacy vocabulary here.
export type RunStatusValue = string

export interface RunCreateParams {
  condition: RunCondition
  mode: RunMode
  n_sample?: number
  seed?: number
  oversample_pool?: number | null
  /** Response-only derived field (never sent in a request): how many
   * candidates survived filter_by_token_limit() out of oversample_pool --
   * a cheap sanity signal the pool was large enough. null on old records
   * that predate this field. */
  n_candidates_after_token_filter?: number | null
  index_pool?: number | null
  top_k?: number
  n_anchor?: number
  n_semantic_expansion?: number
  require_citation?: boolean
  fusion_mode?: 'trust_weighted' | 'uniform'
  fusion_w_path_trust?: number
  fusion_w_intrinsic?: number
  semantic_expansion_trust_cap?: number
  enable_semantic_expansion?: boolean
  n_low_level?: number
  n_high_level?: number
  require_grounding?: boolean
  log_full_candidates?: boolean | null
  /** Response-only. "v2" (current), "v1 (inferred)" when reliably inferable
   * from the record's date, or null ("--") otherwise. Never sent in a
   * request. */
  prompt_version?: string | null
  /** Response-only, Condition D only -- null for every other condition. */
  d_retrieval_version?: string | null
  /** Condition C only. Sendable in a request ("v2"/"v3", see
   * docs/DECISION_C_SCORING.md) and also returned on a response record
   * (null for every other condition). */
  c_retrieval_version?: string | null
  /** Condition C v3 only. */
  alpha?: number | null
  /** Condition C only -- "test" (default) or "dev" (Dev (n=50) preset). */
  sample_split?: string
  /** Condition C v3 exploratory switches -- see
   * docs/DECISION_C_SCORING.md's "Exploratory switches" section. */
  max_hops?: number
  edge_types?: string[] | null
  use_author_trust?: boolean
  accepted_only?: boolean
  question_id?: number | null
  /** Factorial Batch page only -- shared by every run in one "Confirm &
   * Launch" click (see FactorialBatchPage.tsx), recorded into
   * run_history.jsonl so the Hallucination Judge page's run picker can
   * group runs launched together without re-inferring it. */
  batch_id?: string
  batch_launched_at?: string
  provider: string
  model: string
}

export interface RunResultItem {
  question_id: number
  index: number
  total: number
  status: 'done' | 'failed'
  similarity: number | null
  error: string | null
}

export interface RunSummary {
  n_processed?: number
  // Judge-v1 dashboard job summary shape (engine_service.
  // run_judge_v1_batch_dashboard) -- a judge-v1 job run isn't a
  // generator run, so it doesn't set most of the fields below.
  per_judge?: Record<string, { judged: number; skipped: number; failed: number; parse_errors: number }>
  cosine_similarity_mean: number | null
  cosine_similarity_median?: number
  pct_similarity_above_0_5?: number
  pct_with_citation?: number
  pct_with_valid_citation?: number
  avg_retrieval_latency_sec?: number
  duration_sec?: number
  require_grounding?: boolean
  enable_semantic_expansion?: boolean
  // 3-way citation split (docs/NF2_ROOT_CAUSE_PLACEHOLDER_CITATIONS.md) --
  // pct_with_valid_citation above is the "valid" share; these three are
  // the other two slices of the same 100% plus a precision score.
  pct_citation_no_citation?: number
  pct_citation_invalid_only?: number
  fabricated_citation_rate?: number
  citation_precision?: number
  n_context_items_used_mean?: number
  n_context_items_used_min?: number
}

export interface RunState {
  run_id: string
  condition: RunCondition
  mode: RunMode
  status: RunStatusValue
  // Only present when this RunState came from get_history_detail() (a
  // history_id, not a live run_id) -- history_service.py's
  // derive_run_label(), see types/history.ts's HistoryItem for the same
  // field on list items.
  run_label?: string
  params: RunCreateParams
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress: { current: number; total: number }
  results: RunResultItem[]
  summary: RunSummary | null
  error: string | null
  output_path: string | null
  cancel_requested?: boolean
  judge_evaluations?: JudgeEvaluationSummary[]
  /** 1-based, only meaningful while status === "queued" -- best-effort (not
   * a scheduling guarantee), see engine_service.py's queue bookkeeping. */
  queue_position?: number | null
}

export interface RetrievedContextItem {
  question_id?: number
  answer_id?: number
  chunk_text: string
  trust_weight?: number
  combined_score?: number
  relevance_score?: number
  hop?: number | string
  rel_type?: string | null
  source_stage?: 'graph_traversal' | 'semantic_expansion' | 'low_level' | 'high_level'
  is_accepted?: boolean | null
}

export interface AnchorQuestion {
  question_id: number
  similarity: number
}

export interface PromptMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface RunResultDetail {
  question_id: number
  title: string
  tags: string
  n_tokens: number
  view_count: number | null
  question_score: number | null
  accepted_answer_id: number
  ground_truth_answer: string
  retrieved_context?: RetrievedContextItem[]
  anchor_question_ids?: AnchorQuestion[]
  prompt_messages?: PromptMessage[]
  n_anchors?: number
  n_graph_candidates?: number
  n_expansion_candidates?: number
  n_low_level_candidates?: number
  n_high_level_candidates?: number
  /** Actual number of context items sent to the LLM in this prompt, after
   * the top_k cutoff -- same field/meaning for B, C, and D, so context
   * count parity can be verified per-question after a pilot run. */
  n_context_items_used?: number
  require_grounding?: boolean
  enable_semantic_expansion?: boolean
  retrieval_latency_sec?: number
  llm_answer: string
  llm_model: string
  cosine_similarity: number
  has_citation?: boolean
  cited_source_ids?: string[]
  has_valid_citation?: boolean
  valid_cited_source_ids?: string[]
  judge_results?: JudgeQuestionResult[]
}
