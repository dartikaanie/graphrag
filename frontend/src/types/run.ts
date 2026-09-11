export type RunCondition = 'A' | 'B' | 'C'
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
  top_k?: number
  n_anchor?: number
  n_semantic_expansion?: number
  question_id?: number | null
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
  n_processed: number
  cosine_similarity_mean: number | null
  cosine_similarity_median?: number
  pct_similarity_above_0_5?: number
  pct_with_citation?: number
  pct_with_valid_citation?: number
  avg_retrieval_latency_sec?: number
}

export interface RunState {
  run_id: string
  condition: RunCondition
  mode: RunMode
  status: RunStatusValue
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
}

export interface RetrievedContextItem {
  question_id?: number
  answer_id?: number
  chunk_text: string
  trust_weight?: number
  combined_score?: number
  hop?: number
  rel_type?: string | null
  source_stage?: 'graph_traversal' | 'semantic_expansion'
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
  retrieval_latency_sec?: number
  llm_answer: string
  llm_model: string
  cosine_similarity: number
  has_citation?: boolean
  cited_source_ids?: string[]
  has_valid_citation?: boolean
  valid_cited_source_ids?: string[]
}
