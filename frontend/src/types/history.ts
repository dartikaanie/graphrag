export interface HistoryItem {
  run_started_at: string
  condition: 'A' | 'B' | 'C' | 'D'
  status: string
  provider: string
  model: string
  n_sample_target: number
  n_processed: number
  seed: number
  oversample_pool?: number
  top_k?: number
  n_anchor?: number
  n_semantic_expansion?: number
  n_low_level?: number
  n_high_level?: number
  require_grounding?: boolean
  require_citation?: boolean
  fusion_w_path_trust?: number
  // raw run_history.jsonl field name (history_service.py renames it to
  // fusion_w_intrinsic only inside get_history_detail's nested `params`,
  // list items keep the raw field name).
  fusion_w_answer_intrinsic_trust?: number
  semantic_expansion_trust_cap?: number
  enable_semantic_expansion?: boolean
  prompt_version?: string | null
  c_retrieval_version?: string | null
  d_retrieval_version?: string | null
  // 3-axis factorial label (docs/GROUNDING_FACTOR_UI.md), e.g. "A",
  // "B-plain", "C-uniform-grounded" -- single source of truth is
  // history_service.py::derive_run_label(), never derived client-side.
  run_label?: string
  fusion_mode?: string | null
  grounding?: 'on' | 'off' | null
  grounding_inferred?: boolean
  output_path: string
  duration_sec: number
  source?: string
  config_hash?: string | null
  batch_id?: string | null
  batch_launched_at?: string | null
  cosine_similarity_mean: number | null
  cosine_similarity_median?: number
  pct_similarity_above_0_5?: number
  pct_with_citation?: number
  pct_with_valid_citation?: number
  avg_retrieval_latency_sec?: number
  history_id: string
}

export interface HistoryListResponse {
  items: HistoryItem[]
  meta: { page: number; page_size: number; total: number; total_pages: number }
  active_paths: Record<string, string>
}

export interface HistoryBatch {
  batch_id: string
  inferred: boolean
  launched_at: string | null
  runs: HistoryItem[]
}

export interface HistoryBatchesResponse {
  batches: HistoryBatch[]
}
