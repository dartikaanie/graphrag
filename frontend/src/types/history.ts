export interface HistoryItem {
  run_started_at: string
  condition: 'A' | 'B' | 'C'
  status: string
  provider: string
  model: string
  n_sample_target: number
  n_processed: number
  seed: number
  top_k?: number
  n_anchor?: number
  n_semantic_expansion?: number
  output_path: string
  duration_sec: number
  source?: string
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
