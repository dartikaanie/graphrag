export type HallucinationLabel = 'FAKTUAL' | 'HALUSINASI_SEBAGIAN' | 'HALUSINASI_PENUH'

export interface JudgeRunSummary {
  n_total?: number
  n_evaluated_baru?: number
  n_dari_cache?: number
  pct_faktual?: number
  pct_halusinasi_sebagian?: number
  pct_halusinasi_penuh?: number
  mean_faithfulness_score?: number | null
  mean_answer_relevance_score?: number | null
  duration_sec?: number
  kappa_value?: number
  kappa_interpretation?: string
  judge_models_identical?: boolean
  output_path?: string
}

export interface JudgeRunResultItem {
  question_id: number | null
  status: 'cached' | 'evaluated' | 'failed'
  hallucination_label?: HallucinationLabel | null
  error?: string | null
}

export interface JudgeRun {
  run_id: string
  condition: 'JUDGE'
  mode: 'batch'
  status: string
  params: {
    input_path: string
    condition: string
    judge_provider?: string | null
    judge_model?: string | null
    judge_temperature?: number | null
    majority_rounds?: number | null
    force?: boolean
    kappa_validation: boolean
    secondary_judge_provider?: string | null
    secondary_judge_model?: string | null
    kappa_sample_size?: number | null
  }
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress: { current: number; total: number }
  results: JudgeRunResultItem[]
  summary: JudgeRunSummary | null
  error: string | null
  output_path: string | null
}

export interface JudgeAvailableInput {
  path: string
  filename: string
  provider: string | null
  model: string | null
  n_sample: number | null
  seed: number | null
  mode: 'batch' | 'single'
}

export interface JudgeRunCreateParams {
  input_path: string
  condition: 'A' | 'B' | 'C' | 'D'
  judge_provider?: string
  judge_model?: string
  judge_temperature?: number
  majority_rounds?: number
  force?: boolean
  kappa_validation?: boolean
  secondary_judge_provider?: string
  secondary_judge_model?: string
  kappa_sample_size?: number
}

/** Summary of one past judge run, joined onto a condition's History Detail
 * page by input_path -- there can be more than one if the same result file
 * was judged with different configs. Matches
 * judge_lookup_service.get_judge_evaluations_for_input()'s shape. */
export interface JudgeEvaluationSummary {
  judge_provider: string | null
  judge_model: string | null
  judge_temperature: number | null
  majority_rounds: number | null
  n_total: number | null
  pct_faktual: number | null
  pct_sebagian: number | null
  pct_penuh: number | null
  mean_faithfulness: number | null
  mean_answer_relevance: number | null
  kappa_value?: number | null
  kappa_interpretation?: string | null
  evaluated_at: string | null
  output_path: string | null
}

/** One judge config's verdict for a single question, joined onto
 * HistoryResultDetailPage -- matches
 * judge_lookup_service.get_judge_results_for_question()'s shape. */
export interface JudgeQuestionResult {
  judge_provider: string | null
  judge_model: string | null
  judge_temperature: number | null
  majority_rounds: number | null
  hallucination_label: HallucinationLabel
  faithfulness_score: number | null
  answer_relevance_score: number
  justification: string
  evaluated_at: string | null
}
