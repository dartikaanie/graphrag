// GET /api/report -- llm/evaluation/official_report.py (Amendment 14 draft, plan step 9). Read-only.

export interface ThreeNumbers {
  n: number
  n_usable: number
  n_missing: number
  hallucinated: number
  not_hallucinated: number
  abstain: number
  rate_excl_abstain: number | null
  abstain_rate: number | null
  rate_abstain_as_failed: number | null // (hallucinated + ABSTAIN) / usable answers
}

export interface FrozenParameter {
  group: string
  parameter: string
  value: unknown
  source: string
  status: string
  decision: string
}

export interface ReportRun {
  run_id: string
  label: string
  status: string
  condition: string
  grounding: boolean | null
  alpha: number | null
  n: number
  commit: string | null
  config_hash: string | null
  retrieval: {
    context_items_mean: number | null
    empty_context: number | null
    latency_sec: { p50: number | null; p95: number | null; n: number }
  }
  generation: {
    n: number
    abstain_panel: number
    tokens_in: number
    tokens_out: number
    cost_usd: number | null
    sec_per_answer: number | null
    failures: number | null
  }
  judging: {
    label_distribution: Record<string, Record<string, number>>
    unresolved: number
    incomplete_panels: number
    missing_judge_records: Record<string, number>
    failures_unresolved: number
  }
  results: {
    panel: ThreeNumbers
    judges: Record<string, ThreeNumbers>
    cosine: { mean: number | null; median: number | null }
    nf2_citation_validity: number | null
    latency_vs_nf3: {
      method: string
      fallback: boolean
      value_sec: number | null
      target_sec: number
      nf3_applies: boolean
      meets: boolean | null
    }
  }
  latency: {
    end_to_end_sec: { p50: number | null; p95: number | null; n: number }
    generation_sec: { p50: number | null; p95: number | null; n: number }
    retrieval_sec: { p50: number | null; p95: number | null; n: number }
    run_duration_per_answer_sec: number | null
    schema_r3_answers: number
  }
  reliability: {
    fleiss_kappa_3cat: number | null
    n_complete: number
    krippendorff_alpha_nominal: number | null
    all_three_judges_same?: { n_complete: number; same_label: number | null; same_3cat_label: number | null }
  }
}

export interface PairedCounts {
  c_better: number
  equal: number
  c_worse: number
  n_pairs: number
  n_excluded: number
  cells: { only_c: number; only_other: number; both: number; neither: number }
}

export interface Comparison {
  comparison: string
  name?: string
  available: boolean
  difference_panel?: number | null
  difference_panel_abstain_as_failed?: number | null
  paired?: { primary_excl_abstain: PairedCounts; abstain_as_failed: PairedCounts }
  difference_by_judge?: Record<string, number | null>
  flag?: 'consistent' | 'inconclusive'
}

export interface Report {
  meta: { generated_at: string; mock: boolean; panel: string; judge_version: string; n_runs: number; wording: string }
  configuration: {
    frozen_parameters: FrozenParameter[]
    table_hash: string | null
    amendments_in_decision_record: string[]
    amendment14: { approved: boolean; draft: string }
    git_head: string | null
  }
  data: {
    test_sample: { n: number; seed: number; positions: string; frozen_list: string }
    dev_set: { n: number; positions: string; frozen_list: string }
    leakage: { mode: string; n_pool: number | null; modes_built: string[] | null; built_at_utc: string | null; sha256: string | null; rules: string }
  }
  alpha_selection: {
    stage1: {
      from_archive: boolean
      source: string
      table: { alpha: number; pct_questions_with_relevant: number | null; mean_relevance_score: number | null }[]
      top_two: number[] | null
      reasoning: string | null
    }
    stage2: {
      source: string
      winner: number | null
      alpha_star: number | null
      alpha_abl: number | null
      tie_rule_used: string | null
      pooled: Record<string, { rate_excl_abstain: number | null; hallucination_count_excl_abstain: number; n_excl_abstain: number }> | null
    } | null
    official_params_status: string
    alpha_star: number | null
    alpha_abl: number | null
  }
  runs: ReportRun[]
  comparisons: Comparison[]
  budget: {
    budget: { deepinfra_budget_usd: number | null; set_by?: string | null; set_at?: string | null } | null
    spent_usd: number
    remaining_usd: number | null
    judging_cost_usd: number
    ledger: { label: string; cost_usd: number; recorded_at?: string }[]
  }
  limitations: {
    amendment14_text: string | null
    amendment14_status: string
    archived_evidence: Record<string, { path: string; from_archive: boolean; text: string | null }>
    key_numbers: Record<string, unknown>
    wording: string
  }
}

export interface ReportItemRow {
  question_id: number
  title: string | null
  judge_labels: Record<string, string | null>
  panel_label: string
}

export interface ReportClaim {
  claim?: string
  verdict?: string
  severity?: string | null
  evidence?: string
}

export interface ReportItem {
  run_id: string
  question_id: number
  title: string | null
  reference_html: string | null
  context: { answer_id?: number; chunk_text?: string }[]
  candidate: string | null
  judges: Record<string, { label: string | null; claims: ReportClaim[] | null; reasoning: string | null }>
  panel: { panel_label: string; vote_breakdown: Record<string, unknown>; panel_incomplete: boolean }
  mock: boolean
}
