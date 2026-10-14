// Types for the read-only Process & Parameters API (backend/app/routers/process.py).

export interface SourceQuote {
  text: string | null
  file: string
  line: number | null
  line_end?: number
  error?: string
}

export interface LastChange {
  commit: string | null
  date: string | null
  uncommitted: boolean
}

export type ParamStatus = 'frozen' | 'pending' | 'candidate' | 'exploratory' | 'observed'

export interface ProcessParameter {
  key: string
  name: string
  step: string
  unit: string
  kind: 'parameter' | 'result'
  used_by: string[]
  set_by: string | null
  status: ParamStatus
  note: string | null
  value: unknown
  defined_in: { file: string; line: number | null } | null
  source_note: string | null
  error: string | null
  last_changed: LastChange | null
  rationale: SourceQuote | null
  rationale_documented: boolean
  spec: SourceQuote | null
}

export interface ParametersResponse {
  as_of: string
  parameters: ProcessParameter[]
  undocumented: string[]
}

export interface TrustHistoryEntry {
  version: string
  event: string
  summary: string | null
  log_id: string
  file: string
  timestamp_utc: string | null
  applied_at: string | null
  backup: string | null
  verify: string | null
}

export interface VerifyState {
  log_id: string
  file: string
  timestamp_utc: string | null
  counts: Record<string, { nodes: number; versions: Record<string, number> }>
  results: string[]
}

export interface Stage1Row {
  alpha: number
  pct_questions_with_relevant: { v2: number | null; v3: number | null }
  mean_relevance_score: { v2: number | null; v3: number | null }
  pct_accepted_selected: { v2: number | null; v3: number | null }
  n_questions: number | null
}

export interface Stage1Result {
  file: string
  log_id: string | null
  timestamp_utc: string | null
  git_commit: string | null
  judge_model: string | null
  table: Stage1Row[]
  ranked: number[] | null
  top_two: number[] | null
  keep_current: boolean | null
  reasoning: string | null
}

export interface CalibrationStatus {
  status: string
  annotation_files_created: { created_at_utc: string | null; n_items: number | null; n_items_annotator2: number | null } | null
  adjudicated_reference_exists: boolean
  judge_frozen: boolean
  source: string
}

export interface ProcessStep {
  key: string
  number: number
  title: string
  purpose: string
  modules: { file: string; exists: boolean }[]
  inputs: string
  outputs: string
  quotes: (SourceQuote | null)[]
  parameters: string[]
  status: string
  last_evidence_utc: string | null
  logs: string[]
  trust_history?: TrustHistoryEntry[]
  verify?: VerifyState | null
  stage1?: Stage1Result | null
  stage2?: { status: string; outputs_exist: boolean }
  calibration?: CalibrationStatus
  official_params?: Record<string, unknown> | null
}

export interface StepsResponse {
  as_of: string
  sources_as_of: Record<string, string | null>
  headline: {
    stage1: string
    stage2: string
    calibration: string
    official_run: string
    trust_score_version: Record<string, number> | null
  }
  steps: ProcessStep[]
}

export interface DecisionSection {
  heading: string
  line: number
  text: string
}

export interface DecisionEntry {
  id: string
  number: number
  date: string | null
  title: string
  line: number
  line_end: number
  sections: DecisionSection[]
  commits: string[]
  applied: string[]
  why: DecisionSection | null
  intro: string | null
}

export interface DecisionsResponse {
  as_of: string | null
  source: string
  decisions: DecisionEntry[]
}

export interface ProcessRun {
  run_id: string
  file: string
  condition: string
  split: string | null
  alpha: number | null
  grounding: boolean | null
  fusion_mode: string | null
  c_retrieval_version: string | null
  trust_score_version: string | null
  config_hash: string | null
  prompt_version: string | null
  judge_version: string | null
  model: string | null
  n_sample: number | null
  status: string | null
  started_at_utc: string | null
  finished_at_utc: string | null
  commit: string | null
  git_dirty: boolean | null
  items_succeeded: number | null
  run_kind: 'generation' | 'retrieval-only' | 'judge'
}

export interface RunsResponse {
  as_of: string
  dirs: string[]
  runs: ProcessRun[]
}

export interface ProcessLog {
  id: string
  root: string
  name: string
  size_bytes: number
  timestamp_utc: string | null
  step: string | null
}

export interface LogsResponse {
  as_of: string
  roots: string[]
  logs: ProcessLog[]
}

export interface LogTail {
  id: string
  name: string
  step: string | null
  size_bytes: number
  lines_returned: number
  truncated: boolean
  text: string
}

export interface LiveState {
  available: boolean | null
  as_of: string | null
  values: {
    nodes: Record<string, number>
    edges: Record<string, number>
    answer_trust_versions: Record<string, number>
    question_trust_versions: Record<string, number>
  } | null
  message: string | null
}

export interface LiveResponse {
  default: VerifyState | null
  live: LiveState
  queries: Record<string, string>
}
