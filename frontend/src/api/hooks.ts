import { useEffect, useRef, useState } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { API_BASE, apiDelete, apiGet, apiPost, apiPut } from './client'
import type {
  AnswerDetail,
  AnswerListResponse,
  QuestionDetail,
  QuestionListResponse,
  StatsSummary,
  TagDetail,
  TagListResponse,
} from '@/types/api'
import type { GraphData } from '@/types/graph'
import type { RunCreateParams, RunResultDetail, RunState } from '@/types/run'
import type { HistoryBatchesResponse, HistoryListResponse } from '@/types/history'
import type { JudgeAvailableInput, JudgeEvaluationSummary, JudgeRun, JudgeRunCreateParams } from '@/types/judge'
import type { SettingsState, TestConnectionResult } from '@/types/settings'

export function useStatsSummary() {
  return useQuery({
    queryKey: ['stats-summary'],
    queryFn: () => apiGet<StatsSummary>('/api/stats/summary'),
  })
}

export function useQuestions(page: number, pageSize: number, search?: string, tag?: string) {
  return useQuery({
    queryKey: ['questions', page, pageSize, search, tag],
    queryFn: () =>
      apiGet<QuestionListResponse>('/api/questions', { page, page_size: pageSize, search, tag }),
    placeholderData: (prev) => prev,
  })
}

export function useQuestionDetail(id: string | number) {
  return useQuery({
    queryKey: ['question', id],
    queryFn: () => apiGet<QuestionDetail>(`/api/questions/${id}`),
    enabled: !!id,
  })
}

export function useQuestionAnswers(id: string | number) {
  return useQuery({
    queryKey: ['question-answers', id],
    queryFn: () => apiGet<Record<string, unknown>[]>(`/api/questions/${id}/answers`),
    enabled: !!id,
  })
}

export function useAnswers(page: number, pageSize: number, search?: string) {
  return useQuery({
    queryKey: ['answers', page, pageSize, search],
    queryFn: () => apiGet<AnswerListResponse>('/api/answers', { page, page_size: pageSize, search }),
    placeholderData: (prev) => prev,
  })
}

export function useAnswerDetail(id: string | number) {
  return useQuery({
    queryKey: ['answer', id],
    queryFn: () => apiGet<AnswerDetail>(`/api/answers/${id}`),
    enabled: !!id,
  })
}

export function useTags(page: number, pageSize: number, search?: string) {
  return useQuery({
    queryKey: ['tags', page, pageSize, search],
    queryFn: () => apiGet<TagListResponse>('/api/tags', { page, page_size: pageSize, search }),
    placeholderData: (prev) => prev,
  })
}

export function useTagDetail(name: string) {
  return useQuery({
    queryKey: ['tag', name],
    queryFn: () => apiGet<TagDetail>(`/api/tags/${encodeURIComponent(name)}`),
    enabled: !!name,
  })
}

export function usePartialGraph(limit = 100) {
  return useQuery({
    queryKey: ['graph-partial', limit],
    queryFn: () => apiGet<GraphData>('/api/graph/partial', { limit }),
    staleTime: 5 * 60_000,
  })
}

export function useNodeSubgraph(nodeType: 'question' | 'answer' | 'tag', nodeId: string | number, hops = 2) {
  return useQuery({
    queryKey: ['graph-node', nodeType, nodeId, hops],
    queryFn: () => apiGet<GraphData>(`/api/graph/node/${nodeType}/${encodeURIComponent(String(nodeId))}`, { hops }),
    enabled: !!nodeId,
    staleTime: 5 * 60_000,
  })
}

export function useCreateRun() {
  return useMutation({
    mutationFn: (params: RunCreateParams) => apiPost<{ run_id: string }>('/api/runs', params),
  })
}

// ---------------------------------------------------------------------
// Judge-v1 (hallucination, reference-based) -- separate from the legacy
// judge's hooks below (useCreateJudgeRun/useJudgeAvailableInputs/...).
// ---------------------------------------------------------------------

export interface JudgeV1RegistryEntry {
  judge_id: string
  model: string
  role: 'primary' | 'secondary' | 'fallback' | 'exploratory'
  precision: string
  price_per_m_input: number
  price_per_m_output: number
  price_date: string
  base_url: string | null
  api_key_configured: boolean
}

export function useJudgeV1Registry() {
  return useQuery({
    queryKey: ['judge-v1-registry'],
    queryFn: () => apiGet<{ judges: JudgeV1RegistryEntry[] }>('/api/judge-v1/registry'),
  })
}

export interface JudgeV1PlanRow {
  run_id: string
  run_label?: string
  judge_id?: string
  role?: string | null
  items_total?: number
  already_judged?: number
  items_to_judge?: number
  est_tokens_in?: number | null
  est_tokens_out?: number | null
  est_cost_usd?: number | null
  /** True only when the judge_id isn't in the registry at all -- a
   * registered judge ALWAYS has some token estimate (real history, or
   * the registry's own default), so cost is otherwise always computable
   * and a genuine $0 only ever means "nothing left to judge". Never
   * treat a missing est_cost_usd as $0 -- show "unknown" instead. */
  cost_unknown?: boolean
  est_time_sec?: number
  /** "from history" (real judge-v1 records for this judge_model) or
   * "default estimate" (registry fallback) -- null when cost_unknown. */
  token_source?: 'from history' | 'default estimate' | null
  refused_reason?: string
}

export interface JudgeV1PlanJudgeTotals {
  items_to_judge: number
  est_tokens_in: number | null
  est_tokens_out: number | null
  est_cost_usd: number | null
  cost_unknown_rows: number
  est_time_sec: number
}

export interface JudgeV1PlanTotals extends JudgeV1PlanJudgeTotals {
  per_judge: Record<string, JudgeV1PlanJudgeTotals>
}

export function useJudgeV1Plan() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number> }) =>
      apiPost<{ rows: JudgeV1PlanRow[]; totals: JudgeV1PlanTotals }>('/api/judge-v1/plan', body),
  })
}

/** Per (run, judge) "already judged" status for the run picker -- the
 * SAME plan endpoint (and therefore the SAME resume key) the pre-launch
 * confirmation table uses, so the two can never disagree. */
export function useJudgeV1JudgedStatus(runIds: string[], judgeIds: string[]) {
  return useQuery({
    queryKey: ['judge-v1-judged-status', runIds, judgeIds],
    queryFn: () =>
      apiPost<{ rows: JudgeV1PlanRow[] }>('/api/judge-v1/plan', { run_ids: runIds, judge_ids: judgeIds, workers: {} }),
    enabled: runIds.length > 0 && judgeIds.length > 0,
  })
}

export function useCreateJudgeV1Run() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number> }) =>
      apiPost<{ run_id: string }>('/api/judge-v1/launch', body),
  })
}

export interface JudgeV1Job {
  job_id: string
  run_started_at: string
  run_files: string[]
  judge: string
  out: string
  output_file_missing: boolean
  is_other_config: boolean
  run_labels: (string | null)[]
  batch_id: string | null
  served_model_mismatch_count: number
  judged?: number
  failed?: number
  skipped?: number
}

export function useJudgeV1Jobs(showInvalid = false) {
  return useQuery({
    queryKey: ['judge-v1-jobs', showInvalid],
    queryFn: () => apiGet<{ jobs: JudgeV1Job[] }>('/api/judge-v1/jobs', { show_invalid: showInvalid ? 'true' : undefined }),
  })
}

export interface JudgeV1RunLabelSummary {
  n: number
  abstention_rate: number | null
  hall_rate_excl_abstain: number | null
  hall_rate_incl_abstain: number | null
  label_distribution: Record<string, number>
  parse_error_rate: number | null
  consistency_rate: number | null
}

export function useJudgeV1Summary(judgeId: string) {
  return useQuery({
    queryKey: ['judge-v1-summary', judgeId],
    queryFn: () => apiGet<{ summary: Record<string, JudgeV1RunLabelSummary> }>('/api/judge-v1/results/summary', { judge_id: judgeId }),
  })
}

export interface JudgeV1Agreement {
  n_pairs: number
  confusion_matrix: Record<string, Record<string, number>> | null
  kappas: { weighted: number | null; unweighted: number | null; n_used: number; n_excluded: number } | null
  insufficient_data: boolean
}

export function useJudgeV1Agreement(judgeIdA: string, judgeIdB: string) {
  return useQuery({
    queryKey: ['judge-v1-agreement', judgeIdA, judgeIdB],
    queryFn: () => apiGet<JudgeV1Agreement>('/api/judge-v1/results/agreement', { judge_id_a: judgeIdA, judge_id_b: judgeIdB }),
  })
}

export interface JudgeV1Disagreement {
  run_id: string
  run_label: string | null
  question_id: number
  [key: `${string}_label` | `${string}_reasoning`]: string | number | null | undefined
}

export function useJudgeV1Disagreements(
  judgeIdA: string, judgeIdB: string, runLabel?: string, labelA?: string, labelB?: string,
) {
  return useQuery({
    queryKey: ['judge-v1-disagreements', judgeIdA, judgeIdB, runLabel, labelA, labelB],
    queryFn: () =>
      apiGet<{ disagreements: JudgeV1Disagreement[] }>('/api/judge-v1/results/disagreements', {
        judge_id_a: judgeIdA, judge_id_b: judgeIdB, run_label: runLabel, label_a: labelA, label_b: labelB,
      }),
  })
}

// ---------------------------------------------------------------------
// Judge-v2 (hallucination, reference-based, fixed prompt -- see docs/
// JUDGE_V2_CHANGES.md) -- a SEPARATE set of hooks mirroring judge-v1's
// above, never touching them, so v1's existing UI/behavior is
// unaffected. Shares the same JudgeV1PlanRow/JudgeV1Job/JudgeV1Agreement/
// JudgeV1Disagreement response SHAPES (the endpoints return the same
// JSON shape), just a different API base path + extra v2-only summary
// fields.
// ---------------------------------------------------------------------

export interface JudgeV2RunLabelSummary extends JudgeV1RunLabelSummary {
  completeness_distribution?: Record<string, number>
  factual_and_complete_rate?: number | null
  pct_with_reference_conflict?: number | null
  attempted_claims_conflict_rate?: number | null
}

export function useJudgeV2Plan() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number> }) =>
      apiPost<{ rows: JudgeV1PlanRow[]; totals: JudgeV1PlanTotals }>('/api/judge-v2/plan', body),
  })
}

export function useJudgeV2JudgedStatus(runIds: string[], judgeIds: string[]) {
  return useQuery({
    queryKey: ['judge-v2-judged-status', runIds, judgeIds],
    queryFn: () =>
      apiPost<{ rows: JudgeV1PlanRow[] }>('/api/judge-v2/plan', { run_ids: runIds, judge_ids: judgeIds, workers: {} }),
    enabled: runIds.length > 0 && judgeIds.length > 0,
  })
}

export function useCreateJudgeV2Run() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number> }) =>
      apiPost<{ run_id: string }>('/api/judge-v2/launch', body),
  })
}

export function useJudgeV2Jobs(showInvalid = false) {
  return useQuery({
    queryKey: ['judge-v2-jobs', showInvalid],
    queryFn: () => apiGet<{ jobs: JudgeV1Job[] }>('/api/judge-v2/jobs', { show_invalid: showInvalid ? 'true' : undefined }),
  })
}

export function useJudgeV2Summary(judgeId: string) {
  return useQuery({
    queryKey: ['judge-v2-summary', judgeId],
    queryFn: () => apiGet<{ summary: Record<string, JudgeV2RunLabelSummary>; prompt_version: string }>(
      '/api/judge-v2/results/summary', { judge_id: judgeId },
    ),
  })
}

export function useJudgeV2Agreement(judgeIdA: string, judgeIdB: string) {
  return useQuery({
    queryKey: ['judge-v2-agreement', judgeIdA, judgeIdB],
    queryFn: () => apiGet<JudgeV1Agreement>('/api/judge-v2/results/agreement', { judge_id_a: judgeIdA, judge_id_b: judgeIdB }),
  })
}

export function useJudgeV2Disagreements(
  judgeIdA: string, judgeIdB: string, runLabel?: string, labelA?: string, labelB?: string,
) {
  return useQuery({
    queryKey: ['judge-v2-disagreements', judgeIdA, judgeIdB, runLabel, labelA, labelB],
    queryFn: () =>
      apiGet<{ disagreements: JudgeV1Disagreement[] }>('/api/judge-v2/results/disagreements', {
        judge_id_a: judgeIdA, judge_id_b: judgeIdB, run_label: runLabel, label_a: labelA, label_b: labelB,
      }),
  })
}

export interface JudgeVersionComparison {
  judge_id_a: string
  judge_id_b: string
  transition_matrix: Record<string, Record<string, Record<string, number>>>
  kappa_v1: { n_pairs: number; kappas: { weighted: number | null; unweighted: number | null; n_used: number; n_excluded: number } | null; insufficient_data: boolean }
  kappa_v2: { n_pairs: number; kappas: { weighted: number | null; unweighted: number | null; n_used: number; n_excluded: number } | null; insufficient_data: boolean }
}

export function useJudgeVersionComparison(runIds: string[], judgeIdA: string, judgeIdB: string) {
  return useQuery({
    queryKey: ['judge-version-comparison', runIds, judgeIdA, judgeIdB],
    queryFn: () =>
      apiGet<JudgeVersionComparison>('/api/judge-v2/results/version-comparison', {
        run_ids: runIds.join(','), judge_id_a: judgeIdA, judge_id_b: judgeIdB,
      }),
    enabled: runIds.length > 0,
  })
}

export interface ReferenceConflict {
  question_id: number
  run_label: string | null
  judge_id: string
  claim: string
  evidence: string
  verdict: string
}

export function useReferenceConflicts(runIds: string[], judgeIds: string[]) {
  return useQuery({
    queryKey: ['reference-conflicts', runIds, judgeIds],
    queryFn: () =>
      apiGet<{ conflicts: ReferenceConflict[] }>('/api/judge-v2/results/reference-conflicts', {
        run_ids: runIds.join(','), judge_ids: judgeIds.join(','),
      }),
    enabled: runIds.length > 0,
  })
}

// ---------------------------------------------------------------------
// Context relevance v1 -- Step 3 of docs/agent_prompt_judge_results_
// export.md. Fully separate launch flow from judge-v1's above (own
// plan/launch/status/cancel endpoints), reusing the SAME judge registry.
// ---------------------------------------------------------------------

export interface ContextRelevanceV1PlanRow {
  run_id: string
  run_label?: string
  n_context_items?: number
  refused_reason?: string
}

export interface ContextRelevanceV1PlanJudgeTotals {
  role?: string | null
  already_judged: number
  items_to_judge: number
  est_tokens_in: number | null
  est_tokens_out: number | null
  est_cost_usd: number | null
  cost_unknown: boolean
  est_time_sec: number
  token_source: 'from history' | 'default estimate' | null
}

export interface ContextRelevanceV1PlanTotals {
  unique_items: number
  items_to_judge: number
  est_tokens_in: number | null
  est_tokens_out: number | null
  est_cost_usd: number | null
  cost_unknown_rows: number
  est_time_sec: number
  per_judge: Record<string, ContextRelevanceV1PlanJudgeTotals>
}

export function useContextRelevanceV1Plan() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number> }) =>
      apiPost<{ rows: ContextRelevanceV1PlanRow[]; totals: ContextRelevanceV1PlanTotals }>('/api/context-relevance/plan', body),
  })
}

export function useCreateContextRelevanceV1Run() {
  return useMutation({
    mutationFn: (body: { run_ids: string[]; judge_ids: string[]; workers: Record<string, number>; limit?: number }) =>
      apiPost<{ run_id: string }>('/api/context-relevance/launch', body),
  })
}

export interface ContextRelevanceV1RunMetrics {
  n_items_total: number
  n_items_judged: number
  n_items_unjudged: number
  pct_relevant: number | null
  pct_partial: number | null
  pct_irrelevant: number | null
  n_questions: number
  n_questions_with_relevant: number
  pct_questions_with_relevant: number | null
  mean_relevance_score: number | null
  mean_relevance_score_rank1: number | null
  mean_relevance_score_rank2plus: number | null
}

export function useContextRelevanceV1Summary(runId: string | undefined, judgeId: string) {
  return useQuery({
    queryKey: ['context-relevance-v1-summary', runId, judgeId],
    queryFn: () =>
      apiGet<{ run_id: string; run_label: string | null; judge_id: string; metrics: ContextRelevanceV1RunMetrics }>(
        '/api/context-relevance/results/summary', { run_id: runId, judge_id: judgeId },
      ),
    enabled: !!runId,
  })
}

/** Same data/cache as useContextRelevanceV1Summary, but for a variable-
 * length list of run ids -- used by the markdown export, which needs
 * every batch run's metrics gathered up front rather than one hook call
 * per table row (see useHistoryDetails for the same useQueries pattern). */
export function useContextRelevanceV1SummaryMulti(runIds: string[], judgeId: string) {
  return useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ['context-relevance-v1-summary', runId, judgeId],
      queryFn: () =>
        apiGet<{ run_id: string; run_label: string | null; judge_id: string; metrics: ContextRelevanceV1RunMetrics }>(
          '/api/context-relevance/results/summary', { run_id: runId, judge_id: judgeId },
        ),
      enabled: !!runId,
    })),
  })
}

export interface ContextRelevanceLinkRow {
  question_id: number
  has_relevant_context: boolean
  hallucination_label: string | null
}

export interface ContextRelevanceLinkTables {
  label_x_relevant: Record<string, { has_relevant: number; no_relevant: number }>
  abstain_x_relevant: { has_relevant: { abstain: number; not_abstain: number }; no_relevant: { abstain: number; not_abstain: number } } | null
}

export function useContextRelevanceLinkToOutcomes(runId: string | undefined) {
  return useQuery({
    queryKey: ['context-relevance-v1-link', runId],
    queryFn: () =>
      apiGet<{ run_id: string; run_label: string | null; rows: ContextRelevanceLinkRow[]; tables: ContextRelevanceLinkTables }>(
        '/api/context-relevance/results/link-to-outcomes', { run_id: runId },
      ),
    enabled: !!runId,
  })
}

/** Same data/cache as useContextRelevanceLinkToOutcomes, for a variable-
 * length list of run ids (useQueries pattern, see useHistoryDetails). */
export function useContextRelevanceLinkToOutcomesMulti(runIds: string[]) {
  return useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ['context-relevance-v1-link', runId],
      queryFn: () =>
        apiGet<{ run_id: string; run_label: string | null; rows: ContextRelevanceLinkRow[]; tables: ContextRelevanceLinkTables }>(
          '/api/context-relevance/results/link-to-outcomes', { run_id: runId },
        ),
      enabled: !!runId,
    })),
  })
}

export interface ContextRelevanceBatchSummaryRow {
  run_label: string
  run_id: string | null
  no_context: boolean
  metrics: ContextRelevanceV1RunMetrics | null
}

export function useContextRelevanceBatchSummary(batchId: string | undefined, judgeId: string) {
  return useQuery({
    queryKey: ['context-relevance-v1-batch-summary', batchId, judgeId],
    queryFn: () =>
      apiGet<{ batch_id: string; judge_id: string; rows: ContextRelevanceBatchSummaryRow[]; has_any_results: boolean }>(
        '/api/context-relevance/results/batch-summary', { batch_id: batchId, judge_id: judgeId },
      ),
    enabled: !!batchId,
  })
}

export interface ContextRelevanceV1Job {
  job_id: string
  run_started_at: string
  run_files: string[]
  judge: string
  out: string
  output_file_missing: boolean
  is_other_config: boolean
  run_labels: (string | null)[]
  batch_id: string | null
  judged?: number
  failed?: number
  skipped?: number
}

export function useContextRelevanceV1Jobs(showInvalid = false) {
  return useQuery({
    queryKey: ['context-relevance-v1-jobs', showInvalid],
    queryFn: () => apiGet<{ jobs: ContextRelevanceV1Job[] }>('/api/context-relevance/jobs', { show_invalid: showInvalid ? 'true' : undefined }),
  })
}

export interface JudgeV1QuestionDetailContextItem {
  rank: number
  so_question_id: number | null
  answer_id: string | null
  is_accepted: boolean | null
  trust_weight: number | null
  hop: number | string | null
  source_stage: string | null
  combined_score: number | null
  score: number | null
  chunk_text: string
  context_relevance: Record<string, { label: string | null; reason: string | null }>
}

export interface JudgeV1QuestionDetailJudge {
  label: string | null
  derived_label: string | null
  consistent: boolean | null
  claims: Array<{ claim?: string; verdict?: string; severity?: string; evidence?: string }>
  reasoning: string | null
  served_model: string | null
  parse_error: boolean | null
}

export interface JudgeV1QuestionDetail {
  question_id: number
  error?: string
  title?: string
  body?: string
  tags?: string
  reference_answer?: string
  context_items?: JudgeV1QuestionDetailContextItem[]
  candidate_raw?: string
  candidate_blinded?: string
  citation_outcome?: string
  judges?: Record<string, JudgeV1QuestionDetailJudge>
}

export function fetchJudgeV1QuestionDetail(runId: string, questionIds: number[]) {
  return apiGet<{ questions: JudgeV1QuestionDetail[] }>('/api/judge-v1/results/question-detail', {
    run_id: runId, question_ids: questionIds.join(','),
  })
}

export interface CheckCompletedResult {
  config_hash: string
  already_completed: boolean
}

/**
 * Factorial Batch page's skip-detection -- computed server-side, keyed on
 * config_hash (the SAME rule the resume gate itself uses,
 * llm.manifest.read_already_done), rather than a separate client-side
 * field-by-field comparison that could silently drift from it. Results
 * come back in the SAME order as the `runs` sent.
 */
export function useCheckCompleted() {
  return useMutation({
    mutationFn: (runs: RunCreateParams[]) =>
      apiPost<{ results: CheckCompletedResult[] }>('/api/runs/check-completed', { runs }),
  })
}

/**
 * Live run state via SSE, per PLAN_UI_UX.md §5.4/§8 point 4: fetches the
 * current state once immediately (so a page refresh mid-run or after
 * completion shows correct data even before/without a stream connecting),
 * then layers live updates on top via EventSource. The backend closes the
 * stream itself once the run reaches a terminal status.
 */
export function useRunStream(runId: string | undefined) {
  const [run, setRun] = useState<RunState | null>(null)
  const [connectionError, setConnectionError] = useState(false)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    setRun(null)
    setConnectionError(false)

    apiGet<RunState>(`/api/runs/${runId}`)
      .then((r) => !cancelled && setRun(r))
      .catch(() => {})

    const es = new EventSource(`${API_BASE}/api/runs/${runId}/stream`)
    esRef.current = es
    es.onmessage = (event) => {
      try {
        setRun(JSON.parse(event.data) as RunState)
      } catch {
        // ignore malformed event
      }
    }
    es.onerror = () => setConnectionError(true)

    return () => {
      cancelled = true
      es.close()
    }
  }, [runId])

  return { run, connectionError }
}

export function useCancelRun() {
  return useMutation({
    mutationFn: (runId: string) => apiPost(`/api/runs/${runId}/cancel`),
  })
}

export function useRunResultDetail(runId: string, questionId: number | string) {
  return useQuery({
    queryKey: ['run-result-detail', runId, questionId],
    queryFn: () => apiGet<RunResultDetail>(`/api/runs/${runId}/results/${questionId}`),
    enabled: !!runId && questionId !== undefined,
  })
}

export function useHistory(condition: string | undefined, page: number, pageSize: number, showSuperseded = false) {
  return useQuery({
    queryKey: ['history', condition, page, pageSize, showSuperseded],
    queryFn: () =>
      apiGet<HistoryListResponse>('/api/history', {
        condition, page, page_size: pageSize,
        ...(showSuperseded ? { show_superseded: 'true' } : {}),
      }),
    placeholderData: (prev) => prev,
  })
}

/** Every run (A/B/C/D), grouped into launch batches -- for the
 * Hallucination Judge (judge-v1) page's run picker. */
export function useHistoryBatches(showSuperseded = false) {
  return useQuery({
    queryKey: ['history-batches', showSuperseded],
    queryFn: () =>
      apiGet<HistoryBatchesResponse>('/api/history/batches', {
        ...(showSuperseded ? { show_superseded: 'true' } : {}),
      }),
    placeholderData: (prev) => prev,
  })
}

export function useHistoryDetail(historyId: string) {
  return useQuery({
    queryKey: ['history-detail', historyId],
    queryFn: () => apiGet<RunState>(`/api/history/${historyId}`),
    enabled: !!historyId,
  })
}

/** Same data/cache as useHistoryDetail, but for a variable-length list of
 * ids (HistoryComparePage supports comparing any number of runs, not a
 * fixed 4 slots) -- useQueries lets the number of queries vary per render
 * without breaking the rules of hooks the way calling useHistoryDetail in
 * a loop would. Shares its queryKey shape with useHistoryDetail, so a run
 * already cached from the History Detail page doesn't get re-fetched. */
export function useHistoryDetails(historyIds: string[]) {
  return useQueries({
    queries: historyIds.map((id) => ({
      queryKey: ['history-detail', id],
      queryFn: () => apiGet<RunState>(`/api/history/${id}`),
      enabled: !!id,
    })),
  })
}

export interface SampleConsistencyResult {
  sample_consistency: 'identical' | 'subset_nested' | 'different'
  runs: { history_id: string; condition: string; n: number }[]
  intersection_size: number
  is_exact_prefix: boolean | null
  note?: string
}

/** Phase 3 -- did the selected runs actually evaluate the same questions?
 * The whole point of a paired comparison. `ids` sorted+joined into the
 * query key so selection order doesn't create duplicate cache entries. */
export function useCompareConsistency(historyIds: string[]) {
  const key = [...historyIds].sort().join(',')
  return useQuery({
    queryKey: ['compare-consistency', key],
    queryFn: () => apiGet<SampleConsistencyResult>('/api/history/compare/consistency', { ids: historyIds.join(',') }),
    enabled: historyIds.length >= 2,
  })
}

export function useHistoryResultDetail(historyId: string, questionId: number | string) {
  return useQuery({
    queryKey: ['history-result-detail', historyId, questionId],
    queryFn: () => apiGet<RunResultDetail>(`/api/history/${historyId}/results/${questionId}`),
    enabled: !!historyId && questionId !== undefined,
  })
}

export function useSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: () => apiGet<SettingsState>('/api/settings'),
  })
}

export interface OfficialCParams {
  /** "pending_selection" (the C retrieval v3 decision isn't made yet --
   * the Official preset refuses to launch ANY run while this holds,
   * regardless of c_retrieval_version) or "locked" (the Outcome section
   * of docs/DECISION_C_SCORING.md has been filled in). */
  status: 'pending_selection' | 'locked'
  /** null while status is "pending_selection" -- the selection
   * procedure hasn't run yet, so there is no known winning version. */
  c_retrieval_version: string | null
  alpha: number | null
  sample_split: string
  max_hops: number
  edge_types: string[]
  use_author_trust: boolean
  accepted_only: boolean
}

export interface ConfigDefaults {
  default_oversample_pool: number
  max_planned_n_sample: number
  prompt_version: string
  c_retrieval_version: string
  d_retrieval_version: string
  /** Locked Condition C v3 parameters for the "Official (n=384)" preset,
   * read from docs/DECISION_C_SCORING.md's machine-readable block (see
   * backend/app/routers/config.py::_official_c_params()). null if the
   * file doesn't exist. */
  official_c_params: OfficialCParams | null
}

/** Static run defaults (not user-editable, unlike Settings) -- oversample
 * pool/n_sample plus the CURRENT prompt/retrieval versions (shown in the
 * Factorial Batch dry-run, FactorialBatchPage.tsx), so the frontend has
 * one place to read them from instead of hard-coding them. */
export function useConfigDefaults() {
  return useQuery({
    queryKey: ['config-defaults'],
    queryFn: () => apiGet<ConfigDefaults>('/api/config/defaults'),
    staleTime: Infinity, // static backend constant -- never refetch mid-session
  })
}

export interface ReleaseFaissCacheResult {
  released: boolean
  rss_before_mb: number | null
  rss_after_mb: number | null
}

/** Frees the ~4GB shared FAISS index + embedding memmap Condition C/D keep
 * cached between runs -- see Settings page "Release cached FAISS index". */
export function useReleaseFaissCache() {
  return useMutation({
    mutationFn: () => apiPost<ReleaseFaissCacheResult>('/api/config/release-faiss-cache'),
  })
}

export function useArchiveDest() {
  return useQuery({
    queryKey: ['archive-dest'],
    queryFn: () => apiGet<{ dest: string | null }>('/api/settings/archive-dest'),
  })
}

export interface ArchiveResult {
  copied: number
  dry_run: boolean
  dest: string
  sums_path?: string
  mismatches?: string[]
  would_copy?: number
}

export function useArchiveResults() {
  return useMutation({
    mutationFn: (dryRun: boolean) => apiPost<ArchiveResult>('/api/settings/archive', { dry_run: dryRun }),
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (patch: Record<string, unknown>) => apiPut<SettingsState>('/api/settings', patch),
    onSuccess: (data) => queryClient.setQueryData(['settings'], data),
  })
}

export function useTestConnection() {
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => apiPost<TestConnectionResult>('/api/settings/test-connection', body),
  })
}

export function useRunResultGraph(runId: string, questionId: number | string) {
  return useQuery({
    queryKey: ['run-result-graph', runId, questionId],
    queryFn: () => apiGet<GraphData>(`/api/runs/${runId}/results/${questionId}/graph`),
    enabled: !!runId && questionId !== undefined,
  })
}

export function useDeleteHistory() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (historyId: string) => apiDelete<{ status: string; history_id: string }>(`/api/history/${historyId}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['history'] }),
  })
}

export function useJudgeAvailableInputs(condition: string | undefined) {
  return useQuery({
    queryKey: ['judge-available-inputs', condition],
    queryFn: () => apiGet<{ items: JudgeAvailableInput[] }>('/api/judge-runs/available-inputs', { condition }),
    enabled: !!condition,
  })
}

export function useCreateJudgeRun() {
  return useMutation({
    mutationFn: (params: JudgeRunCreateParams) => apiPost<{ run_id: string }>('/api/judge-runs', params),
  })
}

/** Has this exact result file already been judged, and with what
 * config(s)? Used to show cached results + a "re-judge (force)" option
 * instead of a plain Run button once a file is selected. */
export function useJudgeRunsForInput(inputPath: string | undefined) {
  return useQuery({
    queryKey: ['judge-runs-for-input', inputPath],
    queryFn: () => apiGet<{ items: JudgeEvaluationSummary[] }>('/api/judge-runs/for-input', { input_path: inputPath }),
    enabled: !!inputPath,
  })
}

/**
 * Judge runs are polled (not SSE) -- simpler for a run shape that only
 * needs to refresh every couple seconds until it reaches a terminal status,
 * matching the polling approach the spec calls for rather than duplicating
 * the SSE machinery useRunStream uses for the higher-frequency condition runs.
 */
export function useJudgeRunPoll(runId: string | undefined) {
  const [run, setRun] = useState<JudgeRun | null>(null)
  const [connectionError, setConnectionError] = useState(false)

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    setRun(null)
    setConnectionError(false)

    const poll = async () => {
      try {
        const data = await apiGet<JudgeRun>(`/api/judge-runs/${runId}`)
        if (cancelled) return
        setRun(data)
        if (!['completed', 'failed', 'cancelled'].includes(data.status)) {
          timer = setTimeout(poll, 1500)
        }
      } catch {
        if (!cancelled) setConnectionError(true)
      }
    }
    let timer: ReturnType<typeof setTimeout>
    poll()

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [runId])

  return { run, connectionError }
}

export function useHistoryResultGraph(historyId: string, questionId: number | string) {
  return useQuery({
    queryKey: ['history-result-graph', historyId, questionId],
    queryFn: () => apiGet<GraphData>(`/api/history/${historyId}/results/${questionId}/graph`),
    enabled: !!historyId && questionId !== undefined,
  })
}
