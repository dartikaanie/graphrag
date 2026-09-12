import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
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
import type { HistoryListResponse } from '@/types/history'
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

export function useHistory(condition: string | undefined, page: number, pageSize: number) {
  return useQuery({
    queryKey: ['history', condition, page, pageSize],
    queryFn: () => apiGet<HistoryListResponse>('/api/history', { condition, page, page_size: pageSize }),
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
