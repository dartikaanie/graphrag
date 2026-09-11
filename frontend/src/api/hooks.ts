import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { API_BASE, apiGet, apiPost } from './client'
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
