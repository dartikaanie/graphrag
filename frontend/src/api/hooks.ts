import { useQuery } from '@tanstack/react-query'
import { apiGet } from './client'
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
