export interface PageMeta {
  page: number
  page_size: number
  total: number
  total_pages: number
}

export interface QuestionListItem {
  id: number
  title: string
  domain_tag: string | null
  score: number | null
  view_count: number | null
  tags: string | null
  answer_count: number | null
}

export interface QuestionListResponse {
  items: QuestionListItem[]
  meta: PageMeta
}

export interface QuestionDetail {
  id: number
  attributes: Record<string, unknown>
  graph_meta: Record<string, unknown> | null
}

export interface AnswerListItem {
  id: number
  body_preview: string
  score: number | null
  is_accepted: boolean | null
  question_id: number | null
}

export interface AnswerListResponse {
  items: AnswerListItem[]
  meta: PageMeta
}

export interface AnswerDetail {
  id: number
  attributes: Record<string, unknown>
  graph_meta: Record<string, unknown> | null
}

export interface TagListItem {
  name: string
  question_count: number | null
}

export interface TagListResponse {
  items: TagListItem[]
  meta: PageMeta
}

export interface TagDetail {
  name: string
  question_count: number | null
  questions: QuestionListItem[]
}

export interface StatsSummary {
  total_questions: number
  total_answers: number
  total_tags: number
  total_edges: number
}
