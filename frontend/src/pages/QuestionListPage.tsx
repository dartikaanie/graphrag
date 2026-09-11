import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuestions } from '@/api/hooks'
import { DataTable, type Column } from '@/components/DataTable'
import { Pagination } from '@/components/Pagination'
import { SearchInput } from '@/components/SearchInput'
import { Badge } from '@/components/Badge'
import type { QuestionListItem } from '@/types/api'

const PAGE_SIZE = 25

export function QuestionListPage() {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const navigate = useNavigate()
  const { data, isLoading, isError } = useQuestions(page, PAGE_SIZE, search || undefined)

  const columns: Column<QuestionListItem>[] = [
    { key: 'id', header: 'ID', render: (q) => <span className="font-mono text-xs">{q.id}</span> },
    {
      key: 'title',
      header: 'Title',
      className: 'max-w-md',
      render: (q) => <span className="line-clamp-2">{q.title}</span>,
    },
    { key: 'domain_tag', header: 'Domain', render: (q) => q.domain_tag ?? '—' },
    { key: 'score', header: 'Score', render: (q) => q.score ?? '—' },
    { key: 'view_count', header: 'Views', render: (q) => q.view_count?.toLocaleString() ?? '—' },
    {
      key: 'answer_count',
      header: 'Answers',
      render: (q) => (q.answer_count != null ? <Badge tone="primary">{q.answer_count}</Badge> : '—'),
    },
  ]

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Questions</h1>
      <div className="mb-4">
        <SearchInput
          value={search}
          onChange={(v) => {
            setSearch(v)
            setPage(1)
          }}
          placeholder="Search question titles..."
        />
      </div>

      {isError && <div className="text-sm text-danger mb-4">Failed to load questions.</div>}

      <DataTable
        columns={columns}
        rows={data?.items ?? []}
        rowKey={(q) => q.id}
        onRowClick={(q) => navigate(`/questions/${q.id}`)}
        emptyLabel={isLoading ? 'Loading...' : 'No questions found.'}
      />

      {data && data.meta.total_pages > 1 && (
        <Pagination page={page} totalPages={data.meta.total_pages} onChange={setPage} />
      )}
    </div>
  )
}
