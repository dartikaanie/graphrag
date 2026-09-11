import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAnswers } from '@/api/hooks'
import { DataTable, type Column } from '@/components/DataTable'
import { Pagination } from '@/components/Pagination'
import { SearchInput } from '@/components/SearchInput'
import { Badge } from '@/components/Badge'
import type { AnswerListItem } from '@/types/api'

const PAGE_SIZE = 25

export function AnswerListPage() {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const navigate = useNavigate()
  const { data, isLoading, isError } = useAnswers(page, PAGE_SIZE, search || undefined)

  const columns: Column<AnswerListItem>[] = [
    { key: 'id', header: 'ID', render: (a) => <span className="font-mono text-xs">{a.id}</span> },
    {
      key: 'body_preview',
      header: 'Body',
      className: 'max-w-lg',
      render: (a) => <span className="line-clamp-2 text-text-secondary" dangerouslySetInnerHTML={{ __html: a.body_preview }} />,
    },
    { key: 'score', header: 'Score', render: (a) => a.score ?? '—' },
    {
      key: 'is_accepted',
      header: 'Accepted',
      render: (a) => (a.is_accepted ? <Badge tone="success">Accepted</Badge> : '—'),
    },
    {
      key: 'question_id',
      header: 'Question',
      render: (a) => (a.question_id ? <span className="font-mono text-xs">{a.question_id}</span> : '—'),
    },
  ]

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Answers</h1>
      <div className="mb-4">
        <SearchInput
          value={search}
          onChange={(v) => {
            setSearch(v)
            setPage(1)
          }}
          placeholder="Search answer bodies..."
        />
      </div>

      {isError && <div className="text-sm text-danger mb-4">Failed to load answers.</div>}

      <DataTable
        columns={columns}
        rows={data?.items ?? []}
        rowKey={(a) => a.id}
        onRowClick={(a) => navigate(`/answers/${a.id}`)}
        emptyLabel={isLoading ? 'Loading...' : 'No answers found.'}
      />

      {data && data.meta.total_pages > 1 && (
        <Pagination page={page} totalPages={data.meta.total_pages} onChange={setPage} />
      )}
    </div>
  )
}
