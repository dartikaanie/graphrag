import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTags } from '@/api/hooks'
import { DataTable, type Column } from '@/components/DataTable'
import { Pagination } from '@/components/Pagination'
import { SearchInput } from '@/components/SearchInput'
import type { TagListItem } from '@/types/api'

const PAGE_SIZE = 25

export function TagListPage() {
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const navigate = useNavigate()
  const { data, isLoading, isError } = useTags(page, PAGE_SIZE, search || undefined)

  const columns: Column<TagListItem>[] = [
    { key: 'name', header: 'Name', render: (t) => <span className="font-medium">{t.name}</span> },
    { key: 'question_count', header: 'Questions', render: (t) => t.question_count?.toLocaleString() ?? '—' },
  ]

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Tags</h1>
      <div className="mb-4">
        <SearchInput
          value={search}
          onChange={(v) => {
            setSearch(v)
            setPage(1)
          }}
          placeholder="Search tags..."
        />
      </div>

      {isError && <div className="text-sm text-danger mb-4">Failed to load tags.</div>}

      <DataTable
        columns={columns}
        rows={data?.items ?? []}
        rowKey={(t) => t.name}
        onRowClick={(t) => navigate(`/tags/${encodeURIComponent(t.name)}`)}
        emptyLabel={isLoading ? 'Loading...' : 'No tags found.'}
      />

      {data && data.meta.total_pages > 1 && (
        <Pagination page={page} totalPages={data.meta.total_pages} onChange={setPage} />
      )}
    </div>
  )
}
