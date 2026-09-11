import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useHistory } from '@/api/hooks'
import { Pagination } from '@/components/Pagination'
import { Badge } from '@/components/Badge'
import type { HistoryItem } from '@/types/history'

const PAGE_SIZE = 20

const STATUS_TONE: Record<string, 'success' | 'danger' | 'warning' | 'neutral'> = {
  success: 'success',
  failed: 'danger',
  no_results: 'danger',
  no_valid_records: 'danger',
  cancelled: 'warning',
  interrupted: 'warning',
}

function fmtTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function HistoryPage() {
  const [page, setPage] = useState(1)
  const [condition, setCondition] = useState<string | undefined>(undefined)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const navigate = useNavigate()
  const { data, isLoading, isError } = useHistory(condition, page, PAGE_SIZE)

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else if (next.size < 3) next.add(id)
      return next
    })
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-semibold text-text-primary">History</h1>
        {selected.size >= 2 && (
          <button
            onClick={() => navigate(`/history/compare?ids=${Array.from(selected).join(',')}`)}
            className="px-3 py-1.5 text-sm bg-primary text-white rounded-md hover:bg-primary-hover"
          >
            Compare ({selected.size})
          </button>
        )}
      </div>

      <div className="flex items-center gap-3 mb-4">
        <select
          className="px-2.5 py-1.5 text-sm border border-border-strong rounded-md bg-surface text-text-primary"
          value={condition ?? ''}
          onChange={(e) => {
            setCondition(e.target.value || undefined)
            setPage(1)
          }}
        >
          <option value="">All conditions</option>
          <option value="A">Condition A</option>
          <option value="B">Condition B</option>
          <option value="C">Condition C</option>
        </select>
        <span className="text-xs text-text-muted">Select up to 3 runs to compare.</span>
      </div>

      {isError && <div className="text-sm text-danger mb-4">Failed to load history.</div>}

      <div className="overflow-x-auto border border-border rounded-lg">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border-strong">
              <th className="w-8 px-3 py-2.5" />
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">Timestamp</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">Condition</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">N</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">Model</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">Avg Similarity</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">NF2%</th>
              <th className="text-left font-medium text-text-secondary px-3 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody>
            {!isLoading && data?.items.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-8 text-center text-text-muted">
                  No runs yet.
                </td>
              </tr>
            )}
            {data?.items.map((item: HistoryItem) => (
              <tr key={item.history_id} className="border-b border-border last:border-b-0 hover:bg-bg">
                <td className="px-3 py-2.5">
                  <input type="checkbox" checked={selected.has(item.history_id)} onChange={() => toggle(item.history_id)} />
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {fmtTimestamp(item.run_started_at)}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {item.condition}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {item.n_processed}/{item.n_sample_target}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {item.provider}/{item.model}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {item.cosine_similarity_mean != null ? item.cosine_similarity_mean.toFixed(4) : '—'}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  {item.pct_with_valid_citation != null ? `${item.pct_with_valid_citation.toFixed(1)}%` : '—'}
                </td>
                <td className="px-3 py-2.5 cursor-pointer" onClick={() => navigate(`/history/${item.history_id}`)}>
                  <Badge tone={STATUS_TONE[item.status] ?? 'neutral'}>{item.status}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && data.meta.total_pages > 1 && (
        <Pagination page={page} totalPages={data.meta.total_pages} onChange={setPage} />
      )}
    </div>
  )
}
