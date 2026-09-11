import { Link, useNavigate, useParams } from 'react-router-dom'
import { useHistoryDetail } from '@/api/hooks'
import { MetricSummaryCards } from '@/components/MetricSummaryCards'
import { DataTable, type Column } from '@/components/DataTable'
import { Badge } from '@/components/Badge'
import type { RunResultItem } from '@/types/run'

export function HistoryDetailPage() {
  const { history_id = '' } = useParams()
  const navigate = useNavigate()
  const { data: run, isLoading, isError } = useHistoryDetail(history_id)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !run) return <div className="text-sm text-danger">History entry not found.</div>

  const columns: Column<RunResultItem>[] = [
    { key: 'question_id', header: 'ID', render: (r) => <span className="font-mono text-xs">{r.question_id}</span> },
    { key: 'similarity', header: 'Similarity', render: (r) => (r.similarity != null ? r.similarity.toFixed(4) : '—') },
  ]

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/history" className="hover:text-primary">
          ← Back to History
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">
        Run {run.run_id} — Condition {run.condition}
      </h1>
      <p className="text-sm text-text-secondary mb-4">
        {new Date(run.created_at).toLocaleString()} · Mode: {run.mode} · Provider: {run.params.provider} · Model:{' '}
        {run.params.model} · Status: <Badge tone={run.status === 'success' || run.status === 'completed' ? 'success' : 'warning'}>{run.status}</Badge>
      </p>

      {run.summary && (
        <div className="mb-6">
          <h2 className="text-sm font-medium text-text-secondary mb-2">Result Summary</h2>
          <MetricSummaryCards summary={run.summary} condition={run.condition} />
        </div>
      )}

      {run.results.length > 0 ? (
        <div>
          <h2 className="text-sm font-medium text-text-secondary mb-2">Per-Question Results</h2>
          <DataTable
            columns={columns}
            rows={run.results}
            rowKey={(r) => r.question_id}
            onRowClick={(r) => navigate(`/history/${history_id}/q/${r.question_id}`)}
          />
        </div>
      ) : (
        <div className="text-sm text-text-muted">
          No per-question results available (the underlying results file may have been moved or deleted).
        </div>
      )}
    </div>
  )
}
