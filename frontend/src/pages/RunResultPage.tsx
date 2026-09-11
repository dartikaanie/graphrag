import { Link, useNavigate, useParams } from 'react-router-dom'
import { useCancelRun, useRunStream } from '@/api/hooks'
import { ProgressRunPanel } from '@/components/ProgressRunPanel'
import { MetricSummaryCards } from '@/components/MetricSummaryCards'
import { DataTable, type Column } from '@/components/DataTable'
import { Badge } from '@/components/Badge'
import type { RunResultItem } from '@/types/run'

export function RunResultPage() {
  const { condition = 'a', run_id = '' } = useParams()
  const navigate = useNavigate()
  const { run, connectionError } = useRunStream(run_id)
  const cancelRun = useCancelRun()

  if (!run) return <div className="text-sm text-text-muted">Loading run...</div>

  const isTerminal = ['completed', 'failed', 'cancelled'].includes(run.status)

  const resultColumns: Column<RunResultItem>[] = [
    { key: 'question_id', header: 'ID', render: (r) => <span className="font-mono text-xs">{r.question_id}</span> },
    {
      key: 'status',
      header: 'Status',
      render: (r) => (r.status === 'failed' ? <Badge tone="danger">Failed</Badge> : <Badge tone="success">Done</Badge>),
    },
    { key: 'similarity', header: 'Similarity', render: (r) => (r.similarity != null ? r.similarity.toFixed(4) : '—') },
  ]

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to={`/experiment/${condition}`} className="hover:text-primary">
          ← Back to Condition {condition.toUpperCase()}
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">
        Run {run.run_id} — Condition {run.condition}
      </h1>
      <p className="text-sm text-text-secondary mb-4">
        Mode: {run.mode} · Provider: {run.params.provider} · Model: {run.params.model} · Status:{' '}
        <span className="font-medium">{run.status}</span>
      </p>

      {connectionError && !isTerminal && (
        <div className="text-xs text-warning mb-2">Live stream disconnected — status will refresh on next poll.</div>
      )}

      {!isTerminal && (
        <ProgressRunPanel
          current={run.progress.current}
          total={run.progress.total}
          results={run.results}
          onStop={() => cancelRun.mutate(run.run_id)}
          stopping={cancelRun.isPending || run.cancel_requested}
        />
      )}

      {run.status === 'failed' && run.error && (
        <div className="border border-danger/40 bg-white rounded-lg p-4 text-sm text-danger mb-4">{run.error}</div>
      )}

      {isTerminal && run.summary && (
        <div className="mb-6">
          <h2 className="text-sm font-medium text-text-secondary mb-2">Result Summary</h2>
          <MetricSummaryCards summary={run.summary} condition={run.condition} />
        </div>
      )}

      {isTerminal && run.results.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-text-secondary mb-2">Per-Question Results</h2>
          <DataTable
            columns={resultColumns}
            rows={run.results}
            rowKey={(r) => r.question_id}
            onRowClick={(r) => r.status === 'done' && navigate(`/experiment/${condition}/runs/${run_id}/q/${r.question_id}`)}
          />
        </div>
      )}
    </div>
  )
}
