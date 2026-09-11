import type { RunResultItem } from '@/types/run'

interface ProgressRunPanelProps {
  current: number
  total: number
  results: RunResultItem[]
  onStop?: () => void
  stopping?: boolean
}

export function ProgressRunPanel({ current, total, results, onStop, stopping }: ProgressRunPanelProps) {
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-text-secondary">
          Progress: {current} / {total}
        </span>
        {onStop && (
          <button
            onClick={onStop}
            disabled={stopping}
            className="px-3 py-1 text-xs border border-border-strong rounded-md text-danger hover:bg-bg disabled:opacity-50"
          >
            {stopping ? 'Stopping…' : '■ Stop'}
          </button>
        )}
      </div>
      <div className="h-1.5 rounded-full bg-border overflow-hidden mb-3">
        <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="font-mono text-xs bg-bg rounded-md p-3 max-h-56 overflow-y-auto flex flex-col gap-1">
        {results.length === 0 && <span className="text-text-muted">Waiting for the first result...</span>}
        {results.map((r) => (
          <div key={r.question_id} className={r.status === 'failed' ? 'text-danger' : 'text-text-secondary'}>
            {r.status === 'failed' ? (
              <>✗ Id={r.question_id} failed: {r.error}</>
            ) : (
              <>✓ Id={r.question_id} similarity={r.similarity?.toFixed(3)}</>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
