import type { RunResultItem } from '@/types/run'

interface ProgressRunPanelProps {
  current: number
  total: number
  results: RunResultItem[]
  onStop?: () => void
  stopping?: boolean
  /** Current run status -- when "queued", the panel shows a distinct
   * waiting-for-a-slot message instead of the progress bar (progress.total
   * is already set from the request, so a plain "0 / N" would misleadingly
   * look like the run has started and produced nothing). */
  status?: string
  /** 1-based, only meaningful while status === "queued". */
  queuePosition?: number | null
}

/** `compact`: a tighter shape for pages showing many cards at once (e.g.
 * FactorialResultsPage's 9 cards) -- a one-line "Queued (position N)" +
 * Cancel instead of the pulsing bar + full-size log box, and a shorter
 * log area, so queued/running/done cards don't visually jump in size. */
export function ProgressRunPanel({
  current, total, results, onStop, stopping, status, queuePosition, compact,
}: ProgressRunPanelProps & { compact?: boolean }) {
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0
  const isQueued = status === 'queued'

  if (compact && isQueued) {
    return (
      <div className="border border-border rounded-lg bg-surface p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-text-secondary">
            Queued{queuePosition != null ? ` (position ${queuePosition})` : ''}
          </span>
          {onStop && (
            <button
              onClick={onStop}
              disabled={stopping}
              className="px-3 py-1 text-xs border border-border-strong rounded-md text-danger hover:bg-bg disabled:opacity-50"
            >
              {stopping ? 'Stopping…' : '■ Cancel'}
            </button>
          )}
        </div>
        <p className="text-xs text-text-muted mt-1">Waiting for a free run slot…</p>
      </div>
    )
  }

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-text-secondary">
          {isQueued
            ? `Queued${queuePosition != null ? ` (position ${queuePosition})` : ''} — waiting for a free run slot…`
            : `Progress: ${current} / ${total}`}
        </span>
        {onStop && (
          <button
            onClick={onStop}
            disabled={stopping}
            className="px-3 py-1 text-xs border border-border-strong rounded-md text-danger hover:bg-bg disabled:opacity-50"
          >
            {stopping ? 'Stopping…' : isQueued ? '■ Cancel (leave queue)' : '■ Stop'}
          </button>
        )}
      </div>
      {isQueued ? (
        <div className="h-1.5 rounded-full bg-border overflow-hidden mb-3">
          <div className="h-full bg-text-muted animate-pulse" style={{ width: '100%' }} />
        </div>
      ) : (
        <div className="h-1.5 rounded-full bg-border overflow-hidden mb-3">
          <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
        </div>
      )}
      <div className={`font-mono text-xs bg-bg rounded-md p-3 overflow-y-auto flex flex-col gap-1 ${compact ? 'max-h-40' : 'max-h-56'}`}>
        {isQueued && (
          <span className="text-text-muted">Another run is currently using this resource — this run will start automatically once it's free.</span>
        )}
        {!isQueued && results.length === 0 && <span className="text-text-muted">Waiting for the first result...</span>}
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
