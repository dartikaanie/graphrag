import { useProcessLogTail } from '@/api/hooks'
import type { LogsResponse } from '@/types/process'
import { formatTime, STEP_LABELS } from '@/lib/processFormat'

const TAIL_OPTIONS = [100, 200, 500, 1000]

interface Props {
  data: LogsResponse
  selected: string | null
  onSelect: (id: string) => void
  tail: number
  onTailChange: (n: number) => void
}

export function LogsTab({ data, selected, onSelect, tail, onTailChange }: Props) {
  const current = selected ?? data.logs[0]?.id ?? null
  const { data: log, isLoading, error } = useProcessLogTail(current, tail)

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(260px,340px)_1fr]">
      <div className="border border-border rounded-lg bg-surface overflow-hidden">
        <div className="px-3 py-2 border-b border-border text-xs text-text-muted">
          {data.logs.length} files from the allowlist, newest first
        </div>
        <ul className="max-h-[70vh] overflow-y-auto">
          {data.logs.map((l) => (
            <li key={l.id}>
              <button
                onClick={() => onSelect(l.id)}
                className={`w-full text-left px-3 py-2 border-b border-border last:border-b-0 ${
                  l.id === current ? 'bg-primary-soft' : 'hover:bg-bg'
                }`}
              >
                <div className="font-mono text-xs break-all text-text-primary">{l.name}</div>
                <div className="text-xs text-text-muted">
                  {l.step ? STEP_LABELS[l.step] : 'no step'} · {formatTime(l.timestamp_utc)} ·{' '}
                  {(l.size_bytes / 1024).toFixed(1)} KB
                </div>
              </button>
            </li>
          ))}
        </ul>
        <div className="px-3 py-2 border-t border-border text-xs text-text-muted">
          Allowed directories:
          {data.roots.map((r) => (
            <div key={r} className="font-mono break-all">
              {r}
            </div>
          ))}
        </div>
      </div>

      <div className="border border-border rounded-lg bg-surface min-w-0">
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-border">
          <div className="font-mono text-xs break-all text-text-primary">{log?.name ?? current ?? 'No log selected'}</div>
          <label className="flex items-center gap-1.5 text-xs text-text-secondary">
            Last
            <select
              value={tail}
              onChange={(e) => onTailChange(Number(e.target.value))}
              className="px-1.5 py-1 text-xs border border-border-strong rounded-md bg-surface"
            >
              {TAIL_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            lines
          </label>
        </div>
        {isLoading && <div className="px-3 py-6 text-sm text-text-muted">Loading…</div>}
        {error && <div className="px-3 py-6 text-sm text-danger">Could not load: {(error as Error).message}</div>}
        {log && (
          <>
            <pre className="px-3 py-2 text-xs font-mono text-text-primary whitespace-pre overflow-auto max-h-[70vh]">
              {log.text}
            </pre>
            <div className="px-3 py-2 border-t border-border text-xs text-text-muted">
              {log.lines_returned} lines shown{log.truncated ? ' (tail only)' : ''} · secrets redacted
            </div>
          </>
        )}
      </div>
    </div>
  )
}
