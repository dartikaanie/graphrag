import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useCancelRun, useRunStream } from '@/api/hooks'
import { Badge } from '@/components/Badge'
import { ProgressRunPanel } from '@/components/ProgressRunPanel'
import type { RunCondition, RunStatusValue } from '@/types/run'

/** Canonical factorial order -- also the order FactorialBatchPage.tsx's
 * buildFactorialRuns() emits runs in (and therefore submits/enqueues
 * them in), so a fresh batch's cards land in this order without needing
 * to re-sort here. Sorted again below anyway, defensively, so this page
 * never depends on the query param preserving that order. */
const FACTORIAL_ORDER = [
  'A', 'B-plain', 'B-grounded',
  'C-uniform-plain', 'C-uniform-grounded', 'C-trust-plain', 'C-trust-grounded',
  'D-plain', 'D-grounded',
]

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(4)
}

const STATUS_TONE: Record<string, 'success' | 'danger' | 'warning' | 'primary' | 'neutral'> = {
  completed: 'success',
  failed: 'danger',
  cancelled: 'warning',
  running: 'primary',
  queued: 'neutral',
}

function RunCard({ runLabel, condition, runId }: { runLabel: string; condition: RunCondition; runId: string }) {
  const navigate = useNavigate()
  const { run, connectionError } = useRunStream(runId)
  const cancelRun = useCancelRun()

  // No "Loading…" placeholder -- until the first fetch/SSE event lands,
  // treat the card as queued (same shape as a real queued run) rather
  // than rendering nothing/a bare loading line.
  const status: RunStatusValue = run?.status ?? 'queued'
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(status)
  const current = run?.progress.current ?? 0
  const total = run?.progress.total ?? 0
  const results = run?.results ?? []

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-text-primary">{runLabel}</h2>
          <Badge tone={STATUS_TONE[status] ?? 'neutral'}>{status}</Badge>
        </div>
        <Link to={`/experiment/${condition.toLowerCase()}/runs/${runId}`} className="text-xs text-primary hover:underline">
          Full view →
        </Link>
      </div>

      {connectionError && !isTerminal && <div className="text-xs text-warning mb-2">Live stream disconnected.</div>}

      <ProgressRunPanel
        current={current}
        total={total}
        results={results}
        onStop={isTerminal ? undefined : () => cancelRun.mutate(runId)}
        stopping={cancelRun.isPending || run?.cancel_requested}
        status={status}
        queuePosition={run?.queue_position}
        compact
      />

      {status === 'failed' && run?.error && (
        <div className="mt-2 text-xs text-danger border border-danger/40 bg-white rounded-md p-2">{run.error}</div>
      )}

      {isTerminal && run?.summary && (
        <div className="mt-2 flex items-center justify-between text-xs text-text-secondary border-t border-border pt-2">
          <span>
            N={run.summary.n_processed ?? '—'} · Avg Sim={fmt(run.summary.cosine_similarity_mean)} · NF2=
            {run.summary.pct_with_valid_citation !== undefined ? `${run.summary.pct_with_valid_citation.toFixed(1)}%` : '—'} ·
            Fabric={run.summary.fabricated_citation_rate !== undefined ? `${run.summary.fabricated_citation_rate.toFixed(1)}%` : '—'}
          </span>
          <button
            onClick={() => navigate(`/experiment/${condition.toLowerCase()}/runs/${runId}`)}
            className="px-2.5 py-1 text-xs border border-border-strong rounded-md text-text-primary hover:bg-bg shrink-0 ml-3"
          >
            View per-question results
          </button>
        </div>
      )}
    </div>
  )
}

/** Results view for FactorialBatchPage.tsx's 9-run launch -- generalizes
 * RunAllResultsPage.tsx's 4-column layout to an arbitrary labeled set
 * ("A", "B-plain", "B-grounded", "C-uniform-plain", ...), encoded in the
 * `runs` query param as "label:condition:runId" entries joined by commas
 * (URLSearchParams already handles the encoding/decoding). */
export function FactorialResultsPage() {
  const [searchParams] = useSearchParams()

  const runsParam = searchParams.get('runs') ?? ''
  const entries = runsParam
    .split(',')
    .filter(Boolean)
    .map((entry) => {
      const [runLabel, condition, runId] = entry.split(':')
      return { runLabel, condition: condition as RunCondition, runId }
    })
    .sort((a, b) => FACTORIAL_ORDER.indexOf(a.runLabel) - FACTORIAL_ORDER.indexOf(b.runLabel))

  const failedParam = searchParams.get('failed')
  const failures = failedParam
    ? failedParam.split('|').map((entry) => {
        const [runLabel, ...rest] = entry.split(':')
        return { runLabel, message: rest.join(':') }
      })
    : []

  const skippedParam = searchParams.get('skipped')
  const skipped = skippedParam ? skippedParam.split(',').filter(Boolean) : []

  if (entries.length === 0 && failures.length === 0) {
    return <div className="text-sm text-danger">No runs started — launch a new batch from "Factorial Batch".</div>
  }

  return (
    <div className="max-w-2xl">
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/experiment/factorial" className="hover:text-primary">
          ← Back to Factorial Batch
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-2">Factorial Batch — {entries.length} run(s) launched</h1>

      {skipped.length > 0 && (
        <div className="mb-4 px-3 py-2.5 rounded-md border border-border bg-white text-text-secondary text-sm">
          Skipped (identical config already completed): {skipped.join(', ')}
        </div>
      )}

      {failures.length > 0 && (
        <div className="mb-4 px-3 py-2.5 rounded-md border border-danger/40 bg-white text-danger text-sm">
          ⚠ {failures.length} run(s) failed to start — the others below are still running normally.
          <ul className="mt-1 ml-4 list-disc">
            {failures.map((f) => (
              <li key={f.runLabel}>
                {f.runLabel}: {f.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-col gap-4">
        {entries.map((e) => (
          <RunCard key={e.runLabel} runLabel={e.runLabel} condition={e.condition} runId={e.runId} />
        ))}
      </div>
    </div>
  )
}
