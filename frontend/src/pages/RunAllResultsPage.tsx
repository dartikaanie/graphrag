import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useCancelRun, useRunStream } from '@/api/hooks'
import { ProgressRunPanel } from '@/components/ProgressRunPanel'
import type { RunCondition } from '@/types/run'

const LABELS: Record<RunCondition, string> = {
  A: 'A — Pure LLM',
  B: 'B — LLM + RAG',
  C: 'C — LLM + GraphRAG',
  D: 'D — Dual-Level Retrieval (LightRAG-adapted)',
}

function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(4)
}

function ConditionColumn({ condition, runId }: { condition: RunCondition; runId: string }) {
  const navigate = useNavigate()
  const { run, connectionError } = useRunStream(runId)
  const cancelRun = useCancelRun()

  if (!run) return <div className="text-sm text-text-muted">Loading {condition}...</div>

  const isTerminal = ['completed', 'failed', 'cancelled'].includes(run.status)

  return (
    <div className="border border-border rounded-lg bg-surface p-4 flex-1 min-w-0">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-text-primary">{LABELS[condition]}</h2>
        <Link to={`/experiment/${condition.toLowerCase()}/runs/${runId}`} className="text-xs text-primary hover:underline">
          Full view →
        </Link>
      </div>
      <p className="text-xs text-text-secondary mb-3">
        Status: <span className="font-medium">{run.status}</span>
      </p>

      {connectionError && !isTerminal && <div className="text-xs text-warning mb-2">Live stream disconnected.</div>}

      {!isTerminal && (
        <ProgressRunPanel
          current={run.progress.current}
          total={run.progress.total}
          results={run.results}
          onStop={() => cancelRun.mutate(run.run_id)}
          stopping={cancelRun.isPending || run.cancel_requested}
          status={run.status}
          queuePosition={run.queue_position}
        />
      )}

      {run.status === 'failed' && run.error && (
        <div className="text-xs text-danger border border-danger/40 bg-white rounded-md p-2">{run.error}</div>
      )}

      {isTerminal && run.summary && (
        <div className="flex flex-col gap-2 text-sm">
          <div className="flex justify-between">
            <span className="text-text-secondary">Avg Cosine Similarity</span>
            <span className="font-medium">{fmt(run.summary.cosine_similarity_mean)}</span>
          </div>
          {run.summary.pct_with_valid_citation !== undefined && (
            <div className="flex justify-between">
              <span className="text-text-secondary">NF2 (valid citation)</span>
              <span className="font-medium">{run.summary.pct_with_valid_citation.toFixed(1)}%</span>
            </div>
          )}
          <button
            onClick={() => navigate(`/experiment/${condition.toLowerCase()}/runs/${runId}`)}
            className="mt-1 px-3 py-1.5 text-xs border border-border-strong rounded-md text-text-primary hover:bg-bg"
          >
            View per-question results
          </button>
        </div>
      )}
    </div>
  )
}

export function RunAllResultsPage() {
  const [searchParams] = useSearchParams()
  const runIds: Partial<Record<RunCondition, string>> = {
    A: searchParams.get('a') ?? undefined,
    B: searchParams.get('b') ?? undefined,
    C: searchParams.get('c') ?? undefined,
    D: searchParams.get('d') ?? undefined,
  }
  // "condition:message|condition:message" -- set by RunAllConditionsPage
  // when Promise.allSettled found that SOME conditions failed to even
  // submit (e.g. a validation error), while the others started fine.
  const failedParam = searchParams.get('failed')
  const failures = failedParam
    ? failedParam.split('|').map((entry) => {
        const [condition, ...rest] = entry.split(':')
        return { condition: condition as RunCondition, message: rest.join(':') }
      })
    : []

  const startedConditions = (['A', 'B', 'C', 'D'] as const).filter((c) => runIds[c])

  if (startedConditions.length === 0) {
    return <div className="text-sm text-danger">No runs started — start a new comparison from "Run All Conditions".</div>
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/experiment/all" className="hover:text-primary">
          ← Back to Run All Conditions
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-2">Comparing Condition A / B / C / D</h1>

      {failures.length > 0 && (
        <div className="mb-4 px-3 py-2.5 rounded-md border border-danger/40 bg-white text-danger text-sm">
          ⚠ {failures.length} of 4 condition{failures.length > 1 ? 's' : ''} failed to start — the others below are
          still running normally.
          <ul className="mt-1 ml-4 list-disc">
            {failures.map((f) => (
              <li key={f.condition}>
                Condition {f.condition}: {f.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex gap-4 items-start">
        {startedConditions.map((c) => (
          <ConditionColumn key={c} condition={c} runId={runIds[c]!} />
        ))}
      </div>
    </div>
  )
}

