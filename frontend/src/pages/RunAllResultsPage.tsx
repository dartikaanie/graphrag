import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { useCancelRun, useRunStream } from '@/api/hooks'
import { ProgressRunPanel } from '@/components/ProgressRunPanel'
import type { RunCondition } from '@/types/run'

const LABELS: Record<RunCondition, string> = { A: 'Condition A — Pure LLM', B: 'Condition B — LLM + RAG', C: 'Condition C — LLM + GraphRAG' }

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
  }

  if (!runIds.A || !runIds.B || !runIds.C) {
    return <div className="text-sm text-danger">Missing run ids — start a new comparison from "Run All Conditions".</div>
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/experiment/all" className="hover:text-primary">
          ← Back to Run All Conditions
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Comparing Condition A / B / C</h1>

      <div className="flex gap-4 items-start">
        <ConditionColumn condition="A" runId={runIds.A} />
        <ConditionColumn condition="B" runId={runIds.B} />
        <ConditionColumn condition="C" runId={runIds.C} />
      </div>
    </div>
  )
}

