import { Link, useSearchParams } from 'react-router-dom'
import { useHistoryDetail } from '@/api/hooks'
import type { RunState } from '@/types/run'

interface MetricRow {
  label: string
  get: (run: RunState) => number | null | undefined
  fmt: (v: number) => string
  higherIsBetter: boolean
}

const ROWS: MetricRow[] = [
  { label: 'N processed', get: (r) => r.summary?.n_processed, fmt: (v) => String(v), higherIsBetter: true },
  { label: 'Avg Cosine Similarity', get: (r) => r.summary?.cosine_similarity_mean, fmt: (v) => v.toFixed(4), higherIsBetter: true },
  { label: '% Similarity > 0.5', get: (r) => r.summary?.pct_similarity_above_0_5, fmt: (v) => `${v.toFixed(1)}%`, higherIsBetter: true },
  { label: 'NF2 — Valid Citation %', get: (r) => r.summary?.pct_with_valid_citation, fmt: (v) => `${v.toFixed(1)}%`, higherIsBetter: true },
  { label: 'Avg Retrieval Latency (s)', get: (r) => r.summary?.avg_retrieval_latency_sec, fmt: (v) => v.toFixed(2), higherIsBetter: false },
]

export function HistoryComparePage() {
  const [searchParams] = useSearchParams()
  const ids = (searchParams.get('ids') ?? '').split(',').filter(Boolean)

  // Hooks must run unconditionally in the same order every render, so we
  // call useHistoryDetail once per fixed slot (max 3, per HistoryPage's
  // selection cap) rather than inside a variable-length loop.
  const q0 = useHistoryDetail(ids[0] ?? '')
  const q1 = useHistoryDetail(ids[1] ?? '')
  const q2 = useHistoryDetail(ids[2] ?? '')
  const runs = [q0.data, q1.data, q2.data].filter((r): r is RunState => !!r)

  if (ids.length === 0) {
    return <div className="text-sm text-text-muted">No runs selected. Go back to History and select 2-3 runs.</div>
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/history" className="hover:text-primary">
          ← Back to History
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Compare Runs</h1>

      {runs.length < ids.length && <div className="text-sm text-text-muted mb-4">Loading...</div>}

      {runs.length > 0 && (
        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-bg border-b border-border-strong">
                <th className="text-left font-medium text-text-secondary px-4 py-2.5">Metric</th>
                {runs.map((r) => (
                  <th key={r.run_id} className="text-left font-medium text-text-secondary px-4 py-2.5">
                    Condition {r.condition}
                    <div className="text-xs font-normal text-text-muted">{r.run_id}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => {
                const values = runs.map((r) => row.get(r))
                const numeric = values.filter((v): v is number => typeof v === 'number')
                const best = numeric.length > 0 ? (row.higherIsBetter ? Math.max(...numeric) : Math.min(...numeric)) : null

                return (
                  <tr key={row.label} className="border-b border-border last:border-b-0">
                    <td className="px-4 py-2.5 text-text-secondary">{row.label}</td>
                    {values.map((v, i) => {
                      const isBest = typeof v === 'number' && best !== null && v === best && numeric.length > 1
                      return (
                        <td
                          key={runs[i].run_id}
                          className={`px-4 py-2.5 ${isBest ? 'font-semibold bg-success/10 text-success' : 'text-text-primary'}`}
                        >
                          {typeof v === 'number' ? row.fmt(v) : '—'}
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
