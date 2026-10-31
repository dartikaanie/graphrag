import { useSecondaryPanel } from '@/api/hooks'

const METRIC_TITLE: Record<string, string> = {
  ansrel: 'Answer Relevance',
  ctxrel: 'Context Relevance',
}

export function SecondaryPanelPage() {
  const { data, isLoading, isError } = useSecondaryPanel()
  const metrics = Object.values(data?.metrics ?? {})

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">Secondary Metrics Panel</h1>
      <p className="text-sm text-text-secondary mb-4">
        Official 3-judge panel results for the secondary metrics (read-only, from results/official_secondary_panel/).
        Judged in stages: a judge that has not finished shows fewer rows or a smaller N, and the combined panel
        verdict is only valid once all three judges are complete.
      </p>
      {isLoading && <div className="text-sm text-text-muted">Loading...</div>}
      {isError && <div className="text-sm text-danger">Failed to load secondary panel results.</div>}
      {data && metrics.length === 0 && (
        <div className="text-sm text-text-muted">No secondary panel results yet.</div>
      )}
      {metrics.map((m) => {
        const labels = Array.from(new Set(m.rows.flatMap((r) => Object.keys(r.labels)))).sort()
        const failures = Object.entries(m.failures_by_judge)
        return (
          <div key={m.metric} className="mb-8">
            <h2 className="text-sm font-medium text-text-secondary mb-1">
              {METRIC_TITLE[m.metric] ?? m.metric} ({m.metric})
            </h2>
            <p className="text-xs text-text-muted mb-2">
              Judges present: {m.judges.join(', ')} · total cost ${m.total_cost_usd.toFixed(4)}
              {failures.length > 0 && <> · unresolved failures: {failures.map(([j, c]) => `${j} ${c}`).join(', ')}</>}
            </p>
            <div className="overflow-x-auto border border-border rounded-lg">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-bg border-b border-border-strong text-text-secondary">
                    <th className="text-left font-medium px-3 py-2">Run file</th>
                    <th className="text-left font-medium px-3 py-2">Judge</th>
                    <th className="text-left font-medium px-3 py-2">N</th>
                    {labels.map((l) => <th key={l} className="text-left font-medium px-3 py-2 whitespace-nowrap">{l}</th>)}
                    <th className="text-left font-medium px-3 py-2">Mean score</th>
                    <th className="text-left font-medium px-3 py-2">Unparsed</th>
                    <th className="text-left font-medium px-3 py-2">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {m.rows.map((r) => (
                    <tr key={`${r.source_file}|${r.judge_id}`} className="border-b border-border last:border-b-0">
                      <td className="px-3 py-1.5 font-mono max-w-xs truncate" title={r.source_file}>{r.source_file}</td>
                      <td className="px-3 py-1.5" title={r.judge_model ?? ''}>{r.judge_id}</td>
                      <td className="px-3 py-1.5">{r.n}</td>
                      {labels.map((l) => {
                        const c = r.labels[l] ?? 0
                        return <td key={l} className="px-3 py-1.5 whitespace-nowrap">{c} ({r.n ? ((c / r.n) * 100).toFixed(1) : '—'}%)</td>
                      })}
                      <td className="px-3 py-1.5">{r.mean_score != null ? r.mean_score.toFixed(3) : '—'}</td>
                      <td className="px-3 py-1.5">{r.n_unparsed}</td>
                      <td className="px-3 py-1.5">${r.cost_usd.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}
