import { useState } from 'react'
import { useJudgeVersionComparison, useReferenceConflicts } from '@/api/hooks'

/** Step 5: a v1-vs-v2 comparison view for the same runs/judge model --
 * a label transition matrix (v1 label -> v2 label) per run_label, and
 * kappa(primary, secondary) for v1 vs v2 side by side, plus a
 * "Reference conflicts" list for human review. Reuses the currently-
 * selected run_ids from the run picker above it on the page -- shows
 * a hint instead of querying when nothing is selected. */
export function JudgeVersionComparisonSection({ selectedRunIds }: { selectedRunIds: string[] }) {
  const [judgeIdA, setJudgeIdA] = useState('primary')
  const [judgeIdB, setJudgeIdB] = useState('secondary')

  const { data: comparison } = useJudgeVersionComparison(selectedRunIds, judgeIdA, judgeIdB)
  const { data: conflictsData } = useReferenceConflicts(selectedRunIds, ['primary', 'secondary'])

  return (
    <div className="border border-border rounded-lg bg-surface p-4 mb-4">
      <h2 className="text-sm font-semibold text-text-primary mb-2">v1 vs v2 comparison</h2>
      {selectedRunIds.length === 0 ? (
        <div className="text-sm text-text-muted">Select runs above to compare judge-v1 and judge-v2 on them.</div>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-3 text-xs">
            <span className="text-text-secondary">Transition matrix for</span>
            <select value={judgeIdA} onChange={(e) => setJudgeIdA(e.target.value)} className="px-1.5 py-1 border border-border-strong rounded-md">
              <option value="primary">primary</option>
              <option value="secondary">secondary</option>
            </select>
            <span className="text-text-secondary">· κ(primary, secondary) vs</span>
            <select value={judgeIdB} onChange={(e) => setJudgeIdB(e.target.value)} className="px-1.5 py-1 border border-border-strong rounded-md">
              <option value="secondary">secondary</option>
              <option value="primary">primary</option>
            </select>
          </div>

          {comparison && (
            <>
              <div className="flex items-center gap-4 mb-3 text-xs">
                <div>
                  <span className="font-medium">κ(v1):</span>{' '}
                  {comparison.kappa_v1.insufficient_data ? 'insufficient data' :
                    `weighted=${comparison.kappa_v1.kappas?.weighted ?? '—'}, unweighted=${comparison.kappa_v1.kappas?.unweighted ?? '—'} (n=${comparison.kappa_v1.n_pairs})`}
                </div>
                <div>
                  <span className="font-medium">κ(v2):</span>{' '}
                  {comparison.kappa_v2.insufficient_data ? 'insufficient data' :
                    `weighted=${comparison.kappa_v2.kappas?.weighted ?? '—'}, unweighted=${comparison.kappa_v2.kappas?.unweighted ?? '—'} (n=${comparison.kappa_v2.n_pairs})`}
                </div>
              </div>

              {Object.keys(comparison.transition_matrix).length === 0 ? (
                <div className="text-sm text-text-muted mb-3">
                  No (run_id, question_id) overlap between v1 and v2 records for these runs yet.
                </div>
              ) : (
                Object.entries(comparison.transition_matrix).map(([runLabel, fromLabels]) => {
                  const toLabels = Array.from(new Set(Object.values(fromLabels).flatMap((row) => Object.keys(row))))
                  const fromLabelKeys = Object.keys(fromLabels)
                  return (
                    <div key={runLabel} className="mb-3">
                      <div className="text-xs font-medium text-text-primary mb-1">{runLabel}</div>
                      <table className="text-xs border border-border rounded-md">
                        <thead>
                          <tr className="bg-bg border-b border-border">
                            <th className="px-2 py-1 text-left">v1 \ v2</th>
                            {toLabels.map((l) => <th key={l} className="px-2 py-1 text-left">{l}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {fromLabelKeys.map((fromLabel) => (
                            <tr key={fromLabel} className="border-b border-border/50 last:border-b-0">
                              <td className="px-2 py-1 font-medium">{fromLabel}</td>
                              {toLabels.map((toLabel) => (
                                <td key={toLabel} className="px-2 py-1">{fromLabels[fromLabel][toLabel] ?? 0}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )
                })
              )}
            </>
          )}

          <h3 className="text-sm font-medium text-text-secondary mt-4 mb-2">Reference conflicts (for human review)</h3>
          <div className="overflow-x-auto border border-border rounded-lg">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-text-secondary border-b border-border bg-bg">
                  <th className="px-3 py-2">question_id</th>
                  <th className="px-3 py-2">run_label</th>
                  <th className="px-3 py-2">judge</th>
                  <th className="px-3 py-2">claim</th>
                  <th className="px-3 py-2">evidence</th>
                </tr>
              </thead>
              <tbody>
                {(conflictsData?.conflicts ?? []).map((c, i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="px-3 py-1.5">{c.question_id}</td>
                    <td className="px-3 py-1.5">{c.run_label}</td>
                    <td className="px-3 py-1.5">{c.judge_id}</td>
                    <td className="px-3 py-1.5">{c.claim}</td>
                    <td className="px-3 py-1.5">{c.evidence}</td>
                  </tr>
                ))}
                {(conflictsData?.conflicts ?? []).length === 0 && (
                  <tr><td colSpan={5} className="px-3 py-2 text-text-muted">No reference conflicts flagged for these runs yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
