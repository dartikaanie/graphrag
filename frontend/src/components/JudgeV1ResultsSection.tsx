import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  useJudgeV1Agreement,
  useJudgeV1Disagreements,
  useJudgeV1Summary,
  type JudgeV1RunLabelSummary,
} from '@/api/hooks'

/** "Hallucination (reference-based, judge-v1)" -- a SEPARATE metric from
 * the legacy Faithfulness table (different judge, different question:
 * candidate vs. the accepted answer + judge's own knowledge, not vs.
 * retrieved_context). Official numbers come from whichever judge_id
 * currently holds the "primary" role; other judges (fallback/
 * exploratory) are shown in their own block, never mixed into the
 * primary row.
 *
 * Shared between HistoryComparePage (Run Comparison) and
 * JudgeV1ResultsPage (per-batch judge results) -- same summary table +
 * agreement picker + filterable disagreement list in both places, so
 * they can never silently drift apart. */
export const JUDGE_ID_CHOICES = ['primary', 'secondary', 'fallback', 'exploratory']

export function JudgeV1ResultsSection({ runLabels }: { runLabels: string[] }) {
  const { data: primarySummary } = useJudgeV1Summary('primary')
  const { data: secondarySummary } = useJudgeV1Summary('secondary')

  const [judgeIdA, setJudgeIdA] = useState('primary')
  const [judgeIdB, setJudgeIdB] = useState('secondary')
  const { data: agreement } = useJudgeV1Agreement(judgeIdA, judgeIdB)

  const [filterRunLabel, setFilterRunLabel] = useState('')
  const [filterLabelA, setFilterLabelA] = useState('')
  const [filterLabelB, setFilterLabelB] = useState('')
  const { data: disagreementsData } = useJudgeV1Disagreements(
    judgeIdA, judgeIdB, filterRunLabel || undefined, filterLabelA || undefined, filterLabelB || undefined,
  )

  const relevantLabels = runLabels.filter(
    (l) => primarySummary?.summary?.[l] || secondarySummary?.summary?.[l],
  )
  if (relevantLabels.length === 0) return null

  return (
    <div className="mt-6">
      <h2 className="text-sm font-medium text-text-secondary mb-2">Hallucination (reference-based, judge-v1)</h2>
      <p className="text-xs text-text-muted mb-2">
        Official numbers come from the judge model currently holding the "primary" role. Candidate vs. the
        accepted answer -- a different metric from the Faithfulness table (candidate vs. retrieved context).
      </p>
      <JudgeV1SummaryTable runLabels={relevantLabels} primarySummary={primarySummary?.summary} secondarySummary={secondarySummary?.summary} />

      <div className="flex items-center gap-2 mb-2 mt-3 text-xs">
        <span className="text-text-secondary">Agreement between</span>
        <select value={judgeIdA} onChange={(e) => setJudgeIdA(e.target.value)} className="px-1.5 py-1 border border-border-strong rounded-md">
          {JUDGE_ID_CHOICES.map((j) => <option key={j} value={j}>{j}</option>)}
        </select>
        <span className="text-text-secondary">and</span>
        <select value={judgeIdB} onChange={(e) => setJudgeIdB(e.target.value)} className="px-1.5 py-1 border border-border-strong rounded-md">
          {JUDGE_ID_CHOICES.map((j) => <option key={j} value={j}>{j}</option>)}
        </select>
      </div>

      {agreement && !agreement.insufficient_data && agreement.kappas && (
        <div className="text-xs text-text-secondary mb-4">
          n={agreement.n_pairs} pairs, weighted κ={agreement.kappas.weighted}, unweighted κ={agreement.kappas.unweighted}
        </div>
      )}
      {agreement?.insufficient_data && (
        <div className="text-xs text-text-muted mb-4">Insufficient paired data for {judgeIdA}-vs-{judgeIdB} agreement yet.</div>
      )}

      <h3 className="text-sm font-medium text-text-secondary mb-2">Disagreements ({judgeIdA} vs {judgeIdB})</h3>
      <div className="flex items-center gap-2 mb-2 text-xs">
        <select value={filterRunLabel} onChange={(e) => setFilterRunLabel(e.target.value)} className="px-1.5 py-1 border border-border-strong rounded-md">
          <option value="">All run_labels</option>
          {relevantLabels.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
        <input
          placeholder={`${judgeIdA} label`} value={filterLabelA} onChange={(e) => setFilterLabelA(e.target.value)}
          className="px-1.5 py-1 border border-border-strong rounded-md w-40"
        />
        <input
          placeholder={`${judgeIdB} label`} value={filterLabelB} onChange={(e) => setFilterLabelB(e.target.value)}
          className="px-1.5 py-1 border border-border-strong rounded-md w-40"
        />
      </div>
      <div className="overflow-x-auto border border-border rounded-lg">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-text-secondary border-b border-border bg-bg">
              <th className="px-3 py-2">run_label</th>
              <th className="px-3 py-2">question_id</th>
              <th className="px-3 py-2">{judgeIdA} label</th>
              <th className="px-3 py-2">{judgeIdB} label</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {(disagreementsData?.disagreements ?? []).map((d, i) => (
              <tr key={i} className="border-b border-border/50">
                <td className="px-3 py-1.5">{d.run_label}</td>
                <td className="px-3 py-1.5">{d.question_id}</td>
                <td className="px-3 py-1.5">{d[`${judgeIdA}_label`] ?? '—'}</td>
                <td className="px-3 py-1.5">{d[`${judgeIdB}_label`] ?? '—'}</td>
                <td className="px-3 py-1.5">
                  <Link to={`/history/${d.run_id}/q/${d.question_id}`} className="text-primary hover:underline">
                    detail →
                  </Link>
                </td>
              </tr>
            ))}
            {(disagreementsData?.disagreements ?? []).length === 0 && (
              <tr><td colSpan={5} className="px-3 py-2 text-text-muted">No disagreements found for this filter.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function fmtPct(v: number | null): string {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`
}

function fmtRate(v: number | null): string {
  return v == null ? '—' : v.toFixed(3)
}

/** Per-run_label summary, primary and secondary side by side -- full
 * FAKTUAL/SEBAGIAN/PENUH/ABSTAIN counts+%, both hallucination rates,
 * parse-error and consistency rates. Used standalone by
 * JudgeV1ResultsPage (which doesn't need the agreement/disagreement
 * panels above it) and internally by JudgeV1ResultsSection. */
export function JudgeV1SummaryTable({
  runLabels, primarySummary, secondarySummary,
}: {
  runLabels: string[]
  primarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
  secondarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
}) {
  const LABELS = ['FAKTUAL', 'HALUSINASI_SEBAGIAN', 'HALUSINASI_PENUH', 'ABSTAIN']

  const renderJudgeCols = (s: JudgeV1RunLabelSummary | undefined) => (
    <>
      <td className="px-3 py-2">{s?.n ?? '—'}</td>
      {LABELS.map((l) => {
        const n = s?.label_distribution?.[l] ?? 0
        const pct = s?.n ? n / s.n : null
        return <td key={l} className="px-3 py-2 whitespace-nowrap">{s ? `${n} (${fmtPct(pct)})` : '—'}</td>
      })}
      <td className="px-3 py-2">{s ? fmtPct(s.abstention_rate) : '—'}</td>
      <td className="px-3 py-2">{s ? fmtRate(s.hall_rate_excl_abstain) : '—'}</td>
      <td className="px-3 py-2">{s ? fmtRate(s.hall_rate_incl_abstain) : '—'}</td>
      <td className="px-3 py-2">{s ? fmtPct(s.parse_error_rate) : '—'}</td>
      <td className="px-3 py-2">{s ? fmtPct(s.consistency_rate) : '—'}</td>
    </>
  )

  return (
    <div className="overflow-x-auto border border-border rounded-lg mb-3">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-bg border-b border-border-strong text-text-secondary">
            <th rowSpan={2} className="text-left font-medium px-3 py-2 align-bottom">run_label</th>
            <th colSpan={8} className="font-medium px-3 py-1.5 text-center border-l border-border">primary</th>
            <th colSpan={8} className="font-medium px-3 py-1.5 text-center border-l border-border">secondary</th>
          </tr>
          <tr className="bg-bg border-b border-border-strong text-text-secondary">
            {[0, 1].map((side) => (
              <>
                <th className={`text-left font-medium px-3 py-1.5 ${side === 1 ? 'border-l border-border' : ''}`} key={`n-${side}`}>N</th>
                {LABELS.map((l) => <th key={`${l}-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">{l}</th>)}
                <th key={`abst-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">Abstain%</th>
                <th key={`hexcl-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">Hall(excl)</th>
                <th key={`hincl-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">Hall(incl)</th>
                <th key={`perr-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">Parse-err%</th>
                <th key={`cons-${side}`} className="text-left font-medium px-3 py-1.5 whitespace-nowrap">Consist.%</th>
              </>
            ))}
          </tr>
        </thead>
        <tbody>
          {runLabels.map((label) => (
            <tr key={label} className="border-b border-border last:border-b-0">
              <td className="px-3 py-2 font-medium text-text-primary">{label}</td>
              {renderJudgeCols(primarySummary?.[label])}
              {renderJudgeCols(secondarySummary?.[label])}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
