import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  useHistoryBatches,
  useJudgeV1Agreement,
  useJudgeV1Disagreements,
  useJudgeV1Jobs,
  useJudgeV1Summary,
  type JudgeV1Disagreement,
  type JudgeV1RunLabelSummary,
} from '@/api/hooks'
import { JudgeV1ResultsSection } from '@/components/JudgeV1ResultsSection'
import { downloadMarkdown, mdTable } from '@/lib/markdown'

/** Same canonical factorial order as FactorialResultsPage.tsx's
 * FACTORIAL_ORDER -- duplicated here (not imported) since that page's
 * constant isn't exported and this is the only other place that needs
 * it; if a run_label axis is ever added, both lists must be updated
 * together. */
const FACTORIAL_ORDER = [
  'A', 'B-plain', 'B-grounded',
  'C-uniform-plain', 'C-uniform-grounded', 'C-trust-plain', 'C-trust-grounded',
  'D-plain', 'D-grounded',
]

function fmtJakarta(iso: string | null): string {
  if (!iso) return 'Unknown time'
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Jakarta', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(new Date(iso))
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? ''
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')} (Asia/Jakarta)`
}

function buildJudgeV1ReportMarkdown(opts: {
  batchId: string
  launchedAt: string | null
  runLabels: string[]
  primarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
  secondarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
  agreement: {
    n_pairs: number
    kappas: { weighted: number | null; unweighted: number | null; n_used: number; n_excluded: number } | null
    confusion_matrix: Record<string, Record<string, number>> | null
    insufficient_data: boolean
  } | undefined
  disagreements: JudgeV1Disagreement[]
}): string {
  const { batchId, launchedAt, runLabels, primarySummary, secondarySummary, agreement, disagreements } = opts

  const summaryHeader = [
    'run_label',
    'primary N', 'primary FAKTUAL', 'primary SEBAGIAN', 'primary PENUH', 'primary ABSTAIN',
    'primary Abstain%', 'primary Hall(excl)', 'primary Hall(incl)', 'primary Parse-err%', 'primary Consist%',
    'secondary N', 'secondary FAKTUAL', 'secondary SEBAGIAN', 'secondary PENUH', 'secondary ABSTAIN',
    'secondary Abstain%', 'secondary Hall(excl)', 'secondary Hall(incl)', 'secondary Parse-err%', 'secondary Consist%',
  ]
  const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
  const rate = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3))
  const row = (s: JudgeV1RunLabelSummary | undefined) => [
    String(s?.n ?? '—'),
    String(s?.label_distribution?.FAKTUAL ?? 0),
    String(s?.label_distribution?.HALUSINASI_SEBAGIAN ?? 0),
    String(s?.label_distribution?.HALUSINASI_PENUH ?? 0),
    String(s?.label_distribution?.ABSTAIN ?? 0),
    pct(s?.abstention_rate), rate(s?.hall_rate_excl_abstain), rate(s?.hall_rate_incl_abstain),
    pct(s?.parse_error_rate), pct(s?.consistency_rate),
  ]
  const summaryRows = runLabels.map((label) => [
    label, ...row(primarySummary?.[label]), ...row(secondarySummary?.[label]),
  ])

  const agreementLines: string[] = []
  if (agreement?.kappas) {
    agreementLines.push(
      `n pairs = ${agreement.n_pairs}`,
      `weighted κ = ${agreement.kappas.weighted ?? '—'}`,
      `unweighted κ = ${agreement.kappas.unweighted ?? '—'}`,
      '',
    )
    if (agreement.confusion_matrix) {
      const labels = Object.keys(agreement.confusion_matrix)
      agreementLines.push(
        mdTable(['primary \\ secondary', ...labels], labels.map((rowLabel) => [
          rowLabel, ...labels.map((colLabel) => String(agreement.confusion_matrix![rowLabel]?.[colLabel] ?? 0)),
        ])),
      )
    }
  } else {
    agreementLines.push('Insufficient paired data for primary-vs-secondary agreement.')
  }

  const disagreementHeader = ['question_id', 'run_label', 'primary label', 'secondary label', 'primary reasoning', 'secondary reasoning']
  const disagreementRows = disagreements.map((d) => [
    String(d.question_id ?? '—'),
    String(d.run_label ?? '—'),
    String(d.primary_label ?? '—'),
    String(d.secondary_label ?? '—'),
    String(d.primary_reasoning ?? '—'),
    String(d.secondary_reasoning ?? '—'),
  ])

  const lines = [
    '# Judge-v1 Results Report',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Batch: ${batchId}`,
    `Launched: ${fmtJakarta(launchedAt)}`,
    `Run labels covered: ${runLabels.join(', ')}`,
    '',
    '## Summary (primary & secondary, per run_label)',
    '',
    mdTable(summaryHeader, summaryRows),
    '',
    '## Agreement (primary vs secondary)',
    '',
    ...agreementLines,
    '',
    '## Disagreements (primary vs secondary)',
    '',
    mdTable(disagreementHeader, disagreementRows),
    '',
  ]
  return lines.join('\n')
}

export function JudgeV1ResultsPage() {
  const { batchId } = useParams<{ batchId: string }>()
  const { data: batchesData } = useHistoryBatches(true)
  const batch = batchesData?.batches.find((b) => b.batch_id === batchId)

  const { data: jobsData } = useJudgeV1Jobs()
  const jobsForBatch = useMemo(
    () => (jobsData?.jobs ?? []).filter((j) => j.batch_id === batchId),
    [jobsData, batchId],
  )

  const runLabels = useMemo(() => {
    const present = new Set((batch?.runs ?? []).map((r) => r.run_label).filter((l): l is string => !!l))
    return FACTORIAL_ORDER.filter((l) => present.has(l))
  }, [batch])

  const { data: primarySummary } = useJudgeV1Summary('primary')
  const { data: secondarySummary } = useJudgeV1Summary('secondary')
  const { data: agreement } = useJudgeV1Agreement('primary', 'secondary')
  const { data: disagreementsData } = useJudgeV1Disagreements('primary', 'secondary')

  const disagreementsForBatch = useMemo(
    () => (disagreementsData?.disagreements ?? []).filter((d) => runLabels.includes(d.run_label ?? '')),
    [disagreementsData, runLabels],
  )

  const handleExport = () => {
    const md = buildJudgeV1ReportMarkdown({
      batchId: batchId ?? 'unknown',
      launchedAt: batch?.launched_at ?? null,
      runLabels,
      primarySummary: primarySummary?.summary,
      secondarySummary: secondarySummary?.summary,
      agreement,
      disagreements: disagreementsForBatch,
    })
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    downloadMarkdown(`judge-v1-report_${batchId}_${stamp}.md`, md)
  }

  if (!batchId) {
    return <div className="text-sm text-text-muted">No batch specified.</div>
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/evaluation/judge-v1" className="hover:text-primary">
          ← Back to Hallucination Judge
        </Link>
      </div>
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-lg font-semibold text-text-primary">Judge-v1 Results</h1>
        <button
          onClick={handleExport}
          className="px-3 py-1.5 text-sm border border-border-strong rounded-md text-text-primary hover:bg-bg"
        >
          ↓ Download report (.md)
        </button>
      </div>
      <p className="text-sm text-text-secondary mb-4">
        Batch {batchId} — launched {fmtJakarta(batch?.launched_at ?? null)} · {runLabels.length} run label(s)
      </p>

      {!batch && <div className="text-sm text-text-muted mb-4">This batch wasn't found (it may have been superseded or removed).</div>}

      {jobsForBatch.length > 0 && (
        <div className="border border-border rounded-lg bg-surface p-3 mb-4 text-xs text-text-secondary">
          <div className="font-medium text-text-primary mb-1">Judge jobs covering this batch</div>
          {jobsForBatch.map((j) => (
            <div key={j.job_id} className="flex items-center gap-2">
              <span className="font-medium">{j.judge}</span>
              <span>judged: {j.judged ?? '—'}</span>
              <span>failed: {j.failed ?? '—'}</span>
              {j.output_file_missing && <span className="text-danger">output file missing</span>}
              {j.served_model_mismatch_count > 0 && (
                <span className="text-warning">{j.served_model_mismatch_count} served-model mismatch(es)</span>
              )}
            </div>
          ))}
        </div>
      )}

      {runLabels.length === 0 ? (
        <div className="text-sm text-text-muted">No judge-v1 results found for this batch's run_labels yet.</div>
      ) : (
        <JudgeV1ResultsSection runLabels={runLabels} />
      )}
    </div>
  )
}
