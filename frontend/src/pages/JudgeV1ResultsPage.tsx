import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  fetchJudgeV1QuestionDetail,
  useContextRelevanceBatchSummary,
  useContextRelevanceLinkToOutcomesMulti,
  useHistoryBatches,
  useHistoryDetail,
  useJudgeV1Agreement,
  useJudgeV1Disagreements,
  useJudgeV1Jobs,
  useJudgeV1Summary,
  type ContextRelevanceBatchSummaryRow,
  type ContextRelevanceLinkTables,
  type JudgeV1Disagreement,
  type JudgeV1QuestionDetail,
  type JudgeV1RunLabelSummary,
} from '@/api/hooks'
import { JudgeV1ResultsSection } from '@/components/JudgeV1ResultsSection'
import { downloadMarkdown, mdTable } from '@/lib/markdown'
import type { HistoryItem } from '@/types/history'

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

/** Shared by both the batch report export and (via a thin wrapper) the
 * Run Comparison export -- a markdown table of the context-relevance
 * batch-summary rows, plus a markdown rendering of each run's link-to-
 * outcomes contingency tables. `linkRows` is indexed in the SAME order
 * as the non-no-context rows of `ctxrelRows` (i.e. one entry per run
 * that has a run_id, skipping A/no-context rows). */
function buildContextRelevanceMarkdown(
  ctxrelRows: ContextRelevanceBatchSummaryRow[],
  linkRows: Array<{ runLabel: string; tables: ContextRelevanceLinkTables | undefined }>,
): string[] {
  const pct = (v: number | null | undefined) => (v == null ? '—' : `${v.toFixed(1)}%`)
  const header = [
    'run_label', 'items judged', '% RELEVANT', '% PARTIAL', '% IRRELEVANT',
    'Questions w/ ≥1 RELEVANT', 'Mean score', 'Mean score (rank 1)', 'Mean score (ranks 2-5)',
  ]
  const rows = ctxrelRows.map((r) => {
    if (r.no_context) return [r.run_label, 'n/a — no context', '—', '—', '—', '—', '—', '—', '—']
    const m = r.metrics
    return [
      r.run_label, String(m?.n_items_judged ?? '—'), pct(m?.pct_relevant), pct(m?.pct_partial), pct(m?.pct_irrelevant),
      pct(m?.pct_questions_with_relevant),
      m?.mean_relevance_score != null ? m.mean_relevance_score.toFixed(3) : '—',
      m?.mean_relevance_score_rank1 != null ? m.mean_relevance_score_rank1.toFixed(3) : '—',
      m?.mean_relevance_score_rank2plus != null ? m.mean_relevance_score_rank2plus.toFixed(3) : '—',
    ]
  })

  const lines = [mdTable(header, rows), '']

  if (linkRows.length > 0) {
    lines.push('### Link to outcomes (hallucination label × has ≥1 RELEVANT context item)', '')
    for (const { runLabel, tables } of linkRows) {
      if (!tables) continue
      lines.push(`**${runLabel}**`, '')
      const labelHeader = ['hallucination label', 'has ≥1 relevant', 'no relevant']
      const labelRows = Object.entries(tables.label_x_relevant).map(([label, counts]) => [
        label, String(counts.has_relevant), String(counts.no_relevant),
      ])
      lines.push(mdTable(labelHeader, labelRows), '')
      if (tables.abstain_x_relevant) {
        const a = tables.abstain_x_relevant
        lines.push(
          mdTable(['ABSTAIN × has ≥1 relevant', 'has ≥1 relevant', 'no relevant'], [
            ['abstain', String(a.has_relevant.abstain), String(a.no_relevant.abstain)],
            ['not abstain', String(a.has_relevant.not_abstain), String(a.no_relevant.not_abstain)],
          ]),
          '',
        )
      }
    }
  }

  return lines
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
  contextRelevance: ContextRelevanceBatchSummaryRow[]
  contextRelevanceHasResults: boolean
  linkRows: Array<{ runLabel: string; tables: ContextRelevanceLinkTables | undefined }>
}): string {
  const {
    batchId, launchedAt, runLabels, primarySummary, secondarySummary, agreement, disagreements,
    contextRelevance, contextRelevanceHasResults, linkRows,
  } = opts

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
    '## Context relevance (primary judge)',
    '',
    ...(contextRelevanceHasResults
      ? buildContextRelevanceMarkdown(contextRelevance, linkRows)
      : ['(no ctxrel results found for these runs yet)', '']),
  ]
  return lines.join('\n')
}

/** Step 2.3's per-question detail export -- one section per question,
 * never a table (the content per question is too varied/long for
 * table cells). Reuses mdEscape-safe plain text throughout via
 * JSON.stringify for claims (small, structured) and plain lines for
 * everything else. */
function buildQuestionDetailMarkdown(runId: string, runLabel: string | null, questions: JudgeV1QuestionDetail[]): string {
  const lines = [
    '# Per-Question Detail Export',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Run: ${runLabel ?? '—'} (${runId})`,
    `Questions: ${questions.length}`,
    '',
  ]

  for (const q of questions) {
    lines.push(`## Question ${q.question_id}`, '')
    if (q.error) {
      lines.push(`_${q.error}_`, '')
      continue
    }
    lines.push(
      `**Title:** ${q.title}`,
      `**Tags:** ${q.tags}`,
      '',
      '**Body:**', '', q.body ?? '', '',
      '**Reference answer:**', '', q.reference_answer ?? '', '',
      '**Candidate (raw):**', '', q.candidate_raw ?? '', '',
      '**Candidate (blinded, as seen by judge):**', '', q.candidate_blinded ?? '', '',
      `**Citation outcome:** ${q.citation_outcome}`,
      '',
    )

    if (q.context_items && q.context_items.length > 0) {
      lines.push('**Retrieved context:**', '')
      const header = ['rank', 'SO qid', 'answer_id', 'accepted', 'trust', 'hop', 'stage', 'score', 'text', 'ctx relevance']
      const rows = q.context_items.map((c) => [
        String(c.rank), String(c.so_question_id ?? '—'), String(c.answer_id ?? '—'),
        String(c.is_accepted ?? '—'), String(c.trust_weight ?? '—'), String(c.hop ?? '—'),
        String(c.source_stage ?? '—'), String(c.combined_score ?? c.score ?? '—'), c.chunk_text,
        Object.entries(c.context_relevance).map(([jid, v]) => `${jid}: ${v.label ?? '—'}`).join('; ') || '—',
      ])
      lines.push(mdTable(header, rows), '')
    }

    if (q.judges && Object.keys(q.judges).length > 0) {
      lines.push('**Judges:**', '')
      for (const [judgeId, j] of Object.entries(q.judges)) {
        lines.push(
          `- ${judgeId}: label=${j.label ?? '—'}, derived=${j.derived_label ?? '—'}, consistent=${j.consistent ?? '—'}, ` +
          `served_model=${j.served_model ?? '—'}, parse_error=${j.parse_error ?? false}`,
          `  reasoning: ${j.reasoning ?? '—'}`,
        )
        if (j.claims.length > 0) {
          lines.push(`  claims: ${JSON.stringify(j.claims)}`)
        }
      }
      lines.push('')
    }
  }

  return lines.join('\n')
}

/** Step 2.3's export UI -- export selected questions, all questions of
 * a run_label, or all disagreement questions, plus the "Export C-trust-
 * grounded (all 10)" quick action. */
function PerQuestionExportSection({
  runs, disagreements,
}: {
  runs: HistoryItem[]
  disagreements: JudgeV1Disagreement[]
}) {
  const [selectedRunLabel, setSelectedRunLabel] = useState(runs[0]?.run_label ?? '')
  const [questionIdsInput, setQuestionIdsInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedRun = runs.find((r) => r.run_label === selectedRunLabel)
  const { data: selectedRunDetail } = useHistoryDetail(selectedRun?.history_id ?? '')
  const trustGroundedRun = runs.find((r) => r.run_label === 'C-trust-grounded')
  const { data: trustGroundedDetail } = useHistoryDetail(trustGroundedRun?.history_id ?? '')

  const exportQuestions = async (runId: string, runLabel: string | null, questionIds: number[]) => {
    if (questionIds.length === 0) {
      setError('No questions to export for this selection.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const resp = await fetchJudgeV1QuestionDetail(runId, questionIds)
      const md = buildQuestionDetailMarkdown(runId, runLabel, resp.questions)
      const stamp = new Date().toISOString().replace(/[:.]/g, '-')
      downloadMarkdown(`question-detail_${runLabel ?? runId}_${stamp}.md`, md)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const handleExportSelected = () => {
    if (!selectedRun) return
    const ids = questionIdsInput.split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n))
    void exportQuestions(selectedRun.history_id, selectedRun.run_label ?? null, ids)
  }

  const handleExportAllOfRunLabel = () => {
    if (!selectedRun) return
    const ids = (selectedRunDetail?.results ?? []).map((r) => r.question_id)
    void exportQuestions(selectedRun.history_id, selectedRun.run_label ?? null, ids)
  }

  const handleExportDisagreements = () => {
    if (!selectedRun) return
    const ids = disagreements.filter((d) => d.run_label === selectedRunLabel).map((d) => d.question_id)
    void exportQuestions(selectedRun.history_id, selectedRun.run_label ?? null, ids)
  }

  return (
    <div className="mt-6 border border-border rounded-lg bg-surface p-4">
      <h2 className="text-sm font-medium text-text-secondary mb-2">Per-question detail export</h2>
      <div className="flex items-center gap-2 mb-3 text-xs">
        <span className="text-text-secondary">Run:</span>
        <select
          value={selectedRunLabel} onChange={(e) => setSelectedRunLabel(e.target.value)}
          className="px-1.5 py-1 border border-border-strong rounded-md"
        >
          {runs.map((r) => <option key={r.history_id} value={r.run_label ?? r.condition}>{r.run_label ?? r.condition}</option>)}
        </select>
      </div>
      <div className="flex items-center gap-2 mb-3 text-xs">
        <input
          placeholder="question ids, comma-separated" value={questionIdsInput}
          onChange={(e) => setQuestionIdsInput(e.target.value)}
          className="px-1.5 py-1 border border-border-strong rounded-md w-64"
        />
        <button onClick={handleExportSelected} disabled={busy} className="px-2.5 py-1 border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50">
          Export selected questions
        </button>
      </div>
      <div className="flex items-center gap-2 mb-3 text-xs">
        <button onClick={handleExportAllOfRunLabel} disabled={busy} className="px-2.5 py-1 border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50">
          Export all questions of this run_label
        </button>
        <button onClick={handleExportDisagreements} disabled={busy} className="px-2.5 py-1 border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50">
          Export disagreement questions
        </button>
      </div>
      {trustGroundedRun && (
        <button
          onClick={() => void exportQuestions(
            trustGroundedRun.history_id, 'C-trust-grounded', (trustGroundedDetail?.results ?? []).map((r) => r.question_id),
          )}
          disabled={busy || !trustGroundedDetail}
          className="px-2.5 py-1 text-xs border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50"
        >
          Export C-trust-grounded (all 10)
        </button>
      )}
      {error && <div className="mt-2 text-xs text-danger">{error}</div>}
    </div>
  )
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

  const eligibleRuns = useMemo(
    () => (batch?.runs ?? []).filter((r) => r.condition !== 'A' && r.status === 'success'),
    [batch],
  )
  const { data: ctxrelBatchSummary } = useContextRelevanceBatchSummary(batchId, 'primary')
  const contextRelevanceRows = ctxrelBatchSummary?.rows ?? []
  const contextRelevanceHasResults = ctxrelBatchSummary?.has_any_results ?? false

  const linkQueries = useContextRelevanceLinkToOutcomesMulti(eligibleRuns.map((r) => r.history_id))
  const linkRows = useMemo(
    () => eligibleRuns.map((r, i) => ({ runLabel: r.run_label ?? r.condition, tables: linkQueries[i]?.data?.tables })),
    [eligibleRuns, linkQueries],
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
      contextRelevance: contextRelevanceRows,
      contextRelevanceHasResults,
      linkRows,
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

      {batch && (
        <div className="mt-6">
          <h2 className="text-sm font-medium text-text-secondary mb-2">Context relevance</h2>
          {!contextRelevanceHasResults ? (
            <div className="text-sm text-text-muted">
              No ctxrel results found for these runs yet -- launch "Context relevance" on this batch from the
              Hallucination Judge (judge-v1) page.
            </div>
          ) : (
            <>
              <div className="overflow-x-auto border border-border rounded-lg mb-4">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-bg border-b border-border-strong text-text-secondary">
                      <th className="text-left font-medium px-3 py-2">run_label</th>
                      <th className="text-left font-medium px-3 py-2">items judged</th>
                      <th className="text-left font-medium px-3 py-2">% RELEVANT</th>
                      <th className="text-left font-medium px-3 py-2">% PARTIAL</th>
                      <th className="text-left font-medium px-3 py-2">% IRRELEVANT</th>
                      <th className="text-left font-medium px-3 py-2">Questions w/ ≥1 RELEVANT</th>
                      <th className="text-left font-medium px-3 py-2">Mean score</th>
                      <th className="text-left font-medium px-3 py-2">Mean score (rank 1)</th>
                      <th className="text-left font-medium px-3 py-2">Mean score (ranks 2-5)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contextRelevanceRows.map((r) => {
                      const pct = (v: number | null | undefined) => (v == null ? '—' : `${v.toFixed(1)}%`)
                      const m = r.metrics
                      if (r.no_context) {
                        return (
                          <tr key={r.run_label} className="border-b border-border last:border-b-0 text-text-muted">
                            <td className="px-3 py-2 font-medium text-text-primary">{r.run_label}</td>
                            <td className="px-3 py-2" colSpan={8}>n/a — no context</td>
                          </tr>
                        )
                      }
                      return (
                        <tr key={r.run_label} className="border-b border-border last:border-b-0">
                          <td className="px-3 py-2 font-medium text-text-primary">{r.run_label}</td>
                          <td className="px-3 py-2">{m?.n_items_judged ?? '—'}</td>
                          <td className="px-3 py-2">{pct(m?.pct_relevant)}</td>
                          <td className="px-3 py-2">{pct(m?.pct_partial)}</td>
                          <td className="px-3 py-2">{pct(m?.pct_irrelevant)}</td>
                          <td className="px-3 py-2">{pct(m?.pct_questions_with_relevant)}</td>
                          <td className="px-3 py-2">{m?.mean_relevance_score != null ? m.mean_relevance_score.toFixed(3) : '—'}</td>
                          <td className="px-3 py-2">{m?.mean_relevance_score_rank1 != null ? m.mean_relevance_score_rank1.toFixed(3) : '—'}</td>
                          <td className="px-3 py-2">{m?.mean_relevance_score_rank2plus != null ? m.mean_relevance_score_rank2plus.toFixed(3) : '—'}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              <h3 className="text-sm font-medium text-text-secondary mb-2">
                Link to outcomes (hallucination label × has ≥1 RELEVANT context item)
              </h3>
              <div className="flex flex-col gap-3">
                {linkRows.filter((l) => l.tables).map((l) => (
                  <LinkToOutcomesCard key={l.runLabel} runLabel={l.runLabel} tables={l.tables!} />
                ))}
                {linkRows.every((l) => !l.tables) && (
                  <div className="text-sm text-text-muted">No link-to-outcomes data available yet.</div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {batch && batch.runs.length > 0 && (
        <PerQuestionExportSection runs={batch.runs} disagreements={disagreementsForBatch} />
      )}
    </div>
  )
}

function LinkToOutcomesCard({ runLabel, tables }: { runLabel: string; tables: ContextRelevanceLinkTables }) {
  return (
    <div className="border border-border rounded-md p-3 text-xs">
      <div className="font-medium text-text-primary mb-2">{runLabel}</div>
      <table className="w-full mb-2">
        <thead>
          <tr className="text-left text-text-secondary border-b border-border">
            <th className="py-1 pr-3">hallucination label</th>
            <th className="py-1 pr-3">has ≥1 relevant</th>
            <th className="py-1">no relevant</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(tables.label_x_relevant).map(([label, counts]) => (
            <tr key={label} className="border-b border-border/50">
              <td className="py-1 pr-3">{label}</td>
              <td className="py-1 pr-3">{counts.has_relevant}</td>
              <td className="py-1">{counts.no_relevant}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {tables.abstain_x_relevant && (
        <table className="w-full">
          <thead>
            <tr className="text-left text-text-secondary border-b border-border">
              <th className="py-1 pr-3">ABSTAIN × has ≥1 relevant</th>
              <th className="py-1 pr-3">has ≥1 relevant</th>
              <th className="py-1">no relevant</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border/50">
              <td className="py-1 pr-3">abstain</td>
              <td className="py-1 pr-3">{tables.abstain_x_relevant.has_relevant.abstain}</td>
              <td className="py-1">{tables.abstain_x_relevant.no_relevant.abstain}</td>
            </tr>
            <tr>
              <td className="py-1 pr-3">not abstain</td>
              <td className="py-1 pr-3">{tables.abstain_x_relevant.has_relevant.not_abstain}</td>
              <td className="py-1">{tables.abstain_x_relevant.no_relevant.not_abstain}</td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  )
}
