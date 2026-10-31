import { usePanelEvaluation, type PanelEvaluation, type PanelJudgeBreakdown, type PanelMetricDefinition, type PanelMetricRun } from '@/api/hooks'
import { mdTable } from '@/lib/markdown'
import { FaithfulnessBlock, faithfulnessMarkdown } from '@/components/FaithfulnessBlock'

const HALL_LABELS = ['FAKTUAL', 'HALUSINASI_SEBAGIAN', 'HALUSINASI_PENUH', 'ABSTAIN']
const REL_LABELS: Record<string, string[]> = {
  ansrel: ['MENJAWAB', 'MENJAWAB_SEBAGIAN', 'TIDAK_MENJAWAB'],
  ctxrel: ['RELEVANT', 'PARTIAL', 'IRRELEVANT'],
}

export function fmtPanelRate(v: number | null | undefined): string {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`
}
function fmtScore(v: number | null | undefined): string {
  return v == null ? '—' : v.toFixed(3)
}
function cell(dist: Record<string, number>, label: string, n: number): string {
  const c = dist[label] ?? 0
  return `${c} (${n ? ((c / n) * 100).toFixed(1) : '—'}%)`
}

type CtxExtraSource = { pct_questions_with_relevant?: number | null; mean_score_rank1?: number | null; mean_score_rank2plus?: number | null; n_unjudged?: number }
/** Context-relevance-only columns (shared by the page tables and the markdown export). */
const CTX_EXTRA: Array<{ h: string; get: (x: CtxExtraSource) => string }> = [
  { h: 'Questions with ≥1 RELEVANT', get: (x) => (x.pct_questions_with_relevant == null ? '—' : `${x.pct_questions_with_relevant.toFixed(1)}%`) },
  { h: 'Mean score rank 1', get: (x) => fmtScore(x.mean_score_rank1) },
  { h: 'Mean score rank 2+', get: (x) => fmtScore(x.mean_score_rank2plus) },
  { h: 'Unjudged items', get: (x) => String(x.n_unjudged ?? 0) },
]

function DefinitionBox({ def }: { def: PanelMetricDefinition }) {
  return (
    <div className="text-xs text-text-secondary border border-border rounded-lg px-3 py-2 mb-3 bg-bg">
      <div><span className="font-medium">Unit judged:</span> {def.unit}</div>
      <div className="mt-1">
        <span className="font-medium">Labels:</span>{' '}
        {def.labels.map((l) => `${l.label}${l.score != null ? ` (${l.score})` : ''} = ${l.meaning}`).join(' · ')}
      </div>
      <div className="mt-1"><span className="font-medium">How calculated:</span> {def.calculation}</div>
    </div>
  )
}

function panelStatus(p: { n_unresolved?: number; n_missing_votes?: number }): string {
  const parts = []
  if (p.n_missing_votes) parts.push(`${p.n_missing_votes} items missing a judge's vote`)
  if (p.n_unresolved) parts.push(`${p.n_unresolved} unresolved`)
  return parts.length ? parts.join(', ') : 'complete'
}

const TH = 'text-left font-medium px-3 py-2 whitespace-nowrap'
const TD = 'px-3 py-2 whitespace-nowrap'

/** Every panel metric (hallucination, answer relevance, context relevance) for
 * one or more History runs: a panel summary table plus a per-judge breakdown
 * with judge identity. Used on History detail (one run) and Compare (many). */
export function PanelEvaluationSection({ runIds, runNames }: { runIds: string[]; runNames?: Record<string, string> }) {
  const { data, isLoading, isError } = usePanelEvaluation(runIds)
  const name = (id: string) => runNames?.[id] ?? id

  return (
    <div className="mt-6 mb-6">
      <h2 className="text-sm font-medium text-text-secondary mb-1">Evaluation metrics — 3-judge panel</h2>
      <p className="text-xs text-text-muted mb-3">
        Panel = majority vote of the judges (median on the ordinal scale if all three differ). Each metric shows the panel
        summary first, then the breakdown per judge. The combined verdict is withheld until all three judges have results; items a judge failed on are decided by the other two when they agree, and counted in Status.
      </p>
      {isLoading && <div className="text-sm text-text-muted">Loading...</div>}
      {isError && <div className="text-sm text-danger">Failed to load panel evaluation.</div>}
      {data?.metrics.map((m) => m.metric === 'faithfulness' ? (
        <FaithfulnessBlock key={m.metric} title={m.title} definition={m.definition} runIds={runIds} runs={m.runs} name={name} />
      ) : (
        <MetricBlock key={m.metric} metric={m.metric} title={m.title} definition={m.definition} runIds={runIds} runs={m.runs} name={name} />
      ))}
    </div>
  )
}

function MetricBlock({
  metric, title, definition, runIds, runs, name,
}: {
  metric: string
  title: string
  definition?: PanelMetricDefinition
  runIds: string[]
  runs: Record<string, PanelMetricRun | null>
  name: (id: string) => string
}) {
  const isHall = metric === 'hallucination'
  const isCtx = metric === 'ctxrel'
  const labels = isHall ? HALL_LABELS : REL_LABELS[metric] ?? []
  const anyData = runIds.some((id) => runs[id] && !runs[id]!.note)

  return (
    <div className="mb-6">
      <h3 className="text-sm font-medium text-text-primary mb-2">{title}</h3>
      {definition && <DefinitionBox def={definition} />}
      {!anyData ? (
        <div className="text-sm text-text-muted border border-border rounded-lg px-3 py-2.5">
          {runIds.map((id) => runs[id]?.note).find(Boolean) ??
            `Not run yet for ${runIds.length > 1 ? 'these runs' : 'this run'} (no official panel output).`}
        </div>
      ) : (
        <>
          <div className="text-xs text-text-secondary mb-1">Panel summary</div>
          <div className="overflow-x-auto border border-border rounded-lg mb-3">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg border-b border-border-strong text-text-secondary">
                  <th className={TH}>Run</th>
                  <th className={TH}>N</th>
                  {isHall ? (
                    <>
                      <th className={TH}>Halluc. (excl. ABSTAIN)</th>
                      <th className={TH}>Halluc. (incl. ABSTAIN)</th>
                      <th className={TH}>Abstain</th>
                      <th className={TH}>Unanimous</th>
                    </>
                  ) : (
                    <>
                      <th className={TH}>Mean score</th>
                      {isCtx && CTX_EXTRA.map((e) => <th key={e.h} className={TH}>{e.h}</th>)}
                    </>
                  )}
                  {labels.map((l) => <th key={l} className={TH}>{l}</th>)}
                  {!isHall && <th className={TH}>Status</th>}
                </tr>
              </thead>
              <tbody>
                {runIds.map((id) => {
                  const r = runs[id]
                  const p = r?.panel
                  return (
                    <tr key={id} className="border-b border-border last:border-b-0">
                      <td className={`${TD} font-medium text-text-primary`}>{name(id)}</td>
                      {!r || r.note ? (
                        <td colSpan={labels.length + 6 + (isCtx ? 4 : 0)} className="px-3 py-2 text-text-muted">{r?.note ?? 'not run'}</td>
                      ) : !p ? (
                        <td colSpan={labels.length + 6 + (isCtx ? 4 : 0)} className="px-3 py-2 text-warning">
                          Panel verdict withheld — only {(r.judges_present ?? []).length} of 3 judges have results ({(r.judges_present ?? []).join(', ') || 'none'}).
                          See the per-judge breakdown below.
                        </td>
                      ) : (
                        <>
                          {(() => {
                            const n = p.n ?? p.n_decided ?? 0
                            return (
                              <>
                                <td className={TD}>{n}</td>
                                {isHall ? (
                                  <>
                                    <td className={`${TD} font-medium`}>{fmtPanelRate(p.hall_rate_excl_abstain)}</td>
                                    <td className={`${TD} font-medium`}>{fmtPanelRate(p.hall_rate_incl_abstain)}</td>
                                    <td className={TD}>{fmtPanelRate(p.abstention_rate)}</td>
                                    <td className={TD}>{fmtPanelRate(p.unanimous_rate)}</td>
                                  </>
                                ) : (
                                  <>
                                    <td className={`${TD} font-medium`}>{fmtScore(p.mean_score)}</td>
                                    {isCtx && CTX_EXTRA.map((e) => <td key={e.h} className={TD}>{e.get(p)}</td>)}
                                  </>
                                )}
                                {labels.map((l) => <td key={l} className={TD}>{cell(p.label_distribution, l, n)}</td>)}
                                {!isHall && <td className={TD}>{panelStatus(p)}</td>}
                              </>
                            )
                          })()}
                        </>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="text-xs text-text-secondary mb-1">Per-judge breakdown</div>
          <div className="overflow-x-auto border border-border rounded-lg">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg border-b border-border-strong text-text-secondary">
                  <th className={TH}>Run</th>
                  <th className={TH}>Judge</th>
                  <th className={TH}>Model</th>
                  <th className={TH}>Provider</th>
                  <th className={TH}>Temp</th>
                  <th className={TH}>N</th>
                  {isHall ? (
                    <>
                      <th className={TH}>Halluc. (excl.)</th>
                      <th className={TH}>Halluc. (incl.)</th>
                      <th className={TH}>Abstain</th>
                    </>
                  ) : (
                    <>
                      <th className={TH}>Mean score</th>
                      <th className={TH}>Unparsed</th>
                      {isCtx && CTX_EXTRA.map((e) => <th key={e.h} className={TH}>{e.h}</th>)}
                    </>
                  )}
                  {labels.map((l) => <th key={l} className={TH}>{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {runIds.flatMap((id) => {
                  const judges: PanelJudgeBreakdown[] = runs[id]?.per_judge ?? []
                  if (judges.length === 0) {
                    return [(
                      <tr key={id} className="border-b border-border">
                        <td className={`${TD} font-medium text-text-primary`}>{name(id)}</td>
                        <td colSpan={labels.length + 8 + (isCtx ? 4 : 0)} className="px-3 py-2 text-text-muted">not run</td>
                      </tr>
                    )]
                  }
                  return judges.map((j, i) => (
                    <tr key={`${id}|${j.judge_id}`} className="border-b border-border last:border-b-0">
                      <td className={`${TD} font-medium text-text-primary`}>{i === 0 ? name(id) : ''}</td>
                      <td className={TD}>{j.judge_id}</td>
                      <td className={`${TD} font-mono`}>{j.judge_model ?? '—'}</td>
                      <td className={TD}>{j.provider ?? '—'}</td>
                      <td className={TD}>{j.temperature ?? '—'}</td>
                      <td className={TD}>{j.n}</td>
                      {isHall ? (
                        <>
                          <td className={TD}>{fmtPanelRate(j.hall_rate_excl_abstain)}</td>
                          <td className={TD}>{fmtPanelRate(j.hall_rate_incl_abstain)}</td>
                          <td className={TD}>{fmtPanelRate(j.abstention_rate)}</td>
                        </>
                      ) : (
                        <>
                          <td className={TD}>{fmtScore(j.mean_score)}</td>
                          <td className={TD}>{j.n_unparsed ?? 0}</td>
                          {isCtx && CTX_EXTRA.map((e) => <td key={e.h} className={TD}>{e.get(j)}</td>)}
                        </>
                      )}
                      {labels.map((l) => <td key={l} className={TD}>{cell(j.label_distribution, l, j.n)}</td>)}
                    </tr>
                  ))
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

/** Markdown for the same tables the section renders (panel summary + per-judge
 * breakdown per metric), for History Compare's "Export as Markdown". */
export function panelEvaluationMarkdown(
  data: PanelEvaluation | undefined, runIds: string[], name: (id: string) => string,
): string[] {
  if (!data) return ['## Evaluation metrics — 3-judge panel', '', '(panel evaluation not loaded)', '']
  const lines = [
    '## Evaluation metrics — 3-judge panel',
    '',
    'Panel = majority vote of the judges (median on the ordinal scale if all three differ). The combined verdict is withheld until all three judges have results; items a judge failed on are decided by the other two when they agree, and counted in Status.',
    '',
  ]
  for (const m of data.metrics) {
    const isHall = m.metric === 'hallucination'
    const labels = isHall ? HALL_LABELS : REL_LABELS[m.metric] ?? []
    const isCtx = m.metric === 'ctxrel'
    if (m.metric === 'faithfulness') {
      lines.push(...faithfulnessMarkdown(m.title, m.definition, runIds, m.runs, name))
      continue
    }
    lines.push(`### ${m.title}`, '')
    if (m.definition) {
      lines.push(
        `- **Unit judged:** ${m.definition.unit}`,
        `- **Labels:** ${m.definition.labels.map((l) => `${l.label}${l.score != null ? ` (${l.score})` : ''} = ${l.meaning}`).join('; ')}`,
        `- **How calculated:** ${m.definition.calculation}`,
        '',
      )
    }
    if (!runIds.some((id) => m.runs[id] && !m.runs[id]!.note)) {
      lines.push(runIds.map((id) => m.runs[id]?.note).find(Boolean) ?? 'Not run yet (no official panel output).', '')
      continue
    }
    const summaryHeader = isHall
      ? ['Run', 'N', 'Halluc. (excl. ABSTAIN)', 'Halluc. (incl. ABSTAIN)', 'Abstain', 'Unanimous', ...labels]
      : ['Run', 'N', 'Mean score', ...(isCtx ? CTX_EXTRA.map((e) => e.h) : []), ...labels, 'Status']
    const summaryRows = runIds.map((id) => {
      const r = m.runs[id]
      const p = r?.panel
      if (!r || r.note) return [name(id), ...summaryHeader.slice(1).map((_, i) => (i === 0 ? (r?.note ?? 'not run') : '—'))]
      if (!p) {
        return [name(id), `withheld — only ${(r.judges_present ?? []).length} of 3 judges have results (${(r.judges_present ?? []).join(', ') || 'none'})`,
          ...summaryHeader.slice(2).map(() => '—')]
      }
      const n = p.n ?? p.n_decided ?? 0
      return isHall
        ? [name(id), String(n), fmtPanelRate(p.hall_rate_excl_abstain), fmtPanelRate(p.hall_rate_incl_abstain),
            fmtPanelRate(p.abstention_rate), fmtPanelRate(p.unanimous_rate), ...labels.map((l) => cell(p.label_distribution, l, n))]
        : [name(id), String(n), fmtScore(p.mean_score), ...(isCtx ? CTX_EXTRA.map((e) => e.get(p)) : []),
            ...labels.map((l) => cell(p.label_distribution, l, n)),
            panelStatus(p)]
    })
    lines.push('**Panel summary**', '', mdTable(summaryHeader, summaryRows), '')

    const judgeHeader = isHall
      ? ['Run', 'Judge', 'Model', 'Provider', 'Temp', 'N', 'Halluc. (excl.)', 'Halluc. (incl.)', 'Abstain', ...labels]
      : ['Run', 'Judge', 'Model', 'Provider', 'Temp', 'N', 'Mean score', 'Unparsed',
          ...(isCtx ? CTX_EXTRA.map((e) => e.h) : []), ...labels]
    const judgeRows: string[][] = []
    for (const id of runIds) {
      const judges: PanelJudgeBreakdown[] = m.runs[id]?.per_judge ?? []
      if (judges.length === 0) {
        judgeRows.push([name(id), 'not run', ...judgeHeader.slice(2).map(() => '—')])
        continue
      }
      for (const j of judges) {
        const base = [name(id), j.judge_id, j.judge_model ?? '—', j.provider ?? '—', String(j.temperature ?? '—'), String(j.n)]
        judgeRows.push(isHall
          ? [...base, fmtPanelRate(j.hall_rate_excl_abstain), fmtPanelRate(j.hall_rate_incl_abstain),
              fmtPanelRate(j.abstention_rate), ...labels.map((l) => cell(j.label_distribution, l, j.n))]
          : [...base, fmtScore(j.mean_score), String(j.n_unparsed ?? 0), ...(isCtx ? CTX_EXTRA.map((e) => e.get(j)) : []),
              ...labels.map((l) => cell(j.label_distribution, l, j.n))])
      }
    }
    lines.push('**Per-judge breakdown**', '', mdTable(judgeHeader, judgeRows), '')
  }
  return lines
}
