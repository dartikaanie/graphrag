import { Link, useSearchParams } from 'react-router-dom'
import {
  useCompareConsistency,
  useContextRelevanceLinkToOutcomesMulti,
  useContextRelevanceV1SummaryMulti,
  useHistoryDetails,
  useJudgeV1Agreement,
  useJudgeV1Disagreements,
  useJudgeV1Summary,
  type ContextRelevanceLinkTables,
  type ContextRelevanceV1RunMetrics,
  type JudgeV1Disagreement,
  type JudgeV1RunLabelSummary,
  type SampleConsistencyResult,
} from '@/api/hooks'
import { JudgeV1ResultsSection } from '@/components/JudgeV1ResultsSection'
import { downloadMarkdown, mdTable } from '@/lib/markdown'
import { fmtDuration } from '@/lib/format'
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
  // 3-way citation split -- the other two slices of the same 100% as
  // "Valid Citation %" above, plus a precision score across the whole run.
  { label: 'NoCit %', get: (r) => r.summary?.pct_citation_no_citation, fmt: (v) => `${v.toFixed(1)}%`, higherIsBetter: false },
  { label: 'InvOnly %', get: (r) => r.summary?.pct_citation_invalid_only, fmt: (v) => `${v.toFixed(1)}%`, higherIsBetter: false },
  { label: 'Fabric %', get: (r) => r.summary?.fabricated_citation_rate, fmt: (v) => `${v.toFixed(1)}%`, higherIsBetter: false },
  { label: 'Precis', get: (r) => r.summary?.citation_precision, fmt: (v) => v.toFixed(3), higherIsBetter: true },
  {
    label: 'Context Items Used — Mean',
    get: (r) => r.summary?.n_context_items_used_mean,
    fmt: (v) => v.toFixed(2),
    higherIsBetter: true,
  },
  {
    label: 'Context Items Used — Min',
    get: (r) => r.summary?.n_context_items_used_min,
    fmt: (v) => String(v),
    higherIsBetter: true,
  },
  { label: 'Avg Retrieval Latency (s)', get: (r) => r.summary?.avg_retrieval_latency_sec, fmt: (v) => v.toFixed(2), higherIsBetter: false },
  { label: 'Total Run Time', get: (r) => r.summary?.duration_sec, fmt: (v) => fmtDuration(v), higherIsBetter: false },
]

interface ParamRow {
  label: string
  get: (run: RunState) => string
}

// Purely informational -- no best/worst highlighting, since these are
// configuration values, not something "better" or "worse". Shown so it's
// visually obvious whether the compared runs actually used matching
// settings (same seed/oversample_pool = a fair sample-for-sample
// comparison) before trusting the metric differences below.
const PARAM_ROWS: ParamRow[] = [
  { label: 'Mode', get: (r) => r.mode },
  { label: 'Provider / Model', get: (r) => `${r.params.provider ?? '—'} / ${r.params.model ?? '—'}` },
  { label: 'Prompt Version', get: (r) => r.params.prompt_version ?? '—' },
  {
    label: 'C Retrieval Version',
    get: (r) => (r.condition === 'C' ? r.params.c_retrieval_version ?? '—' : '—'),
  },
  {
    label: 'D Retrieval Version',
    get: (r) => (r.condition === 'D' ? r.params.d_retrieval_version ?? '—' : '—'),
  },
  { label: 'Seed', get: (r) => (r.params.seed != null ? String(r.params.seed) : '—') },
  { label: 'N Sample', get: (r) => (r.params.n_sample != null ? String(r.params.n_sample) : '—') },
  { label: 'Oversample Pool', get: (r) => (r.params.oversample_pool != null ? String(r.params.oversample_pool) : '—') },
  {
    label: 'Candidates After Token Filter',
    get: (r) =>
      r.params.n_candidates_after_token_filter != null ? String(r.params.n_candidates_after_token_filter) : '—',
  },
  { label: 'Top K', get: (r) => (r.params.top_k != null ? String(r.params.top_k) : '—') },
  { label: 'N Anchor', get: (r) => (r.params.n_anchor != null ? String(r.params.n_anchor) : '—') },
  { label: 'N Semantic Expansion', get: (r) => (r.params.n_semantic_expansion != null ? String(r.params.n_semantic_expansion) : '—') },
  {
    label: 'Require Citation (B)',
    get: (r) => (r.params.require_citation == null ? '—' : r.params.require_citation ? 'Yes' : 'No'),
  },
  { label: 'Fusion Mode (C)', get: (r) => r.params.fusion_mode ?? '—' },
  {
    label: 'Fusion Weights (C)',
    get: (r) =>
      r.params.fusion_w_path_trust == null && r.params.fusion_w_intrinsic == null
        ? '—'
        : `path=${r.params.fusion_w_path_trust ?? '—'} / intrinsic=${r.params.fusion_w_intrinsic ?? '—'}`,
  },
  {
    label: 'Semantic Expansion Enabled (C)',
    get: (r) => (r.params.enable_semantic_expansion == null ? '—' : r.params.enable_semantic_expansion ? 'Yes' : 'No'),
  },
  {
    label: 'N Low-Level / High-Level (D)',
    get: (r) =>
      r.params.n_low_level == null && r.params.n_high_level == null
        ? '—'
        : `${r.params.n_low_level ?? '—'} / ${r.params.n_high_level ?? '—'}`,
  },
  {
    label: 'Grounding Constraint (C/D)',
    get: (r) => (r.params.require_grounding == null ? '—' : r.params.require_grounding ? 'Yes' : 'No'),
  },
  {
    label: 'log_full_candidates (B/C)',
    get: (r) => (r.params.log_full_candidates == null ? '—' : r.params.log_full_candidates ? 'Yes' : 'No'),
  },
]

function columnLabel(run: RunState): string {
  return `Condition ${run.condition} (${run.run_id})`
}

interface ConsistencyBanner {
  tone: 'success' | 'primary' | 'danger'
  text: string
}

// Same 3-way verdict the backend computes (history_service.compute_sample_
// consistency) rendered as plain text -- shared between the on-page banner
// and the markdown export so the two never say something different.
function consistencyBanner(result: SampleConsistencyResult | undefined): ConsistencyBanner | null {
  if (!result) return null
  const n = result.runs.map((r) => r.n).join('/')
  if (result.sample_consistency === 'identical') {
    return { tone: 'success', text: `All runs evaluated the same ${result.runs[0]?.n ?? '?'} questions.` }
  }
  if (result.sample_consistency === 'subset_nested') {
    const prefixNote =
      result.is_exact_prefix === true
        ? 'smaller runs are an exact PREFIX of the larger run(s) (n=10 nested inside n=384-style).'
        : result.is_exact_prefix === false
          ? 'smaller runs are a SUBSET of the larger run(s), but NOT in the same first-N order — same underlying pool, different sampling order.'
          : 'smaller runs are a subset of the larger run(s).'
    return { tone: 'primary', text: `Nested samples (n=${n}): ${prefixNote}` }
  }
  return {
    tone: 'danger',
    text: `Runs evaluated DIFFERENT questions (n=${n}, intersection=${result.intersection_size}). Metric comparison below is NOT paired — treat it as informational only.`,
  }
}

interface JudgeV1ExportData {
  runLabels: string[]
  primarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
  secondarySummary: Record<string, JudgeV1RunLabelSummary> | undefined
  agreement: {
    n_pairs: number
    kappas: { weighted: number | null; unweighted: number | null; n_used: number; n_excluded: number } | null
    insufficient_data: boolean
  } | undefined
  disagreements: JudgeV1Disagreement[]
  contextRelevance: Array<{ run_label: string | null; metrics: ContextRelevanceV1RunMetrics | undefined }>
  contextRelevanceLinkRows: Array<{ runLabel: string; tables: ContextRelevanceLinkTables | undefined }>
}

function buildComparisonMarkdown(
  runs: RunState[], consistency: SampleConsistencyResult | undefined, judgeV1: JudgeV1ExportData | undefined,
): string {
  const header = ['Parameter', ...runs.map(columnLabel)]
  const paramRows = PARAM_ROWS.map((row) => [row.label, ...runs.map((r) => row.get(r))])

  const metricHeader = ['Metric', ...runs.map(columnLabel)]
  const metricRows = ROWS.map((row) => {
    const values = runs.map((r) => row.get(r))
    return [row.label, ...values.map((v) => (typeof v === 'number' ? row.fmt(v) : '—'))]
  })

  const banner = consistencyBanner(consistency)

  const lines = [
    '# Run Comparison',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Runs compared: ${runs.map((r) => `${r.condition} (${r.run_id})`).join(', ')}`,
    '',
    ...(banner ? [`**Sample consistency**: ${banner.text}`, ''] : []),
    '## Parameters',
    '',
    mdTable(header, paramRows),
    '',
    '## Metrics (legacy, faithfulness/generation)',
    '',
    mdTable(metricHeader, metricRows),
    '',
  ]

  if (judgeV1 && judgeV1.runLabels.length > 0) {
    const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
    const rate = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3))
    const summaryHeader = [
      'run_label',
      'primary N', 'primary Abstain%', 'primary Hall(excl)', 'primary Hall(incl)',
      'secondary N', 'secondary Abstain%', 'secondary Hall(excl)', 'secondary Hall(incl)',
    ]
    const summaryRows = judgeV1.runLabels.map((label) => {
      const p = judgeV1.primarySummary?.[label]
      const s = judgeV1.secondarySummary?.[label]
      return [
        label, String(p?.n ?? '—'), pct(p?.abstention_rate), rate(p?.hall_rate_excl_abstain), rate(p?.hall_rate_incl_abstain),
        String(s?.n ?? '—'), pct(s?.abstention_rate), rate(s?.hall_rate_excl_abstain), rate(s?.hall_rate_incl_abstain),
      ]
    })

    const disagreementRows = judgeV1.disagreements.map((d) => [
      String(d.question_id ?? '—'), String(d.run_label ?? '—'),
      String(d.primary_label ?? '—'), String(d.secondary_label ?? '—'),
      String(d.primary_reasoning ?? '—'), String(d.secondary_reasoning ?? '—'),
    ])

    const pctC = (v: number | null | undefined) => (v == null ? '—' : `${v.toFixed(1)}%`)
    const ctxrelHasResults = judgeV1.contextRelevance.some((c) => (c.metrics?.n_items_judged ?? 0) > 0)
    const ctxrelRows = judgeV1.contextRelevance.map((c) => [
      String(c.run_label ?? '—'), String(c.metrics?.n_items_judged ?? '—'),
      pctC(c.metrics?.pct_relevant), pctC(c.metrics?.pct_partial), pctC(c.metrics?.pct_irrelevant),
      pctC(c.metrics?.pct_questions_with_relevant),
      c.metrics?.mean_relevance_score != null ? c.metrics.mean_relevance_score.toFixed(3) : '—',
      c.metrics?.mean_relevance_score_rank1 != null ? c.metrics.mean_relevance_score_rank1.toFixed(3) : '—',
      c.metrics?.mean_relevance_score_rank2plus != null ? c.metrics.mean_relevance_score_rank2plus.toFixed(3) : '—',
    ])

    const ctxrelLines: string[] = []
    if (ctxrelHasResults) {
      ctxrelLines.push(
        mdTable([
          'run_label', 'items judged', '% RELEVANT', '% PARTIAL', '% IRRELEVANT',
          'Questions w/ ≥1 RELEVANT', 'Mean score', 'Mean score (rank 1)', 'Mean score (ranks 2-5)',
        ], ctxrelRows),
        '',
      )
      const linkRows = judgeV1.contextRelevanceLinkRows.filter((l) => l.tables)
      if (linkRows.length > 0) {
        ctxrelLines.push('#### Link to outcomes (hallucination label × has ≥1 RELEVANT context item)', '')
        for (const { runLabel, tables } of linkRows) {
          ctxrelLines.push(`**${runLabel}**`, '')
          const labelRows = Object.entries(tables!.label_x_relevant).map(([label, counts]) => [
            label, String(counts.has_relevant), String(counts.no_relevant),
          ])
          ctxrelLines.push(mdTable(['hallucination label', 'has ≥1 relevant', 'no relevant'], labelRows), '')
        }
      }
    } else {
      ctxrelLines.push('(no ctxrel results found for these runs yet)', '')
    }

    lines.push(
      '## Hallucination (reference-based, judge-v1)',
      '',
      mdTable(summaryHeader, summaryRows),
      '',
      '### Agreement (primary vs secondary)',
      '',
      judgeV1.agreement?.kappas
        ? `n pairs = ${judgeV1.agreement.n_pairs}, weighted κ = ${judgeV1.agreement.kappas.weighted ?? '—'}, unweighted κ = ${judgeV1.agreement.kappas.unweighted ?? '—'}`
        : 'Insufficient paired data for primary-vs-secondary agreement.',
      '',
      '### Disagreements',
      '',
      mdTable(['question_id', 'run_label', 'primary label', 'secondary label', 'primary reasoning', 'secondary reasoning'], disagreementRows),
      '',
      '### Context relevance (primary judge)',
      '',
      ...ctxrelLines,
    )
  }

  return lines.join('\n')
}

export function HistoryComparePage() {
  const [searchParams] = useSearchParams()
  const ids = (searchParams.get('ids') ?? '').split(',').filter(Boolean)

  // useQueries (not one useHistoryDetail call per id) so the number of runs
  // compared can vary freely -- History no longer caps selection at 4.
  const queries = useHistoryDetails(ids)
  const runs = queries.map((q) => q.data).filter((r): r is RunState => !!r)
  const { data: consistency } = useCompareConsistency(ids)
  const banner = consistencyBanner(consistency)

  const runLabels = runs.map((r) => r.run_label).filter((l): l is string => !!l)
  const { data: primarySummary } = useJudgeV1Summary('primary')
  const { data: secondarySummary } = useJudgeV1Summary('secondary')
  const { data: agreement } = useJudgeV1Agreement('primary', 'secondary')
  const { data: disagreementsData } = useJudgeV1Disagreements('primary', 'secondary')
  const disagreementsForCompare = (disagreementsData?.disagreements ?? []).filter((d) => runLabels.includes(d.run_label ?? ''))

  const ctxrelRuns = runs.filter((r) => r.condition !== 'A')
  const ctxrelRunIds = ctxrelRuns.map((r) => r.run_id)
  const ctxrelQueries = useContextRelevanceV1SummaryMulti(ctxrelRunIds, 'primary')
  const contextRelevance = ctxrelRuns.map((r, i) => ({ run_label: r.run_label ?? null, metrics: ctxrelQueries[i]?.data?.metrics }))

  const linkQueries = useContextRelevanceLinkToOutcomesMulti(ctxrelRunIds)
  const contextRelevanceLinkRows = ctxrelRuns.map((r, i) => ({
    runLabel: r.run_label ?? r.condition, tables: linkQueries[i]?.data?.tables,
  }))

  if (ids.length === 0) {
    return <div className="text-sm text-text-muted">No runs selected. Go back to History and select 2 or more runs.</div>
  }

  const handleExport = () => {
    const md = buildComparisonMarkdown(runs, consistency, {
      runLabels, primarySummary: primarySummary?.summary, secondarySummary: secondarySummary?.summary,
      agreement, disagreements: disagreementsForCompare, contextRelevance, contextRelevanceLinkRows,
    })
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    downloadMarkdown(`run-comparison_${runs.map((r) => r.condition).join('')}_${stamp}.md`, md)
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/history" className="hover:text-primary">
          ← Back to History
        </Link>
      </div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-semibold text-text-primary">Compare Runs</h1>
        {runs.length > 0 && (
          <button
            onClick={handleExport}
            className="px-3 py-1.5 text-sm border border-border-strong rounded-md text-text-primary hover:bg-bg"
          >
            ↓ Export as Markdown
          </button>
        )}
      </div>

      {runs.length < ids.length && <div className="text-sm text-text-muted mb-4">Loading...</div>}

      {banner && (
        <div
          className={`mb-4 px-3 py-2.5 rounded-md border text-sm ${
            banner.tone === 'success'
              ? 'bg-white text-success border-success/40'
              : banner.tone === 'primary'
                ? 'bg-primary-soft text-primary border-primary-border'
                : 'bg-white text-danger border-danger/40'
          }`}
        >
          {banner.tone === 'success' ? '✓ ' : banner.tone === 'primary' ? 'ℹ ' : '⚠ '}
          {banner.text}
        </div>
      )}

      {runs.length > 0 && (
        <>
          <h2 className="text-sm font-medium text-text-secondary mb-2">Parameters</h2>
          <div className="overflow-x-auto border border-border rounded-lg mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg border-b border-border-strong">
                  <th className="text-left font-medium text-text-secondary px-4 py-2.5">Parameter</th>
                  {runs.map((r) => (
                    <th key={r.run_id} className="text-left font-medium text-text-secondary px-4 py-2.5">
                      Condition {r.condition}
                      <div className="text-xs font-normal text-text-muted">{r.run_id}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PARAM_ROWS.map((row) => (
                  <tr key={row.label} className="border-b border-border last:border-b-0">
                    <td className="px-4 py-2.5 text-text-secondary">{row.label}</td>
                    {runs.map((r) => (
                      <td key={r.run_id} className="px-4 py-2.5 text-text-primary">
                        {row.get(r)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="text-sm font-medium text-text-secondary mb-2">Metrics</h2>
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

          <JudgeV1ResultsSection runLabels={runs.map((r) => r.run_label).filter((l): l is string => !!l)} />

          {contextRelevance.length > 0 && (
            <div className="mt-6">
              <h2 className="text-sm font-medium text-text-secondary mb-2">Context relevance (primary judge)</h2>
              <div className="overflow-x-auto border border-border rounded-lg">
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
                    {contextRelevance.map((c, i) => {
                      const m = c.metrics
                      const pct = (v: number | null | undefined) => (v == null ? '—' : `${v.toFixed(1)}%`)
                      return (
                        <tr key={i} className="border-b border-border last:border-b-0">
                          <td className="px-3 py-2 font-medium text-text-primary">{c.run_label ?? '—'}</td>
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
                    {contextRelevance.every((c) => !c.metrics || c.metrics.n_items_judged === 0) && (
                      <tr><td colSpan={9} className="px-3 py-2 text-text-muted">No ctxrel results found for these runs yet.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
