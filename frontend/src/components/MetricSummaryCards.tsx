import type { RunCondition, RunSummary } from '@/types/run'
import { fmtDuration } from '@/lib/format'
import { MetricInfoLink } from './MetricInfoLink'

function fmtPct(v: number | undefined): string {
  return v === undefined ? '—' : `${v.toFixed(1)}%`
}

function fmtScore(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : v.toFixed(4)
}

/** Reused for both the live run view and the (future) history detail view. */
export function MetricSummaryCards({ summary }: { summary: RunSummary; condition: RunCondition }) {
  const cards: { label: string; value: string; metricId?: string }[] = [
    { label: 'Processed', value: String(summary.n_processed) },
    { label: 'Avg Cosine Similarity (0–1)', value: fmtScore(summary.cosine_similarity_mean), metricId: 'cosine-similarity' },
    { label: 'Median Cosine Similarity (0–1)', value: fmtScore(summary.cosine_similarity_median), metricId: 'cosine-similarity' },
    { label: '% Similarity > 0.5', value: fmtPct(summary.pct_similarity_above_0_5), metricId: 'cosine-similarity' },
  ]
  // Both NF2 and retrieval latency show whenever the summary actually has
  // them, rather than being gated by condition -- Condition A never
  // retrieves at all (never produces either field), but B produces both
  // when --require-citation is on (default) and always produces latency,
  // so B vs C can be compared directly on both metrics.
  if (summary.pct_with_valid_citation !== undefined) {
    cards.push({ label: 'NF2 — Citation Compliance', value: fmtPct(summary.pct_with_valid_citation), metricId: 'nf2-citation-validity' })
  }
  if (summary.avg_retrieval_latency_sec !== undefined) {
    cards.push({ label: 'Avg Retrieval Latency', value: `${summary.avg_retrieval_latency_sec.toFixed(2)}s`, metricId: 'nf3-retrieval-latency' })
  }
  if (summary.duration_sec !== undefined) {
    cards.push({ label: 'Total Run Time', value: fmtDuration(summary.duration_sec) })
  }

  return (
    <div className="grid grid-cols-4 gap-4">
      {cards.map((c) => (
        <div key={c.label} className="border border-border rounded-lg bg-surface px-4 py-3">
          <div className="text-xs text-text-secondary inline-flex items-center gap-1">
            {c.label}
            {c.metricId && <MetricInfoLink metricId={c.metricId} label={c.label} />}
          </div>
          <div className="text-xl font-semibold text-text-primary mt-1">{c.value}</div>
        </div>
      ))}
    </div>
  )
}
