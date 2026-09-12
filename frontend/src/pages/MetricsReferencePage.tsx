import { useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Badge } from '@/components/Badge'
import { EVALUATION_METRICS, type MetricCategory } from '@/content/evaluationMetrics'
import type { RunCondition } from '@/types/run'

const CATEGORIES: MetricCategory[] = [
  'Semantic Quality',
  'Citation & Grounding',
  'Hallucination & Faithfulness',
  'Retrieval Quality',
  'Efficiency',
]

const ALL_CONDITIONS: RunCondition[] = ['A', 'B', 'C', 'D']

// Same condition -> tone mapping used on MethodologyPage's border accents
// (A=neutral/muted, B=primary, C=success, D=warning), reused here for the
// small per-metric condition badges so the color coding stays consistent
// across the two reference pages.
const CONDITION_BADGE_TONE: Record<RunCondition, 'neutral' | 'primary' | 'success' | 'warning'> = {
  A: 'neutral',
  B: 'primary',
  C: 'success',
  D: 'warning',
}

function MetricCard({ id, isOpen, onToggle }: { id: string; isOpen: boolean; onToggle: () => void }) {
  const metric = EVALUATION_METRICS.find((m) => m.id === id)
  if (!metric) return null

  const related = (metric.relatedTo ?? [])
    .map((relId) => EVALUATION_METRICS.find((m) => m.id === relId))
    .filter((m): m is (typeof EVALUATION_METRICS)[number] => !!m)

  return (
    <div id={metric.id} className="border border-border rounded-lg bg-surface scroll-mt-4">
      <button
        onClick={onToggle}
        className="w-full flex items-start justify-between gap-4 text-left px-4 py-3"
      >
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="text-sm font-semibold text-text-primary">{metric.name}</span>
            {metric.isValidationMetric && <Badge tone="neutral">Validation metric</Badge>}
            {metric.appliesTo.map((c) => (
              <Badge key={c} tone={CONDITION_BADGE_TONE[c]}>{c}</Badge>
            ))}
          </div>
          <p className="text-sm text-text-secondary">{metric.whatItMeasures}</p>
        </div>
        <span className="text-text-muted text-sm shrink-0 mt-0.5">{isOpen ? '−' : '+'}</span>
      </button>

      {isOpen && (
        <div className="px-4 pb-4 flex flex-col gap-3 border-t border-border pt-3">
          <div>
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">How it's computed</h3>
            <p className="text-sm text-text-secondary">{metric.howComputed}</p>
          </div>

          {metric.formula && (
            <div>
              <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">Formula</h3>
              <code className="block text-xs bg-bg border border-border rounded-md px-3 py-2 text-text-primary whitespace-pre-wrap">
                {metric.formula}
              </code>
            </div>
          )}

          <div>
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">How to read it</h3>
            <p className="text-sm text-text-secondary">{metric.interpretationGuide}</p>
          </div>

          <div className="border-l-2 border-border-strong pl-3">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">Limitations</h3>
            <p className="text-sm text-text-secondary">{metric.limitations}</p>
          </div>

          {related.length > 0 && (
            <div className="text-xs text-text-secondary">
              Lihat juga:{' '}
              {related.map((r, i) => (
                <span key={r.id}>
                  <a href={`#${r.id}`} className="text-primary hover:underline">{r.name}</a>
                  {i < related.length - 1 ? ', ' : ''}
                </span>
              ))}
            </div>
          )}

          <div className="text-xs text-text-muted font-mono">
            Diimplementasikan di: {metric.implementedIn.join(', ')}
          </div>
        </div>
      )}
    </div>
  )
}

export function MetricsReferencePage() {
  const location = useLocation()
  const initialOpenId = location.hash ? location.hash.slice(1) : undefined

  const [activeCategory, setActiveCategory] = useState<MetricCategory | 'All'>('All')
  const [activeConditions, setActiveConditions] = useState<Set<RunCondition>>(new Set(ALL_CONDITIONS))
  const [openIds, setOpenIds] = useState<Set<string>>(new Set(initialOpenId ? [initialOpenId] : []))

  // Client-side route changes don't auto-scroll to a #hash the way a full
  // page load does -- scroll to it manually once the target card has
  // rendered (it's conditionally shown by category/condition filters, so
  // wait a tick rather than assuming it's present on the very first paint).
  useEffect(() => {
    if (!initialOpenId) return
    const el = document.getElementById(initialOpenId)
    el?.scrollIntoView({ block: 'start' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toggle = (id: string) => {
    setOpenIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleCondition = (c: RunCondition) => {
    setActiveConditions((prev) => {
      const next = new Set(prev)
      if (next.has(c)) next.delete(c)
      else next.add(c)
      return next
    })
  }

  const filtered = useMemo(
    () =>
      EVALUATION_METRICS.filter(
        (m) =>
          (activeCategory === 'All' || m.category === activeCategory) &&
          m.appliesTo.some((c) => activeConditions.has(c)),
      ),
    [activeCategory, activeConditions],
  )

  return (
    <div className="max-w-4xl">
      <h1 className="text-lg font-semibold text-text-primary mb-1">Metrik Evaluasi</h1>
      <p className="text-sm text-text-secondary mb-6">
        Referensi definisi setiap metrik evaluasi yang dipakai di penelitian ini — apa yang diukur, bagaimana
        dihitung, di kondisi mana berlaku, dan batasannya. Halaman ini bersifat dokumentasi saja (tidak menjalankan
        apa pun); untuk desain retrieval/prompting tiap kondisi, lihat halaman{' '}
        <a href="/methodology" className="text-primary hover:underline">Methodology</a>.
      </p>

      <div className="flex gap-1 mb-4 border-b border-border overflow-x-auto">
        <button
          onClick={() => setActiveCategory('All')}
          className={`px-3 py-2 text-sm border-b-2 -mb-px whitespace-nowrap ${
            activeCategory === 'All' ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
          }`}
        >
          All
        </button>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-2 text-sm border-b-2 -mb-px whitespace-nowrap ${
              activeCategory === cat ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4 mb-4 text-sm">
        <span className="text-text-secondary">Kondisi:</span>
        {ALL_CONDITIONS.map((c) => (
          <label key={c} className="flex items-center gap-1.5">
            <input type="checkbox" checked={activeConditions.has(c)} onChange={() => toggleCondition(c)} />
            <Badge tone={CONDITION_BADGE_TONE[c]}>{c}</Badge>
          </label>
        ))}
      </div>

      <div className="flex flex-col gap-3">
        {filtered.length === 0 && (
          <div className="text-sm text-text-muted border border-border rounded-lg px-4 py-6 text-center">
            Tidak ada metrik yang cocok dengan filter ini.
          </div>
        )}
        {filtered.map((m) => (
          <MetricCard key={m.id} id={m.id} isOpen={openIds.has(m.id)} onToggle={() => toggle(m.id)} />
        ))}
      </div>
    </div>
  )
}
