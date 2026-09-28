import { useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Badge } from '@/components/Badge'
import {
  CATEGORY_LABELS,
  EVALUATION_METRICS,
  type Language,
  type MetricCategory,
} from '@/content/evaluationMetrics'
import type { RunCondition } from '@/types/run'

const CATEGORIES: MetricCategory[] = [
  'Semantic Quality',
  'Citation & Grounding',
  'Hallucination & Faithfulness',
  'Retrieval Quality',
  'Efficiency',
]

const ALL_CONDITIONS: RunCondition[] = ['A', 'B', 'C', 'D']

const LANGUAGE_STORAGE_KEY = 'metrics-reference-language'

function readStoredLanguage(): Language {
  try {
    const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY)
    return stored === 'en' ? 'en' : 'id'
  } catch {
    return 'id'
  }
}

// UI copy for chrome that isn't per-metric content (headings, labels,
// buttons). Per-metric content (whatItMeasures, limitations, etc.) is
// localized directly on EVALUATION_METRICS via LocalizedText.
const UI_TEXT: Record<Language, Record<string, string>> = {
  id: {
    pageTitle: 'Metrik Evaluasi',
    pageIntro:
      'Referensi definisi setiap metrik evaluasi yang dipakai di penelitian ini — apa yang diukur, bagaimana dihitung, di kondisi mana berlaku, dan batasannya. Halaman ini bersifat dokumentasi saja (tidak menjalankan apa pun); untuk desain retrieval/prompting tiap kondisi, lihat halaman Methodology.',
    methodologyLink: 'Methodology',
    all: 'Semua',
    condition: 'Kondisi:',
    onlyWithReferences: 'Tampilkan hanya metrik dengan referensi akademik',
    noMatch: 'Tidak ada metrik yang cocok dengan filter ini.',
    validationMetric: 'Metrik validasi',
    howComputed: "Bagaimana dihitung",
    formula: 'Formula',
    howToRead: 'Cara membaca',
    limitations: 'Batasan',
    references: 'Referensi',
    seeAlso: 'Lihat juga:',
    implementedIn: 'Diimplementasikan di:',
  },
  en: {
    pageTitle: 'Evaluation Metrics',
    pageIntro:
      "Reference definitions for every evaluation metric used in this research — what it measures, how it's computed, which conditions it applies to, and its limitations. This page is documentation only (it doesn't run anything); for each condition's retrieval/prompting design, see the Methodology page.",
    methodologyLink: 'Methodology',
    all: 'All',
    condition: 'Condition:',
    onlyWithReferences: 'Show only metrics with academic references',
    noMatch: 'No metrics match this filter.',
    validationMetric: 'Validation metric',
    howComputed: "How it's computed",
    formula: 'Formula',
    howToRead: 'How to read it',
    limitations: 'Limitations',
    references: 'References',
    seeAlso: 'See also:',
    implementedIn: 'Implemented in:',
  },
}

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

function MetricCard({
  id,
  isOpen,
  onToggle,
  lang,
}: {
  id: string
  isOpen: boolean
  onToggle: () => void
  lang: Language
}) {
  const metric = EVALUATION_METRICS.find((m) => m.id === id)
  if (!metric) return null

  const t = UI_TEXT[lang]
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
            {metric.isValidationMetric && <Badge tone="neutral">{t.validationMetric}</Badge>}
            {metric.appliesTo.map((c) => (
              <Badge key={c} tone={CONDITION_BADGE_TONE[c]}>{c}</Badge>
            ))}
          </div>
          <p className="text-sm text-text-secondary">{metric.whatItMeasures[lang]}</p>
        </div>
        <span className="text-text-muted text-sm shrink-0 mt-0.5">{isOpen ? '−' : '+'}</span>
      </button>

      {isOpen && (
        <div className="px-4 pb-4 flex flex-col gap-3 border-t border-border pt-3">
          <div>
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{t.howComputed}</h3>
            <p className="text-sm text-text-secondary">{metric.howComputed[lang]}</p>
          </div>

          {metric.formula && (
            <div>
              <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{t.formula}</h3>
              <code className="block text-xs bg-bg border border-border rounded-md px-3 py-2 text-text-primary whitespace-pre-wrap">
                {metric.formula}
              </code>
            </div>
          )}

          <div>
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{t.howToRead}</h3>
            <p className="text-sm text-text-secondary">{metric.interpretationGuide[lang]}</p>
          </div>

          <div className="border-l-2 border-border-strong pl-3">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{t.limitations}</h3>
            <p className="text-sm text-text-secondary">{metric.limitations[lang]}</p>
          </div>

          {(metric.references?.length ?? 0) > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1">{t.references}</h3>
              <div className="flex flex-col gap-2">
                {metric.references!.map((ref, i) => (
                  <div key={i} className="text-sm text-text-secondary">
                    <p>
                      {ref.authors} {ref.title} — {ref.year}. {ref.venue}.
                    </p>
                    {ref.doi && (
                      <a
                        href={`https://doi.org/${ref.doi}`}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        https://doi.org/{ref.doi}
                      </a>
                    )}
                    <p className="text-xs text-text-muted italic mt-0.5">{ref.relevance[lang]}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {related.length > 0 && (
            <div className="text-xs text-text-secondary">
              {t.seeAlso}{' '}
              {related.map((r, i) => (
                <span key={r.id}>
                  <a href={`#${r.id}`} className="text-primary hover:underline">{r.name}</a>
                  {i < related.length - 1 ? ', ' : ''}
                </span>
              ))}
            </div>
          )}

          <div className="text-xs text-text-muted font-mono">
            {t.implementedIn} {metric.implementedIn.join(', ')}
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
  const [onlyWithReferences, setOnlyWithReferences] = useState(false)
  const [lang, setLang] = useState<Language>(readStoredLanguage)
  const t = UI_TEXT[lang]

  useEffect(() => {
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, lang)
    } catch {
      // localStorage unavailable (private mode, etc.) -- language choice
      // just won't persist across reloads, which is fine.
    }
  }, [lang])

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
          m.appliesTo.some((c) => activeConditions.has(c)) &&
          (!onlyWithReferences || (m.references?.length ?? 0) > 0),
      ),
    [activeCategory, activeConditions, onlyWithReferences],
  )

  return (
    <div className="max-w-4xl">
      <div className="flex items-start justify-between gap-4 mb-1">
        <h1 className="text-lg font-semibold text-text-primary">{t.pageTitle}</h1>
        <div className="flex items-center gap-1 text-xs shrink-0 border border-border rounded-md overflow-hidden">
          {(['id', 'en'] as Language[]).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`px-2 py-1 uppercase ${
                lang === l ? 'bg-primary text-white' : 'text-text-secondary hover:bg-bg'
              }`}
            >
              {l}
            </button>
          ))}
        </div>
      </div>
      <p className="text-sm text-text-secondary mb-6">
        {t.pageIntro}{' '}
        <a href="/methodology" className="text-primary hover:underline">{t.methodologyLink}</a>.
      </p>

      <div className="flex gap-1 mb-4 border-b border-border overflow-x-auto">
        <button
          onClick={() => setActiveCategory('All')}
          className={`px-3 py-2 text-sm border-b-2 -mb-px whitespace-nowrap ${
            activeCategory === 'All' ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
          }`}
        >
          {t.all}
        </button>
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-3 py-2 text-sm border-b-2 -mb-px whitespace-nowrap ${
              activeCategory === cat ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
            }`}
          >
            {CATEGORY_LABELS[cat][lang]}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4 mb-2 text-sm">
        <span className="text-text-secondary">{t.condition}</span>
        {ALL_CONDITIONS.map((c) => (
          <label key={c} className="flex items-center gap-1.5">
            <input type="checkbox" checked={activeConditions.has(c)} onChange={() => toggleCondition(c)} />
            <Badge tone={CONDITION_BADGE_TONE[c]}>{c}</Badge>
          </label>
        ))}
      </div>

      <div className="flex items-center gap-1.5 mb-4 text-sm">
        <label className="flex items-center gap-1.5">
          <input
            type="checkbox"
            checked={onlyWithReferences}
            onChange={() => setOnlyWithReferences((v) => !v)}
          />
          <span className="text-text-secondary">{t.onlyWithReferences}</span>
        </label>
      </div>

      <div className="flex flex-col gap-3">
        {filtered.length === 0 && (
          <div className="text-sm text-text-muted border border-border rounded-lg px-4 py-6 text-center">
            {t.noMatch}
          </div>
        )}
        {filtered.map((m) => (
          <MetricCard key={m.id} id={m.id} isOpen={openIds.has(m.id)} onToggle={() => toggle(m.id)} lang={lang} />
        ))}
      </div>
    </div>
  )
}
