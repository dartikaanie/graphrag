import { useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCheckCompleted, useConfigDefaults, useCreateRun } from '@/api/hooks'
import { InfoTooltip } from '@/components/InfoTooltip'
import { PARAM_GLOSSARY } from '@/lib/paramGlossary'
import type { RunCreateParams, RunCondition } from '@/types/run'

function Field({
  label,
  glossaryKey,
  children,
}: {
  label: string
  glossaryKey?: keyof typeof PARAM_GLOSSARY
  children: ReactNode
}) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-text-secondary inline-flex items-center gap-1">
        {label}
        {glossaryKey && <InfoTooltip text={PARAM_GLOSSARY[glossaryKey]} />}
      </span>
      {children}
    </label>
  )
}

const inputClass =
  'px-2.5 py-1.5 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border'

interface FactorialRunConfig {
  runLabel: string
  condition: RunCondition
  fusionMode?: 'trust_weighted' | 'uniform'
  grounding?: boolean
  requireCitation?: boolean
  retrievalVersion?: string
  params: RunCreateParams
}

interface SharedParams {
  provider: string
  model: string
  nSample: number
  seed: number
  oversamplePool: number
  topK: number
}

interface CParams {
  nAnchor: number
  nSemanticExpansion: number
  fusionWPathTrust: number
  fusionWIntrinsic: number
  semanticExpansionTrustCap: number
  enableSemanticExpansion: boolean
}

interface DParams {
  nLowLevel: number
  nHighLevel: number
}

/**
 * The factorial design: retrieval method (A / B-FAISS / C-uniform /
 * C-trust_weighted / D-LightRAG) x grounding (on/off), minus A (which has
 * no grounding axis) -- 1 + 2 + 2 + 2 + 2 = 9 runs. Same shared params
 * (provider/model/seed/n_sample/oversample_pool/top_k/...) across every
 * one of the 9, and the same condition-specific params (C: fusion weights/
 * n_anchor/n_semantic_expansion; D: n_low_level/n_high_level) across every
 * run of that condition -- fusion_mode is the one C param that's an
 * EXPERIMENTAL FACTOR, not a form field, so it's fixed per-run below, not
 * taken from CParams. See docs/Agent prompt grounding factor ui.md.
 */
function buildFactorialRuns(shared: SharedParams, cParams: CParams, dParams: DParams, versions: { c_retrieval_version?: string; d_retrieval_version?: string }): FactorialRunConfig[] {
  const base = {
    mode: 'batch' as const,
    provider: shared.provider,
    model: shared.model,
    n_sample: shared.nSample,
    seed: shared.seed,
    oversample_pool: shared.oversamplePool,
  }

  const cShared = {
    top_k: shared.topK,
    n_anchor: cParams.nAnchor,
    n_semantic_expansion: cParams.nSemanticExpansion,
    fusion_w_path_trust: cParams.fusionWPathTrust,
    fusion_w_intrinsic: cParams.fusionWIntrinsic,
    semantic_expansion_trust_cap: cParams.semanticExpansionTrustCap,
    enable_semantic_expansion: cParams.enableSemanticExpansion,
  }

  const dShared = {
    top_k: shared.topK,
    n_low_level: dParams.nLowLevel,
    n_high_level: dParams.nHighLevel,
  }

  return [
    { runLabel: 'A', condition: 'A', params: { condition: 'A', ...base } },
    {
      runLabel: 'B-plain', condition: 'B', grounding: false, requireCitation: true,
      params: { condition: 'B', ...base, top_k: shared.topK, require_citation: true, require_grounding: false },
    },
    {
      runLabel: 'B-grounded', condition: 'B', grounding: true, requireCitation: true,
      params: { condition: 'B', ...base, top_k: shared.topK, require_citation: true, require_grounding: true },
    },
    {
      runLabel: 'C-uniform-plain', condition: 'C', fusionMode: 'uniform', grounding: false,
      retrievalVersion: versions.c_retrieval_version,
      params: { condition: 'C', ...base, ...cShared, fusion_mode: 'uniform', require_grounding: false },
    },
    {
      runLabel: 'C-uniform-grounded', condition: 'C', fusionMode: 'uniform', grounding: true,
      retrievalVersion: versions.c_retrieval_version,
      params: { condition: 'C', ...base, ...cShared, fusion_mode: 'uniform', require_grounding: true },
    },
    {
      runLabel: 'C-trust-plain', condition: 'C', fusionMode: 'trust_weighted', grounding: false,
      retrievalVersion: versions.c_retrieval_version,
      params: { condition: 'C', ...base, ...cShared, fusion_mode: 'trust_weighted', require_grounding: false },
    },
    {
      runLabel: 'C-trust-grounded', condition: 'C', fusionMode: 'trust_weighted', grounding: true,
      retrievalVersion: versions.c_retrieval_version,
      params: { condition: 'C', ...base, ...cShared, fusion_mode: 'trust_weighted', require_grounding: true },
    },
    {
      runLabel: 'D-plain', condition: 'D', grounding: false, retrievalVersion: versions.d_retrieval_version,
      params: { condition: 'D', ...base, ...dShared, require_grounding: false },
    },
    {
      runLabel: 'D-grounded', condition: 'D', grounding: true, retrievalVersion: versions.d_retrieval_version,
      params: { condition: 'D', ...base, ...dShared, require_grounding: true },
    },
  ]
}

export function FactorialBatchPage() {
  const navigate = useNavigate()
  const createRun = useCreateRun()
  const checkCompleted = useCheckCompleted()
  const { data: defaults } = useConfigDefaults()
  const defaultOversamplePool = defaults?.default_oversample_pool ?? 1536

  const [preset, setPreset] = useState<'pilot' | 'official' | 'custom'>('pilot')
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('gpt-4o-mini')
  const [nSample, setNSample] = useState(10)
  const [seed, setSeed] = useState(42)
  const [oversamplePool, setOversamplePool] = useState('')
  const [topK, setTopK] = useState(5)

  // Condition C parameters -- identical across all 4 C runs (both fusion
  // modes x both grounding settings). Pre-filled with the values used in
  // the existing runs (path=0.7/intrinsic=0.3, n_anchor=3,
  // n_semantic_expansion=3, semantic expansion on).
  const [nAnchor, setNAnchor] = useState(3)
  const [nSemanticExpansion, setNSemanticExpansion] = useState(3)
  const [fusionWPathTrust, setFusionWPathTrust] = useState(0.7)
  const [fusionWIntrinsic, setFusionWIntrinsic] = useState(0.3)
  const [semanticExpansionTrustCap, setSemanticExpansionTrustCap] = useState(0.4)
  const [enableSemanticExpansion, setEnableSemanticExpansion] = useState(true)

  // Condition D parameters -- identical across both D runs.
  const [nLowLevel, setNLowLevel] = useState(3)
  const [nHighLevel, setNHighLevel] = useState(3)

  const [showDryRun, setShowDryRun] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const applyPreset = (p: 'pilot' | 'official') => {
    setPreset(p)
    setNSample(p === 'pilot' ? 10 : 384)
    setSeed(42)
  }

  const effectiveOversamplePool = oversamplePool ? Number(oversamplePool) : defaultOversamplePool

  const runs = useMemo(
    () =>
      buildFactorialRuns(
        { provider, model, nSample, seed, oversamplePool: effectiveOversamplePool, topK },
        { nAnchor, nSemanticExpansion, fusionWPathTrust, fusionWIntrinsic, semanticExpansionTrustCap, enableSemanticExpansion },
        { nLowLevel, nHighLevel },
        { c_retrieval_version: defaults?.c_retrieval_version, d_retrieval_version: defaults?.d_retrieval_version },
      ),
    [
      provider, model, nSample, seed, effectiveOversamplePool, topK,
      nAnchor, nSemanticExpansion, fusionWPathTrust, fusionWIntrinsic, semanticExpansionTrustCap, enableSemanticExpansion,
      nLowLevel, nHighLevel, defaults,
    ],
  )

  // "Skip a run whose identical config already completed" -- computed
  // SERVER-SIDE (POST /api/runs/check-completed), keyed on config_hash --
  // the SAME rule the generator scripts' own resume gate uses
  // (llm.manifest.read_already_done), rather than a client-side field-by-
  // field comparison that could silently drift from it. An older history
  // entry with no config_hash at all (pre-dates this fix) never matches.
  const [alreadyCompleted, setAlreadyCompleted] = useState<Set<string>>(new Set())

  const buildDryRun = async () => {
    setError(null)
    try {
      const { results } = await checkCompleted.mutateAsync(runs.map((r) => r.params))
      const done = new Set<string>()
      runs.forEach((r, i) => {
        if (results[i]?.already_completed) done.add(r.runLabel)
      })
      setAlreadyCompleted(done)
      setShowDryRun(true)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const handleLaunch = async () => {
    setError(null)
    setLaunching(true)
    const toLaunch = runs.filter((r) => !alreadyCompleted.has(r.runLabel))

    // Shared by every run in THIS click -- lets the Hallucination Judge
    // page's run picker group them as one batch without having to infer
    // it (see backend/app/services/batch_grouping_service.py).
    const batchId = `batch-${crypto.randomUUID()}`
    const batchLaunchedAt = new Date().toISOString()

    const settled = await Promise.allSettled(
      toLaunch.map((r) => createRun.mutateAsync({ ...r.params, batch_id: batchId, batch_launched_at: batchLaunchedAt })),
    )

    const launched: { runLabel: string; condition: RunCondition; runId: string }[] = []
    const failed: { runLabel: string; message: string }[] = []
    settled.forEach((result, i) => {
      const r = toLaunch[i]
      if (result.status === 'fulfilled') {
        launched.push({ runLabel: r.runLabel, condition: r.condition, runId: result.value.run_id })
      } else {
        failed.push({ runLabel: r.runLabel, message: String(result.reason) })
      }
    })

    setLaunching(false)
    const runsParam = launched.map((l) => `${l.runLabel}:${l.condition}:${l.runId}`).join(',')
    const failedParam = failed.map((f) => `${f.runLabel}:${f.message}`).join('|')
    const skippedParam = Array.from(alreadyCompleted).join(',')
    navigate(
      `/experiment/factorial/runs?runs=${encodeURIComponent(runsParam)}` +
        (failedParam ? `&failed=${encodeURIComponent(failedParam)}` : '') +
        (skippedParam ? `&skipped=${encodeURIComponent(skippedParam)}` : ''),
    )
  }

  return (
    <div className="max-w-5xl">
      <p className="text-sm text-text-secondary mb-4">
        Fires all 9 runs (A + 4 retrieval methods × grounding on/off) with IDENTICAL shared parameters. Build the
        dry-run preview first — nothing launches until you confirm.
      </p>

      <div className="flex gap-3 mb-4">
        <button
          onClick={() => applyPreset('pilot')}
          className={`px-3 py-1.5 text-sm rounded-md border ${preset === 'pilot' ? 'border-primary bg-primary-border text-primary' : 'border-border-strong text-text-primary hover:bg-bg'}`}
        >
          Pilot (n=10)
        </button>
        <button
          onClick={() => applyPreset('official')}
          className={`px-3 py-1.5 text-sm rounded-md border ${preset === 'official' ? 'border-primary bg-primary-border text-primary' : 'border-border-strong text-text-primary hover:bg-bg'}`}
        >
          Official (n=384)
        </button>
        <button
          onClick={() => setPreset('custom')}
          className={`px-3 py-1.5 text-sm rounded-md border ${preset === 'custom' ? 'border-primary bg-primary-border text-primary' : 'border-border-strong text-text-primary hover:bg-bg'}`}
        >
          Custom
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <Field label="Provider"><input className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)} /></Field>
        <Field label="Model"><input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} /></Field>
        <Field label="n_sample">
          <input type="number" className={inputClass} value={nSample} onChange={(e) => { setNSample(Number(e.target.value)); setPreset('custom') }} />
        </Field>
        <Field label="seed"><input type="number" className={inputClass} value={seed} onChange={(e) => setSeed(Number(e.target.value))} /></Field>
        <Field label="oversample_pool" glossaryKey="oversample_pool">
          <input className={inputClass} placeholder={`${defaultOversamplePool} (default)`} value={oversamplePool} onChange={(e) => setOversamplePool(e.target.value)} />
        </Field>
        <Field label="top_k (B/C/D)" glossaryKey="top_k"><input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></Field>
      </div>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <div className="text-sm font-medium text-text-primary mb-1">Condition C parameters</div>
        <p className="text-xs text-text-secondary mb-3">
          Fusion mode: runs both <span className="font-medium">uniform</span> and{' '}
          <span className="font-medium">trust_weighted</span> as an experimental factor — no selector here, both
          are launched. These parameters apply identically to all 4 C runs.
        </p>
        <div className="grid grid-cols-3 gap-4">
          <Field label="n_anchor" glossaryKey="n_anchor">
            <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
          </Field>
          <Field label="n_semantic_expansion" glossaryKey="n_semantic_expansion">
            <input
              type="number" className={inputClass} value={nSemanticExpansion} disabled={!enableSemanticExpansion}
              onChange={(e) => setNSemanticExpansion(Number(e.target.value))}
            />
          </Field>
          <Field label="fusion_w_path_trust" glossaryKey="fusion_w_path_trust">
            <input type="number" step="0.1" min="0" max="1" className={inputClass} value={fusionWPathTrust} onChange={(e) => setFusionWPathTrust(Number(e.target.value))} />
          </Field>
          <Field label="fusion_w_intrinsic" glossaryKey="fusion_w_intrinsic">
            <input type="number" step="0.1" min="0" max="1" className={inputClass} value={fusionWIntrinsic} onChange={(e) => setFusionWIntrinsic(Number(e.target.value))} />
          </Field>
          <Field label="semantic_expansion_trust_cap" glossaryKey="semantic_expansion_trust_cap">
            <input type="number" step="0.1" min="0" max="1" className={inputClass} value={semanticExpansionTrustCap} onChange={(e) => setSemanticExpansionTrustCap(Number(e.target.value))} />
          </Field>
        </div>
        <label className="flex items-center gap-2 text-sm mt-3">
          <input type="checkbox" checked={enableSemanticExpansion} onChange={(e) => setEnableSemanticExpansion(e.target.checked)} />
          <span className="text-text-secondary inline-flex items-center gap-1">
            Enable semantic expansion
            <InfoTooltip text={PARAM_GLOSSARY.enable_semantic_expansion} />
          </span>
        </label>
      </div>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <div className="text-sm font-medium text-text-primary mb-3">Condition D parameters</div>
        <div className="grid grid-cols-3 gap-4">
          <Field label="n_low_level" glossaryKey="n_low_level">
            <input type="number" className={inputClass} value={nLowLevel} onChange={(e) => setNLowLevel(Number(e.target.value))} />
          </Field>
          <Field label="n_high_level" glossaryKey="n_high_level">
            <input type="number" className={inputClass} value={nHighLevel} onChange={(e) => setNHighLevel(Number(e.target.value))} />
          </Field>
        </div>
      </div>

      <button
        onClick={buildDryRun}
        disabled={checkCompleted.isPending}
        className="px-4 py-2 text-sm border border-border-strong rounded-md text-text-primary hover:bg-bg mb-4 disabled:opacity-50"
      >
        {checkCompleted.isPending ? 'Checking…' : 'Build dry-run preview'}
      </button>

      {showDryRun && (
        <div className="border border-border rounded-lg bg-surface p-4 mb-4 overflow-x-auto">
          <h2 className="text-sm font-semibold text-text-primary mb-1">Dry-run — 9 runs, shared params confirmed identical</h2>
          <p className="text-xs text-text-secondary mb-3">
            prompt_version: <span className="font-medium">{defaults?.prompt_version ?? '—'}</span> · C retrieval_version:{' '}
            <span className="font-medium">{defaults?.c_retrieval_version ?? '—'}</span> · D retrieval_version:{' '}
            <span className="font-medium">{defaults?.d_retrieval_version ?? '—'}</span>
          </p>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-secondary border-b border-border">
                <th className="py-1 pr-3">run_label</th>
                <th className="py-1 pr-3">condition</th>
                <th className="py-1 pr-3">fusion_mode</th>
                <th className="py-1 pr-3">grounding</th>
                <th className="py-1 pr-3">require_citation</th>
                <th className="py-1 pr-3">retrieval_version</th>
                <th className="py-1 pr-3">n_sample</th>
                <th className="py-1 pr-3">seed</th>
                <th className="py-1 pr-3">oversample_pool</th>
                <th className="py-1 pr-3">condition params</th>
                <th className="py-1">status</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.runLabel} className="border-b border-border/50">
                  <td className="py-1 pr-3 font-medium">{r.runLabel}</td>
                  <td className="py-1 pr-3">{r.condition}</td>
                  <td className="py-1 pr-3">{r.fusionMode ?? '—'}</td>
                  <td className="py-1 pr-3">{r.grounding === undefined ? '—' : r.grounding ? 'on' : 'off'}</td>
                  <td className="py-1 pr-3">
                    {/* C/D always build their prompt with require_citation=True
                        internally (see build_graphrag_messages/
                        build_lightrag_messages) -- there's just no CLI toggle for
                        it like B has, so it's not a run_history field to match on
                        for skip-detection, only a display fact. */}
                    {r.condition === 'C' || r.condition === 'D'
                      ? 'always'
                      : r.requireCitation === undefined ? '—' : r.requireCitation ? 'true' : 'false'}
                  </td>
                  <td className="py-1 pr-3">{r.retrievalVersion ?? '—'}</td>
                  <td className="py-1 pr-3">{r.params.n_sample}</td>
                  <td className="py-1 pr-3">{r.params.seed}</td>
                  <td className="py-1 pr-3">{r.params.oversample_pool}</td>
                  <td className="py-1 pr-3 whitespace-nowrap">
                    {r.condition === 'C' &&
                      `top_k=${r.params.top_k}, n_anchor=${r.params.n_anchor}, n_exp=${r.params.n_semantic_expansion}, path=${r.params.fusion_w_path_trust}, intrinsic=${r.params.fusion_w_intrinsic}, cap=${r.params.semantic_expansion_trust_cap}, exp_on=${r.params.enable_semantic_expansion}`}
                    {r.condition === 'D' && `top_k=${r.params.top_k}, n_low=${r.params.n_low_level}, n_high=${r.params.n_high_level}`}
                    {r.condition !== 'C' && r.condition !== 'D' && `top_k=${r.params.top_k ?? '—'}`}
                  </td>
                  <td className="py-1">
                    {alreadyCompleted.has(r.runLabel) ? (
                      <span className="text-text-muted">skip (already completed)</span>
                    ) : (
                      <span className="text-primary">will launch</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {error && <div className="mt-3 text-sm text-danger">{error}</div>}

          <button
            onClick={handleLaunch}
            disabled={launching}
            className="mt-4 px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {launching ? 'Launching…' : `Confirm & Launch ${runs.length - alreadyCompleted.size} run(s)`}
          </button>
        </div>
      )}
    </div>
  )
}
