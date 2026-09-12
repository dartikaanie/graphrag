import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateRun } from '@/api/hooks'
import { InfoTooltip } from '@/components/InfoTooltip'
import { PARAM_GLOSSARY } from '@/lib/paramGlossary'
import type { RunCreateParams } from '@/types/run'

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

/**
 * Triggers Condition A, B, C, and D together with IDENTICAL sampling params,
 * for a direct apples-to-apples comparison run. The key correctness detail:
 * each condition's engine_service defaults oversample_pool differently when
 * left unset (A: n_sample*3, B/C/D: n_sample*4 -- see engine_service.py), so
 * this form always computes and sends ONE explicit oversample_pool value to
 * all four requests rather than letting each condition pick its own.
 */
export function RunAllConditionsPage() {
  const navigate = useNavigate()
  const createRun = useCreateRun()

  const [mode, setMode] = useState<'batch' | 'single'>('batch')
  const [nSample, setNSample] = useState(30)
  const [seed, setSeed] = useState(42)
  const [oversamplePool, setOversamplePool] = useState('')
  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('gpt-4o-mini')
  const [topK, setTopK] = useState(5)
  const [nAnchor, setNAnchor] = useState(3)
  const [nSemanticExpansion, setNSemanticExpansion] = useState(3)
  const [requireCitation, setRequireCitation] = useState(true)
  const [fusionMode, setFusionMode] = useState<'trust_weighted' | 'uniform'>('trust_weighted')
  const [fusionWPathTrust, setFusionWPathTrust] = useState(0.7)
  const [fusionWIntrinsic, setFusionWIntrinsic] = useState(0.3)
  const [semanticExpansionTrustCap, setSemanticExpansionTrustCap] = useState(0.4)
  const [enableSemanticExpansion, setEnableSemanticExpansion] = useState(true)
  const [nLowLevel, setNLowLevel] = useState(3)
  const [nHighLevel, setNHighLevel] = useState(3)
  const [requireGrounding, setRequireGrounding] = useState(true)
  const [questionId, setQuestionId] = useState('')
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const effectiveOversamplePool = oversamplePool ? Number(oversamplePool) : nSample * 4

  const handleRunAll = async () => {
    setError(null)
    setStarting(true)
    const shared = {
      mode,
      provider,
      model,
      ...(mode === 'batch'
        ? { n_sample: nSample, seed, oversample_pool: effectiveOversamplePool }
        : { question_id: Number(questionId) }),
    }
    const paramsFor = (condition: 'A' | 'B' | 'C' | 'D'): RunCreateParams => ({
      condition,
      ...shared,
      ...(condition === 'B' ? { top_k: topK, require_citation: requireCitation } : {}),
      ...(condition === 'C'
        ? {
            top_k: topK,
            n_anchor: nAnchor,
            n_semantic_expansion: nSemanticExpansion,
            fusion_mode: fusionMode,
            fusion_w_path_trust: fusionWPathTrust,
            fusion_w_intrinsic: fusionWIntrinsic,
            semantic_expansion_trust_cap: semanticExpansionTrustCap,
            enable_semantic_expansion: enableSemanticExpansion,
            require_grounding: requireGrounding,
          }
        : {}),
      ...(condition === 'D'
        ? {
            top_k: topK,
            n_low_level: nLowLevel,
            n_high_level: nHighLevel,
            require_grounding: requireGrounding,
          }
        : {}),
    })

    try {
      const [a, b, c, d] = await Promise.all([
        createRun.mutateAsync(paramsFor('A')),
        createRun.mutateAsync(paramsFor('B')),
        createRun.mutateAsync(paramsFor('C')),
        createRun.mutateAsync(paramsFor('D')),
      ])
      navigate(`/experiment/all/runs?a=${a.run_id}&b=${b.run_id}&c=${c.run_id}&d=${d.run_id}`)
    } catch (e) {
      setError((e as Error).message)
      setStarting(false)
    }
  }

  const runDisabled = starting || (mode === 'single' && !questionId)

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">Run All Conditions (A + B + C + D)</h1>
      <p className="text-sm text-text-secondary mb-4">
        Runs Condition A, B, C, and D together with identical sampling parameters, for a direct comparison.
      </p>

      <div className="border border-border rounded-lg bg-surface p-4">
        <div className="flex gap-1 mb-4 border-b border-border">
          {(['batch', 'single'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-2 text-sm border-b-2 -mb-px ${
                mode === m ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
              }`}
            >
              {m === 'batch' ? 'Batch' : 'Single Question'}
            </button>
          ))}
        </div>

        {mode === 'batch' ? (
          <div className="grid grid-cols-4 gap-4 mb-2">
            <Field label="n_sample" glossaryKey="n_sample">
              <input type="number" className={inputClass} value={nSample} onChange={(e) => setNSample(Number(e.target.value))} />
            </Field>
            <Field label="seed" glossaryKey="seed">
              <input type="number" className={inputClass} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </Field>
            <Field label="oversample_pool (shared across A/B/C/D)" glossaryKey="oversample_pool">
              <input
                className={inputClass}
                placeholder={`auto (${nSample * 4})`}
                value={oversamplePool}
                onChange={(e) => setOversamplePool(e.target.value)}
              />
            </Field>
            <Field label="provider" glossaryKey="provider">
              <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama (dev/testing only)</option>
              </select>
            </Field>
            <Field label="model" glossaryKey="model">
              <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
            </Field>
            <Field label="top_k (B/C/D)" glossaryKey="top_k">
              <input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            </Field>
            <Field label="n_anchor (C)" glossaryKey="n_anchor">
              <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
            </Field>
            <Field label="n_semantic_expansion (C)" glossaryKey="n_semantic_expansion">
              <input
                type="number" className={inputClass} value={nSemanticExpansion} disabled={!enableSemanticExpansion}
                onChange={(e) => setNSemanticExpansion(Number(e.target.value))}
              />
            </Field>
            <Field label="n_low_level (D)" glossaryKey="n_low_level">
              <input type="number" className={inputClass} value={nLowLevel} onChange={(e) => setNLowLevel(Number(e.target.value))} />
            </Field>
            <Field label="n_high_level (D)" glossaryKey="n_high_level">
              <input type="number" className={inputClass} value={nHighLevel} onChange={(e) => setNHighLevel(Number(e.target.value))} />
            </Field>
          </div>
        ) : (
          <div className="grid grid-cols-4 gap-4 mb-2">
            <Field label="question_id" glossaryKey="question_id">
              <input className={inputClass} placeholder="e.g. 477816" value={questionId} onChange={(e) => setQuestionId(e.target.value)} />
            </Field>
            <Field label="provider" glossaryKey="provider">
              <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama (dev/testing only)</option>
              </select>
            </Field>
            <Field label="model" glossaryKey="model">
              <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
            </Field>
            <Field label="top_k (B/C/D)" glossaryKey="top_k">
              <input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            </Field>
            <Field label="n_anchor (C)" glossaryKey="n_anchor">
              <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
            </Field>
            <Field label="n_semantic_expansion (C)" glossaryKey="n_semantic_expansion">
              <input
                type="number" className={inputClass} value={nSemanticExpansion} disabled={!enableSemanticExpansion}
                onChange={(e) => setNSemanticExpansion(Number(e.target.value))}
              />
            </Field>
            <Field label="n_low_level (D)" glossaryKey="n_low_level">
              <input type="number" className={inputClass} value={nLowLevel} onChange={(e) => setNLowLevel(Number(e.target.value))} />
            </Field>
            <Field label="n_high_level (D)" glossaryKey="n_high_level">
              <input type="number" className={inputClass} value={nHighLevel} onChange={(e) => setNHighLevel(Number(e.target.value))} />
            </Field>
          </div>
        )}

        <label className="flex items-center gap-2 text-sm mt-3">
          <input type="checkbox" checked={requireCitation} onChange={(e) => setRequireCitation(e.target.checked)} />
          <span className="text-text-secondary inline-flex items-center gap-1">
            Condition B: require_citation
            <InfoTooltip text={PARAM_GLOSSARY.require_citation} />
          </span>
        </label>

        <label className="flex items-center gap-2 text-sm mt-2">
          <input type="checkbox" checked={requireGrounding} onChange={(e) => setRequireGrounding(e.target.checked)} />
          <span className="text-text-secondary inline-flex items-center gap-1">
            Condition C/D: Grounding constraint
            <InfoTooltip text={PARAM_GLOSSARY.require_grounding} />
          </span>
        </label>

        <div className="mt-3 border-t border-border pt-3">
          <div className="text-sm font-medium text-text-primary mb-2">Condition C fusion mode (ablation study)</div>
          <div className="grid grid-cols-4 gap-4">
            <Field label="fusion_mode" glossaryKey="fusion_mode">
              <select
                className={inputClass}
                value={fusionMode}
                onChange={(e) => setFusionMode(e.target.value as 'trust_weighted' | 'uniform')}
              >
                <option value="trust_weighted">trust_weighted (default)</option>
                <option value="uniform">uniform (ablation)</option>
              </select>
            </Field>
            <Field label="fusion_w_path_trust" glossaryKey="fusion_w_path_trust">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={fusionWPathTrust} disabled={fusionMode === 'uniform'}
                onChange={(e) => setFusionWPathTrust(Number(e.target.value))}
              />
            </Field>
            <Field label="fusion_w_intrinsic" glossaryKey="fusion_w_intrinsic">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={fusionWIntrinsic} disabled={fusionMode === 'uniform'}
                onChange={(e) => setFusionWIntrinsic(Number(e.target.value))}
              />
            </Field>
            <Field label="semantic_expansion_trust_cap" glossaryKey="semantic_expansion_trust_cap">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={semanticExpansionTrustCap} disabled={fusionMode === 'uniform'}
                onChange={(e) => setSemanticExpansionTrustCap(Number(e.target.value))}
              />
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

        {mode === 'batch' && (
          <div className="mt-2 text-xs text-text-secondary bg-primary-soft border border-primary-border rounded-md px-2 py-1.5">
            All four conditions will use seed={seed} and oversample_pool={effectiveOversamplePool} — identical
            candidate pool and sample, so the same {nSample} questions are evaluated by A, B, C, and D.
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handleRunAll}
            disabled={runDisabled}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {starting ? 'Starting…' : '▶ Run All 4 Conditions'}
          </button>
          {error && <span className="text-sm text-danger">{error}</span>}
        </div>
      </div>
    </div>
  )
}
