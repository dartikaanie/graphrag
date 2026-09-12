import { useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCreateRun } from '@/api/hooks'
import type { RunCreateParams } from '@/types/run'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-text-secondary">{label}</span>
      {children}
    </label>
  )
}

const inputClass =
  'px-2.5 py-1.5 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border'

/**
 * Triggers Condition A, B, and C together with IDENTICAL sampling params,
 * for a direct apples-to-apples comparison run. The key correctness detail:
 * each condition's engine_service defaults oversample_pool differently when
 * left unset (A: n_sample*3, B/C: n_sample*4 -- see engine_service.py), so
 * this form always computes and sends ONE explicit oversample_pool value to
 * all three requests rather than letting each condition pick its own.
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
    const paramsFor = (condition: 'A' | 'B' | 'C'): RunCreateParams => ({
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
          }
        : {}),
    })

    try {
      const [a, b, c] = await Promise.all([
        createRun.mutateAsync(paramsFor('A')),
        createRun.mutateAsync(paramsFor('B')),
        createRun.mutateAsync(paramsFor('C')),
      ])
      navigate(`/experiment/all/runs?a=${a.run_id}&b=${b.run_id}&c=${c.run_id}`)
    } catch (e) {
      setError((e as Error).message)
      setStarting(false)
    }
  }

  const runDisabled = starting || (mode === 'single' && !questionId)

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">Run All Conditions (A + B + C)</h1>
      <p className="text-sm text-text-secondary mb-4">
        Runs Condition A, B, and C together with identical sampling parameters, for a direct comparison.
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
            <Field label="n_sample">
              <input type="number" className={inputClass} value={nSample} onChange={(e) => setNSample(Number(e.target.value))} />
            </Field>
            <Field label="seed">
              <input type="number" className={inputClass} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </Field>
            <Field label="oversample_pool (shared across A/B/C)">
              <input
                className={inputClass}
                placeholder={`auto (${nSample * 4})`}
                value={oversamplePool}
                onChange={(e) => setOversamplePool(e.target.value)}
              />
            </Field>
            <Field label="provider">
              <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama (dev/testing only)</option>
              </select>
            </Field>
            <Field label="model">
              <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
            </Field>
            <Field label="top_k (B/C)">
              <input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            </Field>
            <Field label="n_anchor (C)">
              <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
            </Field>
            <Field label="n_semantic_expansion (C)">
              <input type="number" className={inputClass} value={nSemanticExpansion} onChange={(e) => setNSemanticExpansion(Number(e.target.value))} />
            </Field>
          </div>
        ) : (
          <div className="grid grid-cols-4 gap-4 mb-2">
            <Field label="question_id">
              <input className={inputClass} placeholder="e.g. 477816" value={questionId} onChange={(e) => setQuestionId(e.target.value)} />
            </Field>
            <Field label="provider">
              <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="ollama">ollama (dev/testing only)</option>
              </select>
            </Field>
            <Field label="model">
              <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
            </Field>
            <Field label="top_k (B/C)">
              <input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
            </Field>
            <Field label="n_anchor (C)">
              <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
            </Field>
            <Field label="n_semantic_expansion (C)">
              <input type="number" className={inputClass} value={nSemanticExpansion} onChange={(e) => setNSemanticExpansion(Number(e.target.value))} />
            </Field>
          </div>
        )}

        <label className="flex items-center gap-2 text-sm mt-3">
          <input type="checkbox" checked={requireCitation} onChange={(e) => setRequireCitation(e.target.checked)} />
          <span className="text-text-secondary">
            Condition B: require [SO-&lt;id&gt;] citations (same format as Condition C, for NF2 comparison)
          </span>
        </label>

        <div className="mt-3 border-t border-border pt-3">
          <div className="text-sm font-medium text-text-primary mb-2">Condition C fusion mode (ablation study)</div>
          <div className="grid grid-cols-4 gap-4">
            <Field label="fusion_mode">
              <select
                className={inputClass}
                value={fusionMode}
                onChange={(e) => setFusionMode(e.target.value as 'trust_weighted' | 'uniform')}
              >
                <option value="trust_weighted">trust_weighted (default)</option>
                <option value="uniform">uniform (ablation)</option>
              </select>
            </Field>
            <Field label="fusion_w_path_trust">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={fusionWPathTrust} disabled={fusionMode === 'uniform'}
                onChange={(e) => setFusionWPathTrust(Number(e.target.value))}
              />
            </Field>
            <Field label="fusion_w_intrinsic">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={fusionWIntrinsic} disabled={fusionMode === 'uniform'}
                onChange={(e) => setFusionWIntrinsic(Number(e.target.value))}
              />
            </Field>
            <Field label="semantic_expansion_trust_cap">
              <input
                type="number" step="0.1" min="0" max="1" className={inputClass}
                value={semanticExpansionTrustCap} disabled={fusionMode === 'uniform'}
                onChange={(e) => setSemanticExpansionTrustCap(Number(e.target.value))}
              />
            </Field>
          </div>
        </div>

        {mode === 'batch' && (
          <div className="mt-2 text-xs text-text-secondary bg-primary-soft border border-primary-border rounded-md px-2 py-1.5">
            All three conditions will use seed={seed} and oversample_pool={effectiveOversamplePool} — identical
            candidate pool and sample, so the same {nSample} questions are evaluated by A, B, and C.
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handleRunAll}
            disabled={runDisabled}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {starting ? 'Starting…' : '▶ Run All 3 Conditions'}
          </button>
          {error && <span className="text-sm text-danger">{error}</span>}
        </div>
      </div>
    </div>
  )
}
