import { useState, type ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useCreateRun } from '@/api/hooks'
import { InfoTooltip } from '@/components/InfoTooltip'
import { PARAM_GLOSSARY } from '@/lib/paramGlossary'
import type { RunCondition, RunCreateParams } from '@/types/run'

const CONDITION_LABELS: Record<RunCondition, string> = {
  A: 'Condition A — Pure LLM',
  B: 'Condition B — LLM + RAG',
  C: 'Condition C — LLM + GraphRAG',
}

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

export function RunConditionPage() {
  const { condition: conditionParam = 'a' } = useParams()
  const condition = conditionParam.toUpperCase() as RunCondition
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

  const isRag = condition === 'B' || condition === 'C'
  const isGraph = condition === 'C'
  const isConditionB = condition === 'B'

  const handleRun = () => {
    const params: RunCreateParams = {
      condition,
      mode,
      provider,
      model,
      ...(isConditionB ? { require_citation: requireCitation } : {}),
      ...(isGraph
        ? {
            fusion_mode: fusionMode,
            fusion_w_path_trust: fusionWPathTrust,
            fusion_w_intrinsic: fusionWIntrinsic,
            semantic_expansion_trust_cap: semanticExpansionTrustCap,
          }
        : {}),
      ...(mode === 'batch'
        ? {
            n_sample: nSample,
            seed,
            oversample_pool: oversamplePool ? Number(oversamplePool) : null,
            ...(isRag ? { top_k: topK } : {}),
            ...(isGraph ? { n_anchor: nAnchor, n_semantic_expansion: nSemanticExpansion } : {}),
          }
        : { question_id: Number(questionId) }),
    }
    createRun.mutate(params, {
      onSuccess: (res) => navigate(`/experiment/${conditionParam}/runs/${res.run_id}`),
    })
  }

  const runDisabled = createRun.isPending || (mode === 'single' && !questionId)

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">{CONDITION_LABELS[condition]}</h1>

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
            <Field label="oversample_pool (optional)" glossaryKey="oversample_pool">
              <input className={inputClass} placeholder="auto (3-4x n_sample)" value={oversamplePool} onChange={(e) => setOversamplePool(e.target.value)} />
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
            {isRag && (
              <Field label="top_k" glossaryKey="top_k">
                <input type="number" className={inputClass} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
              </Field>
            )}
            {isGraph && (
              <>
                <Field label="n_anchor" glossaryKey="n_anchor">
                  <input type="number" className={inputClass} value={nAnchor} onChange={(e) => setNAnchor(Number(e.target.value))} />
                </Field>
                <Field label="n_semantic_expansion" glossaryKey="n_semantic_expansion">
                  <input type="number" className={inputClass} value={nSemanticExpansion} onChange={(e) => setNSemanticExpansion(Number(e.target.value))} />
                </Field>
              </>
            )}
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
          </div>
        )}

        {isGraph && (
          <div className="mt-3 border-t border-border pt-3">
            <div className="text-sm font-medium text-text-primary mb-2">Fusion mode (ablation study)</div>
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
            {fusionMode === 'uniform' && (
              <div className="mt-2 text-xs text-text-secondary">
                Uniform mode ignores trust weights entirely — ranking is determined purely by retrieval discovery order (graph traversal before semantic expansion, hop 1 before hop 2).
              </div>
            )}
          </div>
        )}

        {isConditionB && (
          <label className="flex items-center gap-2 text-sm mt-3">
            <input type="checkbox" checked={requireCitation} onChange={(e) => setRequireCitation(e.target.checked)} />
            <span className="text-text-secondary inline-flex items-center gap-1">
              require_citation
              <InfoTooltip text={PARAM_GLOSSARY.require_citation} />
              — same format as Condition C, for NF2 comparison
            </span>
          </label>
        )}

        {(seed !== 42 || (oversamplePool && oversamplePool !== '')) && mode === 'batch' && (
          <div className="mt-2 text-xs text-warning bg-white border border-warning/40 rounded-md px-2 py-1.5">
            ⚠ seed/oversample_pool differ from the default (42 / auto) — use the same values across A/B/C for a fair comparison.
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={runDisabled}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {createRun.isPending ? 'Starting…' : '▶ Run'}
          </button>
          {createRun.isError && <span className="text-sm text-danger">{(createRun.error as Error).message}</span>}
        </div>
      </div>
    </div>
  )
}
