import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useCreateJudgeRun, useJudgeAvailableInputs, useJudgeRunPoll, useJudgeRunsForInput, useSettings } from '@/api/hooks'
import { Badge } from '@/components/Badge'
import { InfoTooltip } from '@/components/InfoTooltip'
import { fmtDuration } from '@/lib/format'
import { PARAM_GLOSSARY } from '@/lib/paramGlossary'
import type { JudgeRunCreateParams } from '@/types/judge'

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

function kappaTone(kappa: number): 'success' | 'warning' | 'danger' {
  if (kappa > 0.6) return 'success'
  if (kappa >= 0.4) return 'warning'
  return 'danger'
}

const LABEL_TONE: Record<string, 'success' | 'warning' | 'danger'> = {
  FAKTUAL: 'success',
  HALUSINASI_SEBAGIAN: 'warning',
  HALUSINASI_PENUH: 'danger',
}

function historyLinkFor(condition: string): string {
  // History detail pages are addressed by history_id (a hash of
  // run_started_at+output_path), not by path -- we don't have that hash
  // here, so send the user to the filtered History list for this
  // condition instead of guessing a history_id.
  return `/history?condition=${condition.toUpperCase()}`
}

export function RunJudgePage() {
  const [searchParams] = useSearchParams()
  const prefillInputPath = searchParams.get('prefill_input_path') ?? ''
  const prefillCondition = (searchParams.get('condition') ?? 'C').toUpperCase() as 'A' | 'B' | 'C' | 'D'

  const { data: settings } = useSettings()
  const createJudgeRun = useCreateJudgeRun()

  const [condition, setCondition] = useState<'A' | 'B' | 'C' | 'D'>(prefillCondition)
  const { data: availableInputs, isLoading: inputsLoading } = useJudgeAvailableInputs(
    prefillInputPath ? undefined : condition,
  )

  // Instead of picking a raw filename, narrow down to the result file via
  // its identifying fields -- provider/model/n_sample/seed -- each option
  // list cascades from the ones already chosen, and inputPath resolves
  // automatically once the filters narrow the match to exactly one file.
  // Skipped entirely when arriving with a prefilled path (from History's
  // "Judge now" button) -- that file is already known.
  const [filterProvider, setFilterProvider] = useState('')
  const [filterModel, setFilterModel] = useState('')
  const [filterNSample, setFilterNSample] = useState('')
  const [filterSeed, setFilterSeed] = useState('')
  const [inputPath, setInputPath] = useState(prefillInputPath)

  const items = availableInputs?.items ?? []

  const providerOptions = useMemo(
    () => Array.from(new Set(items.map((i) => i.provider).filter((v): v is string => !!v))).sort(),
    [items],
  )
  const afterProvider = useMemo(
    () => items.filter((i) => !filterProvider || i.provider === filterProvider),
    [items, filterProvider],
  )
  const modelOptions = useMemo(
    () => Array.from(new Set(afterProvider.map((i) => i.model).filter((v): v is string => !!v))).sort(),
    [afterProvider],
  )
  const afterModel = useMemo(
    () => afterProvider.filter((i) => !filterModel || i.model === filterModel),
    [afterProvider, filterModel],
  )
  const nSampleOptions = useMemo(
    () => Array.from(new Set(afterModel.map((i) => i.n_sample).filter((v): v is number => v != null))).sort((a, b) => a - b),
    [afterModel],
  )
  const afterNSample = useMemo(
    () => afterModel.filter((i) => !filterNSample || i.n_sample === Number(filterNSample)),
    [afterModel, filterNSample],
  )
  const seedOptions = useMemo(
    () => Array.from(new Set(afterNSample.map((i) => i.seed).filter((v): v is number => v != null))).sort((a, b) => a - b),
    [afterNSample],
  )
  const matches = useMemo(
    () => afterNSample.filter((i) => !filterSeed || i.seed === Number(filterSeed)),
    [afterNSample, filterSeed],
  )

  // Reset dependent filters whenever an upstream one changes -- a model
  // chosen under the old provider, say, is meaningless once the provider
  // changes.
  useEffect(() => {
    setFilterModel('')
  }, [filterProvider])
  useEffect(() => {
    setFilterNSample('')
  }, [filterModel])
  useEffect(() => {
    setFilterSeed('')
  }, [filterNSample])

  useEffect(() => {
    if (prefillInputPath) return
    setInputPath(matches.length === 1 ? matches[0].path : '')
  }, [matches, prefillInputPath])

  const [judgeProvider, setJudgeProvider] = useState('openai')
  const [judgeModel, setJudgeModel] = useState('')
  const [judgeTemperature, setJudgeTemperature] = useState(0.1)
  const [majorityRounds, setMajorityRounds] = useState(1)
  const [force, setForce] = useState(false)
  const [kappaValidation, setKappaValidation] = useState(false)
  const [secondaryJudgeProvider, setSecondaryJudgeProvider] = useState('openai')
  const [secondaryJudgeModel, setSecondaryJudgeModel] = useState('')
  const [kappaSampleSize, setKappaSampleSize] = useState(5)

  const [runId, setRunId] = useState<string | undefined>(undefined)
  const { run, connectionError } = useJudgeRunPoll(runId)

  // Has this exact result file already been judged (with any config)? Lets
  // the page show cached results + a "re-judge" option instead of a plain
  // Run button, per the current filter values below (not tied to the
  // current judgeProvider/judgeModel/... choice -- those are shown too, so
  // the user can see whether THIS SPECIFIC config was already run).
  const { data: existingEvaluations } = useJudgeRunsForInput(inputPath || undefined)
  const matchingEvaluation = existingEvaluations?.items.find(
    (e) =>
      e.judge_provider === judgeProvider &&
      e.judge_model === judgeModel &&
      e.judge_temperature === judgeTemperature &&
      e.majority_rounds === majorityRounds,
  )

  // Pre-fill from Settings once loaded -- still overridable per-run below.
  useEffect(() => {
    if (!settings) return
    setJudgeProvider(settings.judge_provider)
    setJudgeModel(settings.judge_model)
    setJudgeTemperature(settings.judge_temperature)
    setMajorityRounds(settings.judge_majority_rounds)
    setSecondaryJudgeProvider(settings.secondary_judge_provider)
    setSecondaryJudgeModel(settings.secondary_judge_model)
    setKappaSampleSize(settings.kappa_sample_size)
  }, [settings])

  // Reset all file filters whenever condition changes -- the previous
  // selection belongs to a different condition's results folder.
  useEffect(() => {
    if (prefillInputPath) return
    setFilterProvider('')
    setFilterModel('')
    setFilterNSample('')
    setFilterSeed('')
  }, [condition, prefillInputPath])

  const judgeModelsIdentical =
    kappaValidation && judgeProvider === secondaryJudgeProvider && judgeModel === secondaryJudgeModel && judgeModel !== ''

  const handleRun = (forceRun: boolean) => {
    const params: JudgeRunCreateParams = {
      input_path: inputPath,
      condition,
      judge_provider: judgeProvider,
      judge_model: judgeModel,
      judge_temperature: judgeTemperature,
      majority_rounds: majorityRounds,
      force: forceRun,
      kappa_validation: kappaValidation,
      ...(kappaValidation
        ? {
            secondary_judge_provider: secondaryJudgeProvider,
            secondary_judge_model: secondaryJudgeModel,
            kappa_sample_size: kappaSampleSize,
          }
        : {}),
    }
    createJudgeRun.mutate(params, {
      onSuccess: (res) => setRunId(res.run_id),
    })
  }

  const runDisabled = createJudgeRun.isPending || !inputPath || (kappaValidation && !secondaryJudgeModel)
  const isTerminal = run ? ['completed', 'failed', 'cancelled'].includes(run.status) : false

  const nCached = run?.results.filter((r) => r.status === 'cached').length ?? 0
  const nEvaluated = run?.results.filter((r) => r.status === 'evaluated').length ?? 0
  const nFailed = run?.results.filter((r) => r.status === 'failed').length ?? 0

  return (
    <div className="max-w-3xl">
      <h1 className="text-lg font-semibold text-text-primary mb-1">LLM-as-Judge Evaluation</h1>
      <p className="text-sm text-text-secondary mb-4">
        Post-hoc evaluation of an already-completed condition's results: Faithfulness, Answer Relevance, and a
        3-class Hallucination Rate, with optional Cohen's Kappa validation against a second judge. Does not re-run
        the original condition.
      </p>

      <div className="border border-border rounded-lg bg-surface p-4">
        {prefillInputPath ? (
          <div className="text-sm text-text-secondary mb-4">
            Judging: <span className="font-mono text-text-primary">{prefillInputPath}</span>{' '}
            <Badge tone="primary">Condition {condition}</Badge>
          </div>
        ) : (
          <>
            <Field label="Condition" glossaryKey="judge_condition">
              <select
                className={`${inputClass} max-w-[240px]`}
                value={condition}
                onChange={(e) => setCondition(e.target.value as 'A' | 'B' | 'C' | 'D')}
              >
                <option value="A">A — Pure LLM</option>
                <option value="B">B — LLM + RAG</option>
                <option value="C">C — LLM + GraphRAG</option>
                <option value="D">D — Dual-Level Retrieval</option>
              </select>
            </Field>

            <div className="text-xs text-text-secondary mt-4 mb-1.5 inline-flex items-center gap-1">
              Result file to judge -- narrow down by its identifying fields:
              <InfoTooltip text={PARAM_GLOSSARY.judge_input_file} />
            </div>
            <div className="grid grid-cols-4 gap-4 mb-2">
              <Field label="Provider" glossaryKey="judge_input_provider">
                <select className={inputClass} value={filterProvider} onChange={(e) => setFilterProvider(e.target.value)} disabled={inputsLoading}>
                  <option value="">Any</option>
                  {providerOptions.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </Field>
              <Field label="Model" glossaryKey="judge_input_model">
                <select className={inputClass} value={filterModel} onChange={(e) => setFilterModel(e.target.value)} disabled={inputsLoading}>
                  <option value="">Any</option>
                  {modelOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </Field>
              <Field label="N Sample" glossaryKey="judge_input_n_sample">
                <select className={inputClass} value={filterNSample} onChange={(e) => setFilterNSample(e.target.value)} disabled={inputsLoading}>
                  <option value="">Any</option>
                  {nSampleOptions.map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </Field>
              <Field label="Seed" glossaryKey="judge_input_seed">
                <select className={inputClass} value={filterSeed} onChange={(e) => setFilterSeed(e.target.value)} disabled={inputsLoading}>
                  <option value="">Any</option>
                  {seedOptions.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </Field>
            </div>

            {inputsLoading ? (
              <div className="text-xs text-text-muted mb-4">Loading available result files…</div>
            ) : matches.length === 0 ? (
              <div className="text-xs text-danger mb-4">No result file matches this combination.</div>
            ) : matches.length === 1 ? (
              <div className="text-xs text-success mb-4">✓ Resolved to: {matches[0].filename}</div>
            ) : (
              <Field label={`Multiple files match (${matches.length}) -- pick one`}>
                <select className={inputClass} value={inputPath} onChange={(e) => setInputPath(e.target.value)}>
                  <option value="">Select a file</option>
                  {matches.map((item) => (
                    <option key={item.path} value={item.path}>{item.filename}</option>
                  ))}
                </select>
              </Field>
            )}
          </>
        )}

        <div className="grid grid-cols-3 gap-4 mb-2">
          <Field label="Judge Provider" glossaryKey="judge_provider">
            <select className={inputClass} value={judgeProvider} onChange={(e) => setJudgeProvider(e.target.value)}>
              <option value="openai">openai</option>
              <option value="anthropic">anthropic</option>
              <option value="ollama">ollama (dev/testing only)</option>
            </select>
          </Field>
          <Field label="Judge Model" glossaryKey="judge_model">
            <input className={inputClass} value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)} />
          </Field>
          <Field label="Judge Temperature" glossaryKey="judge_temperature">
            <input
              type="number" min="0" max="1" step="0.1" className={inputClass}
              value={judgeTemperature} onChange={(e) => setJudgeTemperature(Number(e.target.value))}
            />
          </Field>
        </div>

        <Field label="Majority Rounds" glossaryKey="judge_majority_rounds">
          <input
            type="number" min="1" className={`${inputClass} max-w-[120px]`}
            value={majorityRounds} onChange={(e) => setMajorityRounds(Number(e.target.value))}
          />
        </Field>

        <label className="flex items-center gap-2 text-sm mt-3">
          <input type="checkbox" checked={kappaValidation} onChange={(e) => setKappaValidation(e.target.checked)} />
          <span className="text-text-secondary inline-flex items-center gap-1">
            Run Cohen's Kappa validation
            <InfoTooltip text={PARAM_GLOSSARY.kappa_validation} />
          </span>
        </label>

        {kappaValidation && (
          <div className="mt-3 border-t border-border pt-3">
            <div className="grid grid-cols-3 gap-4">
              <Field label="Secondary Judge Provider" glossaryKey="secondary_judge_provider">
                <select className={inputClass} value={secondaryJudgeProvider} onChange={(e) => setSecondaryJudgeProvider(e.target.value)}>
                  <option value="openai">openai</option>
                  <option value="anthropic">anthropic</option>
                  <option value="ollama">ollama (dev/testing only)</option>
                </select>
              </Field>
              <Field label="Secondary Judge Model" glossaryKey="secondary_judge_model">
                <input className={inputClass} value={secondaryJudgeModel} onChange={(e) => setSecondaryJudgeModel(e.target.value)} />
              </Field>
              <Field label="Kappa Sample Size" glossaryKey="kappa_sample_size">
                <input
                  type="number" min="1" className={inputClass}
                  value={kappaSampleSize} onChange={(e) => setKappaSampleSize(Number(e.target.value))}
                />
              </Field>
            </div>
            {judgeModelsIdentical && (
              <div className="mt-2 text-xs text-warning bg-white border border-warning/40 rounded-md px-2 py-1.5">
                ⚠ Primary and secondary judge are identical ({judgeProvider}/{judgeModel}) — Kappa validation with
                the same model will not detect self-enhancement bias, only its own internal variance.
              </div>
            )}
          </div>
        )}

        {matchingEvaluation && !runId && (
          <div className="mt-3 border border-primary-border bg-primary-soft rounded-md px-3 py-2.5">
            <div className="text-sm text-text-primary mb-1">
              ✓ Already judged with this exact config ({matchingEvaluation.n_total} questions,{' '}
              {matchingEvaluation.evaluated_at ? new Date(matchingEvaluation.evaluated_at).toLocaleString() : 'unknown time'}).
            </div>
            <div className="text-xs text-text-secondary">
              FAKTUAL {matchingEvaluation.pct_faktual}% · SEBAGIAN {matchingEvaluation.pct_sebagian}% · PENUH{' '}
              {matchingEvaluation.pct_penuh}%
            </div>
          </div>
        )}

        <div className="mt-4 flex items-center gap-3">
          {matchingEvaluation ? (
            <button
              onClick={() => handleRun(true)}
              disabled={runDisabled}
              className="px-4 py-2 text-sm border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50"
            >
              {createJudgeRun.isPending ? 'Starting…' : '↻ Re-judge (force)'}
            </button>
          ) : (
            <button
              onClick={() => handleRun(force)}
              disabled={runDisabled}
              className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
            >
              {createJudgeRun.isPending ? 'Starting…' : '▶ Run Judge'}
            </button>
          )}
          {!matchingEvaluation && (
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              <span className="text-text-secondary inline-flex items-center gap-1">
                Force (start fresh, ignore any partial resume)
                <InfoTooltip text={PARAM_GLOSSARY.judge_force} />
              </span>
            </label>
          )}
          {createJudgeRun.isError && <span className="text-sm text-danger">{(createJudgeRun.error as Error).message}</span>}
        </div>
      </div>

      {runId && (
        <div className="mt-6 border border-border rounded-lg bg-surface p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-text-primary">Run {runId}</h2>
            {run && <Badge tone={run.status === 'completed' ? 'success' : run.status === 'failed' ? 'danger' : 'neutral'}>{run.status}</Badge>}
          </div>

          {connectionError && !isTerminal && <div className="text-xs text-warning mb-2">Polling disconnected, retrying…</div>}

          {run && !isTerminal && (
            <div className="text-sm text-text-secondary mb-2">
              Progress: {run.progress.current} / {run.progress.total} ({nCached} from cache, {nEvaluated} newly evaluated
              {nFailed > 0 ? `, ${nFailed} failed` : ''})
            </div>
          )}

          {run?.status === 'failed' && run.error && (
            <div className="text-xs text-danger border border-danger/40 bg-white rounded-md p-2">{run.error}</div>
          )}

          {run?.summary && (
            <>
              <div className="grid grid-cols-3 gap-4 mt-2">
                <div className="border border-border rounded-md px-3 py-2">
                  <div className="text-xs text-text-secondary mb-1">Hallucination Distribution</div>
                  <div className="flex flex-col gap-1 text-sm">
                    <div className="flex justify-between"><Badge tone={LABEL_TONE.FAKTUAL}>FAKTUAL</Badge><span>{run.summary.pct_faktual ?? '—'}%</span></div>
                    <div className="flex justify-between"><Badge tone={LABEL_TONE.HALUSINASI_SEBAGIAN}>SEBAGIAN</Badge><span>{run.summary.pct_halusinasi_sebagian ?? '—'}%</span></div>
                    <div className="flex justify-between"><Badge tone={LABEL_TONE.HALUSINASI_PENUH}>PENUH</Badge><span>{run.summary.pct_halusinasi_penuh ?? '—'}%</span></div>
                  </div>
                </div>
                <div className="border border-border rounded-md px-3 py-2">
                  <div className="text-xs text-text-secondary mb-1">Mean Faithfulness</div>
                  <div className="text-xl font-semibold text-text-primary">
                    {run.summary.mean_faithfulness_score != null ? run.summary.mean_faithfulness_score.toFixed(4) : '—'}
                  </div>
                  <div className="text-xs text-text-secondary mt-2 mb-1">Mean Answer Relevance</div>
                  <div className="text-xl font-semibold text-text-primary">
                    {run.summary.mean_answer_relevance_score != null ? run.summary.mean_answer_relevance_score.toFixed(4) : '—'}
                  </div>
                </div>
                {run.summary.kappa_value !== undefined && (
                  <div className="border border-border rounded-md px-3 py-2">
                    <div className="text-xs text-text-secondary mb-1">Cohen's Kappa</div>
                    <div className="text-xl font-semibold text-text-primary mb-1">{run.summary.kappa_value.toFixed(4)}</div>
                    <Badge tone={kappaTone(run.summary.kappa_value)}>{run.summary.kappa_interpretation}</Badge>
                    {run.summary.judge_models_identical && (
                      <div className="text-xs text-warning mt-2">⚠ Judges are identical — bias not detectable.</div>
                    )}
                  </div>
                )}
                {run.summary.duration_sec !== undefined && (
                  <div className="text-xs text-text-secondary col-span-3">Total run time: {fmtDuration(run.summary.duration_sec)}</div>
                )}
              </div>
              {isTerminal && run.status === 'completed' && (
                <Link
                  to={historyLinkFor(condition)}
                  className="inline-block mt-3 text-sm text-primary hover:underline"
                >
                  Lihat di History →
                </Link>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
