import { useEffect, useState, type ReactNode } from 'react'
import { useSettings, useTestConnection, useUpdateSettings } from '@/api/hooks'
import { InfoTooltip } from '@/components/InfoTooltip'
import { PARAM_GLOSSARY } from '@/lib/paramGlossary'

const inputClass =
  'px-2.5 py-1.5 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border w-full'

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

function SecretField({
  label,
  glossaryKey,
  placeholder,
  value,
  onChange,
}: {
  label: string
  glossaryKey?: keyof typeof PARAM_GLOSSARY
  placeholder: string
  value: string
  onChange: (v: string) => void
}) {
  const [show, setShow] = useState(false)
  return (
    <Field label={label} glossaryKey={glossaryKey}>
      <div className="flex gap-2">
        <input
          type={show ? 'text' : 'password'}
          className={inputClass}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          onClick={() => setShow((s) => !s)}
          className="px-2 text-xs border border-border-strong rounded-md text-text-secondary hover:bg-bg shrink-0"
        >
          {show ? 'Hide' : 'Show'}
        </button>
      </div>
    </Field>
  )
}

function TestConnectionButton({
  onTest,
}: {
  onTest: () => Promise<{ ok: boolean; message: string }>
}) {
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [pending, setPending] = useState(false)

  const run = async () => {
    setPending(true)
    setResult(null)
    try {
      setResult(await onTest())
    } catch (e) {
      setResult({ ok: false, message: (e as Error).message })
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="flex items-center gap-3 mt-2">
      <button
        onClick={run}
        disabled={pending}
        className="px-3 py-1.5 text-xs border border-border-strong rounded-md text-text-primary hover:bg-bg disabled:opacity-50"
      >
        {pending ? 'Testing…' : 'Test Connection'}
      </button>
      {result && (
        <span className={`text-xs ${result.ok ? 'text-success' : 'text-danger'}`}>
          {result.ok ? '✓ Connected' : `✗ Failed: ${result.message}`}
        </span>
      )}
    </div>
  )
}

export function SettingsPage() {
  const { data, isLoading } = useSettings()
  const updateSettings = useUpdateSettings()
  const testConnection = useTestConnection()

  const [provider, setProvider] = useState('openai')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [anthropicApiKey, setAnthropicApiKey] = useState('')
  const [ollamaHost, setOllamaHost] = useState('')
  const [neo4jUri, setNeo4jUri] = useState('')
  const [neo4jUser, setNeo4jUser] = useState('')
  const [neo4jPassword, setNeo4jPassword] = useState('')
  const [neo4jDatabase, setNeo4jDatabase] = useState('')
  const [questionsParquet, setQuestionsParquet] = useState('')
  const [answersParquet, setAnswersParquet] = useState('')

  const [judgeProvider, setJudgeProvider] = useState('openai')
  const [judgeModel, setJudgeModel] = useState('')
  const [judgeTemperature, setJudgeTemperature] = useState(0.1)
  const [kappaValidationEnabled, setKappaValidationEnabled] = useState(false)
  const [secondaryJudgeProvider, setSecondaryJudgeProvider] = useState('openai')
  const [secondaryJudgeModel, setSecondaryJudgeModel] = useState('')
  const [kappaSampleSize, setKappaSampleSize] = useState(50)
  const [judgeMajorityRounds, setJudgeMajorityRounds] = useState(1)

  useEffect(() => {
    if (!data) return
    setProvider(data.provider)
    setModel(data.model)
    setOllamaHost(data.ollama_host)
    setNeo4jUri(data.neo4j_uri)
    setNeo4jUser(data.neo4j_user)
    setNeo4jDatabase(data.neo4j_database)
    setQuestionsParquet(data.questions_parquet)
    setAnswersParquet(data.answers_parquet)
    setJudgeProvider(data.judge_provider)
    setJudgeModel(data.judge_model)
    setJudgeTemperature(data.judge_temperature)
    setSecondaryJudgeProvider(data.secondary_judge_provider)
    setSecondaryJudgeModel(data.secondary_judge_model)
    setKappaSampleSize(data.kappa_sample_size)
    setJudgeMajorityRounds(data.judge_majority_rounds)
  }, [data])

  if (isLoading || !data) return <div className="text-sm text-text-muted">Loading...</div>

  const saveProvider = () =>
    updateSettings.mutate({ provider, model, api_key: apiKey, anthropic_api_key: anthropicApiKey, ollama_host: ollamaHost })

  const saveDataSource = () =>
    updateSettings.mutate({
      neo4j_uri: neo4jUri,
      neo4j_user: neo4jUser,
      neo4j_password: neo4jPassword,
      neo4j_database: neo4jDatabase,
      questions_parquet: questionsParquet,
      answers_parquet: answersParquet,
    })

  const saveJudge = () =>
    updateSettings.mutate({
      judge_provider: judgeProvider,
      judge_model: judgeModel,
      judge_temperature: judgeTemperature,
      secondary_judge_provider: secondaryJudgeProvider,
      secondary_judge_model: secondaryJudgeModel,
      kappa_sample_size: kappaSampleSize,
      judge_majority_rounds: judgeMajorityRounds,
    })

  const judgeModelsIdentical =
    kappaValidationEnabled && judgeProvider === secondaryJudgeProvider && judgeModel === secondaryJudgeModel && judgeModel !== ''

  return (
    <div className="max-w-2xl">
      <h1 className="text-lg font-semibold text-text-primary mb-6">Settings</h1>

      <section className="mb-8">
        <h2 className="text-sm font-semibold text-text-primary mb-1">LLM Provider</h2>
        <p className="text-xs text-text-muted mb-3">Used by Condition A/B/C runs triggered from this dashboard.</p>
        <div className="flex flex-col gap-4">
          <Field label="Provider" glossaryKey="provider">
            <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">openai</option>
              <option value="anthropic">anthropic</option>
              <option value="ollama">ollama (dev/testing only)</option>
            </select>
          </Field>
          <Field label="Model" glossaryKey="model">
            <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
          </Field>
          <SecretField
            label="OpenAI API Key"
            glossaryKey="api_key"
            placeholder={data.api_key.is_set ? `Currently set (${data.api_key.masked})` : 'Not set'}
            value={apiKey}
            onChange={setApiKey}
          />
          <SecretField
            label="Anthropic API Key"
            glossaryKey="anthropic_api_key"
            placeholder={data.anthropic_api_key.is_set ? `Currently set (${data.anthropic_api_key.masked})` : 'Not set'}
            value={anthropicApiKey}
            onChange={setAnthropicApiKey}
          />
          {provider === 'ollama' && (
            <Field label="Ollama Host" glossaryKey="ollama_host">
              <input className={inputClass} value={ollamaHost} onChange={(e) => setOllamaHost(e.target.value)} />
            </Field>
          )}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={saveProvider}
            disabled={updateSettings.isPending}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            Save
          </button>
        </div>
        <TestConnectionButton
          onTest={() =>
            testConnection.mutateAsync({
              target: 'llm',
              provider,
              model,
              api_key: apiKey || undefined,
              ollama_host: ollamaHost,
            })
          }
        />
      </section>

      <div className="border-t border-border my-6" />

      <section>
        <h2 className="text-sm font-semibold text-text-primary mb-1">Data Source</h2>
        <p className="text-xs text-text-muted mb-3">Neo4j knowledge graph and SORD Parquet paths.</p>
        <div className="flex flex-col gap-4">
          <Field label="Neo4j URI" glossaryKey="neo4j_uri">
            <input className={inputClass} value={neo4jUri} onChange={(e) => setNeo4jUri(e.target.value)} />
          </Field>
          <Field label="Neo4j User" glossaryKey="neo4j_user">
            <input className={inputClass} value={neo4jUser} onChange={(e) => setNeo4jUser(e.target.value)} />
          </Field>
          <SecretField
            label="Neo4j Password"
            glossaryKey="neo4j_password"
            placeholder={data.neo4j_password.is_set ? `Currently set (${data.neo4j_password.masked})` : 'Not set'}
            value={neo4jPassword}
            onChange={setNeo4jPassword}
          />
          <Field label="Neo4j Database" glossaryKey="neo4j_database">
            <input className={inputClass} value={neo4jDatabase} onChange={(e) => setNeo4jDatabase(e.target.value)} />
          </Field>
          <Field label="Questions Parquet Path" glossaryKey="questions_parquet">
            <input className={inputClass} value={questionsParquet} onChange={(e) => setQuestionsParquet(e.target.value)} />
          </Field>
          <Field label="Answers Parquet Path" glossaryKey="answers_parquet">
            <input className={inputClass} value={answersParquet} onChange={(e) => setAnswersParquet(e.target.value)} />
          </Field>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={saveDataSource}
            disabled={updateSettings.isPending}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            Save
          </button>
        </div>
        <TestConnectionButton
          onTest={() =>
            testConnection.mutateAsync({
              target: 'neo4j',
              neo4j_uri: neo4jUri,
              neo4j_user: neo4jUser,
              neo4j_password: neo4jPassword || undefined,
            })
          }
        />
      </section>

      <div className="border-t border-border my-6" />

      <section>
        <h2 className="text-sm font-semibold text-text-primary mb-1">LLM-as-Judge Configuration</h2>
        <p className="text-xs text-text-muted mb-3">
          Used by the LLM-as-Judge evaluation (Faithfulness / Answer Relevance / Hallucination Rate). Reuses the API
          key configured above for whichever provider is selected here.
        </p>
        <div className="flex flex-col gap-4">
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
          <Field label="Judge Majority Rounds" glossaryKey="judge_majority_rounds">
            <input
              type="number" min="1" className={inputClass}
              value={judgeMajorityRounds} onChange={(e) => setJudgeMajorityRounds(Number(e.target.value))}
            />
          </Field>
          <p className="text-xs text-text-muted -mt-2">
            Number of independent rounds per question for majority-vote consensus — recommended {'>'}1 ONLY for a
            Cohen's Kappa validation subsample, not the full batch, since API cost scales linearly with it.
          </p>

          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={kappaValidationEnabled} onChange={(e) => setKappaValidationEnabled(e.target.checked)} />
            <span className="text-text-secondary inline-flex items-center gap-1">
              Enable Cohen's Kappa validation
              <InfoTooltip text={PARAM_GLOSSARY.kappa_validation} />
            </span>
          </label>

          {kappaValidationEnabled && (
            <>
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
              {judgeModelsIdentical && (
                <div className="text-xs text-warning bg-white border border-warning/40 rounded-md px-2 py-1.5">
                  ⚠ Primary and secondary judge are identical ({judgeProvider}/{judgeModel}) — Kappa validation
                  with the same model will not detect self-enhancement bias, only its own internal variance.
                </div>
              )}
            </>
          )}
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={saveJudge}
            disabled={updateSettings.isPending}
            className="px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </section>

      <div className="mt-8 text-xs text-text-muted border-t border-border pt-4">
        Keys are stored encrypted outside this repository and are never displayed again or written to logs.
      </div>
    </div>
  )
}
