import { useEffect, useState, type ReactNode } from 'react'
import { useSettings, useTestConnection, useUpdateSettings } from '@/api/hooks'

const inputClass =
  'px-2.5 py-1.5 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border w-full'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-text-secondary">{label}</span>
      {children}
    </label>
  )
}

function SecretField({
  label,
  placeholder,
  value,
  onChange,
}: {
  label: string
  placeholder: string
  value: string
  onChange: (v: string) => void
}) {
  const [show, setShow] = useState(false)
  return (
    <Field label={label}>
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

  return (
    <div className="max-w-2xl">
      <h1 className="text-lg font-semibold text-text-primary mb-6">Settings</h1>

      <section className="mb-8">
        <h2 className="text-sm font-semibold text-text-primary mb-1">LLM Provider</h2>
        <p className="text-xs text-text-muted mb-3">Used by Condition A/B/C runs triggered from this dashboard.</p>
        <div className="flex flex-col gap-4">
          <Field label="Provider">
            <select className={inputClass} value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">openai</option>
              <option value="anthropic">anthropic</option>
              <option value="ollama">ollama (dev/testing only)</option>
            </select>
          </Field>
          <Field label="Model">
            <input className={inputClass} value={model} onChange={(e) => setModel(e.target.value)} />
          </Field>
          <SecretField
            label="OpenAI API Key"
            placeholder={data.api_key.is_set ? `Currently set (${data.api_key.masked})` : 'Not set'}
            value={apiKey}
            onChange={setApiKey}
          />
          <SecretField
            label="Anthropic API Key"
            placeholder={data.anthropic_api_key.is_set ? `Currently set (${data.anthropic_api_key.masked})` : 'Not set'}
            value={anthropicApiKey}
            onChange={setAnthropicApiKey}
          />
          {provider === 'ollama' && (
            <Field label="Ollama Host">
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
          <Field label="Neo4j URI">
            <input className={inputClass} value={neo4jUri} onChange={(e) => setNeo4jUri(e.target.value)} />
          </Field>
          <Field label="Neo4j User">
            <input className={inputClass} value={neo4jUser} onChange={(e) => setNeo4jUser(e.target.value)} />
          </Field>
          <SecretField
            label="Neo4j Password"
            placeholder={data.neo4j_password.is_set ? `Currently set (${data.neo4j_password.masked})` : 'Not set'}
            value={neo4jPassword}
            onChange={setNeo4jPassword}
          />
          <Field label="Neo4j Database">
            <input className={inputClass} value={neo4jDatabase} onChange={(e) => setNeo4jDatabase(e.target.value)} />
          </Field>
          <Field label="Questions Parquet Path">
            <input className={inputClass} value={questionsParquet} onChange={(e) => setQuestionsParquet(e.target.value)} />
          </Field>
          <Field label="Answers Parquet Path">
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

      <div className="mt-8 text-xs text-text-muted border-t border-border pt-4">
        Keys are stored encrypted outside this repository and are never displayed again or written to logs.
      </div>
    </div>
  )
}
