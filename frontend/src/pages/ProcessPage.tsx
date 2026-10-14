import { useSearchParams } from 'react-router-dom'
import {
  useProcessDecisions,
  useProcessLogs,
  useProcessParameters,
  useProcessRuns,
  useProcessSteps,
} from '@/api/hooks'
import { DecisionsTab } from '@/components/process/DecisionsTab'
import { LogsTab } from '@/components/process/LogsTab'
import { ParametersTab } from '@/components/process/ParametersTab'
import { PipelineTab } from '@/components/process/PipelineTab'
import { RunsTab } from '@/components/process/RunsTab'
import { LoadState } from '@/components/process/shared'
import { formatTime, formatValue } from '@/lib/processFormat'

const TABS = [
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'parameters', label: 'Parameters' },
  { key: 'decisions', label: 'Decisions' },
  { key: 'runs', label: 'Runs' },
  { key: 'logs', label: 'Logs' },
] as const
type Tab = (typeof TABS)[number]['key']

function Headline({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border rounded-lg bg-surface px-4 py-3 min-w-0">
      <div className="text-xs text-text-secondary">{label}</div>
      <div className="text-sm font-medium text-text-primary mt-1 break-words">{value}</div>
    </div>
  )
}

export function ProcessPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const tabParam = searchParams.get('tab')
  const tab: Tab = TABS.some((t) => t.key === tabParam) ? (tabParam as Tab) : 'pipeline'
  const step = searchParams.get('step') ?? 'all'
  const focusDecision = searchParams.get('decision')
  const logId = searchParams.get('log')
  const tail = Number(searchParams.get('tail') ?? 200)

  const go = (next: Record<string, string>) => setSearchParams(next, { replace: false })

  const steps = useProcessSteps()
  const params = useProcessParameters()
  const decisions = useProcessDecisions()
  const runs = useProcessRuns()
  const logs = useProcessLogs()

  const headline = steps.data?.headline

  return (
    <div className="max-w-[1400px]">
      <h1 className="text-lg font-semibold text-text-primary mb-1">Process &amp; Parameters</h1>
      <p className="text-sm text-text-secondary mb-1 max-w-3xl">
        The experiment end to end: what each step does, where it is implemented, which parameters it uses and why,
        which decision set them, and the evidence. Values are read from the code, config, logs and reports that use
        them; rationale is quoted from the repository's documents. This page is read-only.
      </p>
      <p className="text-xs text-text-muted mb-4">
        As of {formatTime(steps.data?.as_of)}
        {steps.data?.sources_as_of['docs/DECISION_C_SCORING.md'] && (
          <> · decision record modified {formatTime(steps.data.sources_as_of['docs/DECISION_C_SCORING.md'])}</>
        )}
        {steps.data?.sources_as_of.latest_log && <> · latest log {formatTime(steps.data.sources_as_of.latest_log)}</>}
      </p>

      {headline && (
        <div className="grid gap-3 grid-cols-2 lg:grid-cols-5 mb-5">
          <Headline label="Stage 1 (alpha)" value={headline.stage1} />
          <Headline label="Stage 2" value={headline.stage2} />
          <Headline label="Judge calibration" value={headline.calibration} />
          <Headline label="Official run (n=384)" value={headline.official_run} />
          <Headline
            label="Answer trustScoreVersion (verify log)"
            value={headline.trust_score_version ? formatValue(headline.trust_score_version) : '—'}
          />
        </div>
      )}

      <div className="flex gap-1 border-b border-border mb-5 overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => go(t.key === 'pipeline' ? {} : { tab: t.key })}
            className={`px-3 py-2 text-sm border-b-2 -mb-px whitespace-nowrap ${
              tab === t.key ? 'border-primary text-primary font-medium' : 'border-transparent text-text-secondary'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'pipeline' && (
        <>
          <LoadState isLoading={steps.isLoading} error={steps.error} />
          {steps.data && (
            <PipelineTab
              data={steps.data}
              params={params.data?.parameters}
              onOpenParameters={(s) => go({ tab: 'parameters', step: s })}
              onOpenLog={(id) => go({ tab: 'logs', log: id })}
              onOpenDecision={(id) => go({ tab: 'decisions', decision: id })}
            />
          )}
        </>
      )}
      {tab === 'parameters' && (
        <>
          <LoadState isLoading={params.isLoading} error={params.error} />
          {params.data && (
            <ParametersTab
              data={params.data}
              step={step}
              onStepChange={(s) => go(s === 'all' ? { tab: 'parameters' } : { tab: 'parameters', step: s })}
              onOpenDecision={(id) => go({ tab: 'decisions', decision: id })}
            />
          )}
        </>
      )}
      {tab === 'decisions' && (
        <>
          <LoadState isLoading={decisions.isLoading} error={decisions.error} />
          {decisions.data && <DecisionsTab data={decisions.data} focus={focusDecision} />}
        </>
      )}
      {tab === 'runs' && (
        <>
          <LoadState isLoading={runs.isLoading} error={runs.error} />
          {runs.data && <RunsTab data={runs.data} />}
        </>
      )}
      {tab === 'logs' && (
        <>
          <LoadState isLoading={logs.isLoading} error={logs.error} />
          {logs.data && (
            <LogsTab
              data={logs.data}
              selected={logId}
              onSelect={(id) => go({ tab: 'logs', log: id, tail: String(tail) })}
              tail={tail}
              onTailChange={(n) => go({ tab: 'logs', ...(logId ? { log: logId } : {}), tail: String(n) })}
            />
          )}
        </>
      )}
    </div>
  )
}
