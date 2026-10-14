import { RefreshCw } from 'lucide-react'
import { useProcessLive, useRefreshProcessLive } from '@/api/hooks'
import type { ProcessParameter, ProcessStep, StepsResponse } from '@/types/process'
import { NotDocumented, ParamStatusBadge, QuoteBlock, SectionTitle, SourceRef, StepStatusBadge } from './shared'
import { formatTime, formatValue } from '@/lib/processFormat'

interface Props {
  data: StepsResponse
  params: ProcessParameter[] | undefined
  onOpenParameters: (step: string) => void
  onOpenLog: (logId: string) => void
  onOpenDecision: (id: string) => void
}

function StepParams({ step, params, onOpenDecision }: { step: ProcessStep; params: ProcessParameter[]; onOpenDecision: (id: string) => void }) {
  const rows = params.filter((p) => p.step === step.key)
  if (rows.length === 0) return null
  return (
    <div className="overflow-x-auto border border-border rounded-md">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-bg border-b border-border text-left text-text-secondary">
            <th className="font-medium px-3 py-2">Parameter / result</th>
            <th className="font-medium px-3 py-2">Value</th>
            <th className="font-medium px-3 py-2 min-w-[320px]">Why this value</th>
            <th className="font-medium px-3 py-2">Set by</th>
            <th className="font-medium px-3 py-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.key} className="border-b border-border last:border-b-0 align-top">
              <td className="px-3 py-2">
                <div className="text-text-primary">{p.name}</div>
                {p.defined_in && <SourceRef file={p.defined_in.file} line={p.defined_in.line} />}
              </td>
              <td className="px-3 py-2 whitespace-nowrap">
                {p.error ? (
                  <span className="text-danger text-xs">{p.error}</span>
                ) : (
                  <>
                    <span className="font-mono text-text-primary">{formatValue(p.value)}</span>
                    {p.unit && <div className="text-xs text-text-muted">{p.unit}</div>}
                  </>
                )}
              </td>
              <td className="px-3 py-2">
                {p.kind === 'result' ? (
                  <span className="text-xs text-text-muted">Observed result</span>
                ) : p.rationale ? (
                  <QuoteBlock quote={p.rationale} />
                ) : (
                  <NotDocumented />
                )}
                {p.note && <div className="text-xs text-warning mt-1">{p.note}</div>}
              </td>
              <td className="px-3 py-2 whitespace-nowrap">
                {p.set_by ? (
                  <button onClick={() => onOpenDecision(p.set_by!)} className="text-xs text-primary hover:underline">
                    {p.set_by}
                  </button>
                ) : (
                  <span className="text-xs text-text-muted">—</span>
                )}
              </td>
              <td className="px-3 py-2">
                <ParamStatusBadge status={p.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TrustHistory({ step }: { step: ProcessStep }) {
  if (!step.trust_history?.length) return null
  return (
    <div>
      <SectionTitle>trustScoreVersion history</SectionTitle>
      <div className="overflow-x-auto border border-border rounded-md">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border text-left text-text-secondary">
              <th className="font-medium px-3 py-2">Version</th>
              <th className="font-medium px-3 py-2">Event</th>
              <th className="font-medium px-3 py-2">Applied at</th>
              <th className="font-medium px-3 py-2">Backup</th>
              <th className="font-medium px-3 py-2 min-w-[280px]">Verify (from log)</th>
            </tr>
          </thead>
          <tbody>
            {step.trust_history.map((h) => (
              <tr key={h.log_id} className="border-b border-border last:border-b-0 align-top">
                <td className="px-3 py-2 font-mono whitespace-nowrap">{h.version}</td>
                <td className="px-3 py-2">
                  <div>{h.event}</div>
                  {h.summary && <div className="text-xs text-text-muted">{h.summary}</div>}
                  <SourceRef file={h.file} />
                </td>
                <td className="px-3 py-2 whitespace-nowrap font-mono text-xs">{h.applied_at ?? '—'}</td>
                <td className="px-3 py-2 font-mono text-xs break-all">{h.backup ?? '—'}</td>
                <td className="px-3 py-2 text-xs">{h.verify ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Stage1Table({ step, onOpenLog }: { step: ProcessStep; onOpenLog: (id: string) => void }) {
  const s1 = step.stage1
  if (!s1) return <div className="text-sm text-text-muted">No stage-1 report found.</div>
  return (
    <div>
      <SectionTitle>Stage 1 result of record (dev set)</SectionTitle>
      <div className="overflow-x-auto border border-border rounded-md">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border text-left text-text-secondary">
              <th className="font-medium px-3 py-2">alpha</th>
              <th className="font-medium px-3 py-2">% questions with ≥1 RELEVANT (v2 → v3)</th>
              <th className="font-medium px-3 py-2">Mean relevance score (v2 → v3)</th>
              <th className="font-medium px-3 py-2">% accepted in top-k (v2 → v3)</th>
            </tr>
          </thead>
          <tbody>
            {s1.table.map((r) => (
              <tr
                key={r.alpha}
                className={`border-b border-border last:border-b-0 ${s1.top_two?.includes(r.alpha) ? 'bg-primary-soft' : ''}`}
              >
                <td className="px-3 py-2 font-mono">{r.alpha}</td>
                <td className="px-3 py-2 font-mono">
                  {formatValue(r.pct_questions_with_relevant.v2)} → {formatValue(r.pct_questions_with_relevant.v3)}
                </td>
                <td className="px-3 py-2 font-mono">
                  {formatValue(r.mean_relevance_score.v2)} → {formatValue(r.mean_relevance_score.v3)}
                </td>
                <td className="px-3 py-2 font-mono">
                  {formatValue(r.pct_accepted_selected.v2)} → {formatValue(r.pct_accepted_selected.v3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-sm mt-2 space-y-1">
        <div>
          Ranking: <span className="font-mono">{formatValue(s1.ranked)}</span> · top two:{' '}
          <span className="font-mono">{formatValue(s1.top_two)}</span> · keep current:{' '}
          <span className="font-mono">{formatValue(s1.keep_current)}</span>
        </div>
        {s1.reasoning && <div className="text-text-secondary">{s1.reasoning}</div>}
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
          <SourceRef file={s1.file} />
          <span>· {formatTime(s1.timestamp_utc)}</span>
          {s1.git_commit && <span className="font-mono">· commit {s1.git_commit}</span>}
          {s1.judge_model && <span>· judge {s1.judge_model}</span>}
          {s1.log_id && (
            <button onClick={() => onOpenLog(s1.log_id!)} className="text-primary hover:underline">
              Open report
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function StepExtras({ step, onOpenLog }: { step: ProcessStep; onOpenLog: (id: string) => void }) {
  return (
    <>
      {step.key === 'trust' && <TrustHistory step={step} />}
      {step.key === 'alpha_selection' && (
        <>
          <Stage1Table step={step} onOpenLog={onOpenLog} />
          {step.stage2 && (
            <div className="text-sm">
              <span className="font-medium">Stage 2:</span> {step.stage2.status}
            </div>
          )}
        </>
      )}
      {step.key === 'judge' && step.calibration && (
        <div className="text-sm space-y-0.5">
          <div>
            <span className="font-medium">Calibration (Amendment 4):</span> {step.calibration.status}
          </div>
          {step.calibration.annotation_files_created && (
            <div className="text-text-secondary">
              Annotation files created {formatTime(step.calibration.annotation_files_created.created_at_utc)} ·{' '}
              {step.calibration.annotation_files_created.n_items} items (annotator 2:{' '}
              {step.calibration.annotation_files_created.n_items_annotator2})
            </div>
          )}
          <div className="text-text-secondary">
            Judge frozen: {step.calibration.judge_frozen ? 'yes' : 'no'} · adjudicated reference:{' '}
            {step.calibration.adjudicated_reference_exists ? 'exists' : 'not yet'}
          </div>
          <div className="text-xs text-text-muted">
            Aggregate status only. Calibration items, labels and the annotation key are not shown on this page.
          </div>
        </div>
      )}
    </>
  )
}

function StepCard({ step, params, onOpenParameters, onOpenLog, onOpenDecision }: Omit<Props, 'data' | 'params'> & {
  step: ProcessStep
  params: ProcessParameter[]
}) {
  return (
    <section id={`step-${step.key}`} className="border border-border rounded-lg bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
        <div className="flex items-start gap-3">
          <div className="h-7 w-7 shrink-0 rounded-full border border-border-strong text-sm font-semibold text-text-secondary flex items-center justify-center">
            {step.number}
          </div>
          <div>
            <h2 className="text-base font-semibold text-text-primary">{step.title}</h2>
            <p className="text-sm text-text-secondary mt-0.5 max-w-3xl">{step.purpose}</p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <StepStatusBadge status={step.status} />
          <span className="text-xs text-text-muted">Last evidence: {formatTime(step.last_evidence_utc)}</span>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3 text-sm mb-4 mt-3">
        <div>
          <SectionTitle>Implemented in</SectionTitle>
          <ul className="space-y-0.5">
            {step.modules.map((m) => (
              <li key={m.file} className={`font-mono text-xs break-all ${m.exists ? 'text-text-primary' : 'text-danger'}`}>
                {m.file}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <SectionTitle>Inputs</SectionTitle>
          <p className="text-text-secondary">{step.inputs}</p>
        </div>
        <div>
          <SectionTitle>Outputs</SectionTitle>
          <p className="text-text-secondary">{step.outputs}</p>
        </div>
      </div>

      <div className="space-y-4">
        {step.quotes.some(Boolean) && (
          <div>
            <SectionTitle>From the documentation</SectionTitle>
            <div className="space-y-2">
              {step.quotes.map((q, i) => (q ? <QuoteBlock key={i} quote={q} /> : null))}
            </div>
          </div>
        )}
        <StepParams step={step} params={params} onOpenDecision={onOpenDecision} />
        <StepExtras step={step} onOpenLog={onOpenLog} />
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-4 pt-3 border-t border-border text-xs">
        <button onClick={() => onOpenParameters(step.key)} className="text-primary hover:underline">
          View in parameter table
        </button>
        {step.logs.length > 0 && (
          <span className="text-text-muted">
            Logs:{' '}
            {step.logs.slice(0, 4).map((id, i) => (
              <span key={id}>
                {i > 0 && ', '}
                <button onClick={() => onOpenLog(id)} className="text-primary hover:underline font-mono">
                  {id.split('/')[1]}
                </button>
              </span>
            ))}
            {step.logs.length > 4 && <span> and {step.logs.length - 4} more</span>}
          </span>
        )}
      </div>
    </section>
  )
}

function GraphStatePanel() {
  const { data } = useProcessLive()
  const refresh = useRefreshProcessLive()
  const verify = data?.default
  const liveState = data?.live
  return (
    <div className="border border-border rounded-lg bg-surface p-4 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <h2 className="text-sm font-semibold text-text-primary">Graph state</h2>
        <button
          onClick={() => refresh.mutate()}
          disabled={refresh.isPending}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs border border-border-strong rounded-md text-text-secondary hover:bg-bg disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${refresh.isPending ? 'animate-spin' : ''}`} />
          Refresh live values
        </button>
      </div>
      <div className="grid gap-4 md:grid-cols-2 text-sm">
        <div>
          <SectionTitle>From the latest verify log</SectionTitle>
          {verify ? (
            <>
              {Object.entries(verify.counts).map(([label, c]) => (
                <div key={label}>
                  {label}: {c.nodes.toLocaleString()} nodes ·{' '}
                  <span className="font-mono">{formatValue(c.versions)}</span>
                </div>
              ))}
              <div className="text-xs text-text-muted mt-1">
                <SourceRef file={verify.file} /> · {formatTime(verify.timestamp_utc)}
              </div>
            </>
          ) : (
            <div className="text-text-muted">No verify log found.</div>
          )}
        </div>
        <div>
          <SectionTitle>Live (Neo4j, read-only)</SectionTitle>
          {!liveState || liveState.available === null ? (
            <div className="text-text-muted">Not refreshed yet. Live queries run only when you press the button.</div>
          ) : liveState.available === false ? (
            <div className="text-warning">{liveState.message ?? 'Neo4j unavailable'}</div>
          ) : (
            liveState.values && (
              <>
                <div>
                  Answer trustScoreVersion:{' '}
                  <span className="font-mono">{formatValue(liveState.values.answer_trust_versions)}</span>
                </div>
                <div>
                  Question trustScoreVersion:{' '}
                  <span className="font-mono">{formatValue(liveState.values.question_trust_versions)}</span>
                </div>
                <div className="text-xs text-text-secondary mt-1">
                  Nodes: <span className="font-mono">{formatValue(liveState.values.nodes)}</span>
                </div>
                <div className="text-xs text-text-secondary">
                  Edges: <span className="font-mono">{formatValue(liveState.values.edges)}</span>
                </div>
              </>
            )
          )}
          {liveState?.as_of && <div className="text-xs text-text-muted mt-1">As of {formatTime(liveState.as_of)}</div>}
        </div>
      </div>
    </div>
  )
}

export function PipelineTab({ data, params, onOpenParameters, onOpenLog, onOpenDecision }: Props) {
  return (
    <div>
      <GraphStatePanel />
      <div className="space-y-4">
        {data.steps.map((step) => (
          <StepCard
            key={step.key}
            step={step}
            params={params ?? []}
            onOpenParameters={onOpenParameters}
            onOpenLog={onOpenLog}
            onOpenDecision={onOpenDecision}
          />
        ))}
      </div>
    </div>
  )
}
