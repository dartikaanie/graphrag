import { useMemo, useState } from 'react'
import { SearchInput } from '@/components/SearchInput'
import type { ParametersResponse, ProcessParameter } from '@/types/process'
import { NotDocumented, ParamStatusBadge, QuoteBlock, SourceRef } from './shared'
import { formatTime, formatValue, STEP_LABELS } from '@/lib/processFormat'

interface Props {
  data: ParametersResponse
  step: string
  onStepChange: (step: string) => void
  onOpenDecision: (id: string) => void
}

const STATUS_OPTIONS = ['all', 'frozen', 'pending', 'candidate', 'exploratory', 'observed']

function matches(p: ProcessParameter, q: string): boolean {
  if (!q) return true
  const hay = [p.name, p.key, formatValue(p.value), p.defined_in?.file ?? '', p.rationale?.text ?? '', p.spec?.text ?? '']
    .join(' ')
    .toLowerCase()
  return hay.includes(q.toLowerCase())
}

const selectClass =
  'px-2.5 py-2 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border'

export function ParametersTab({ data, step, onStepChange, onOpenDecision }: Props) {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('all')
  const [kind, setKind] = useState<'all' | 'parameter' | 'result'>('parameter')
  const [onlyUndocumented, setOnlyUndocumented] = useState(false)

  const rows = useMemo(
    () =>
      data.parameters.filter(
        (p) =>
          (step === 'all' || p.step === step) &&
          (status === 'all' || p.status === status) &&
          (kind === 'all' || p.kind === kind) &&
          (!onlyUndocumented || (p.kind === 'parameter' && !p.rationale_documented)) &&
          matches(p, query),
      ),
    [data.parameters, step, status, kind, onlyUndocumented, query],
  )

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <SearchInput value={query} onChange={setQuery} placeholder="Search name, value, file, rationale…" />
        <select value={step} onChange={(e) => onStepChange(e.target.value)} className={selectClass} aria-label="Step">
          <option value="all">All steps</option>
          {Object.entries(STEP_LABELS).map(([k, label]) => (
            <option key={k} value={k}>
              {label}
            </option>
          ))}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} className={selectClass} aria-label="Status">
          {STATUS_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s === 'all' ? 'All statuses' : s}
            </option>
          ))}
        </select>
        <select
          value={kind}
          onChange={(e) => setKind(e.target.value as typeof kind)}
          className={selectClass}
          aria-label="Kind"
        >
          <option value="parameter">Parameters</option>
          <option value="result">Observed results</option>
          <option value="all">Parameters and results</option>
        </select>
        <label className="flex items-center gap-1.5 text-sm text-text-secondary">
          <input type="checkbox" checked={onlyUndocumented} onChange={(e) => setOnlyUndocumented(e.target.checked)} />
          Only "Rationale not documented" ({data.undocumented.length})
        </label>
      </div>

      <div className="text-xs text-text-muted mb-2">
        {rows.length} of {data.parameters.length} rows · values read from code, config, logs and reports at{' '}
        {formatTime(data.as_of)}
      </div>

      <div className="overflow-x-auto border border-border rounded-lg bg-surface">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border-strong text-left text-text-secondary">
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Name</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Value</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Step</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Defined in</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Used by</th>
              <th className="font-medium px-3 py-2.5 min-w-[340px]">Rationale</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Set by</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Status</th>
              <th className="font-medium px-3 py-2.5 whitespace-nowrap">Last changed</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={9} className="px-4 py-8 text-center text-text-muted">
                  No rows match these filters.
                </td>
              </tr>
            )}
            {rows.map((p) => (
              <tr key={p.key} className="border-b border-border last:border-b-0 align-top">
                <td className="px-3 py-2.5">
                  <div className="text-text-primary">{p.name}</div>
                  <div className="font-mono text-xs text-text-muted">{p.key}</div>
                </td>
                <td className="px-3 py-2.5 min-w-[140px]">
                  {p.error ? (
                    <span className="text-xs text-danger">{p.error}</span>
                  ) : (
                    <>
                      <div className="font-mono text-text-primary break-words">{formatValue(p.value)}</div>
                      {p.unit && <div className="text-xs text-text-muted">{p.unit}</div>}
                      {p.source_note && <div className="text-xs text-text-muted">{p.source_note}</div>}
                    </>
                  )}
                </td>
                <td className="px-3 py-2.5 whitespace-nowrap text-text-secondary">{STEP_LABELS[p.step] ?? p.step}</td>
                <td className="px-3 py-2.5 min-w-[180px]">
                  {p.defined_in && <SourceRef file={p.defined_in.file} line={p.defined_in.line} />}
                </td>
                <td className="px-3 py-2.5 min-w-[180px]">
                  <div className="flex flex-col gap-0.5">
                    {p.used_by.map((f) => (
                      <span key={f} className="font-mono text-xs text-text-secondary break-all">
                        {f.split('/').pop()}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-3 py-2.5 space-y-2">
                  {p.kind === 'result' ? (
                    <span className="text-xs text-text-muted">Observed result</span>
                  ) : p.rationale ? (
                    <QuoteBlock quote={p.rationale} />
                  ) : (
                    <NotDocumented />
                  )}
                  {p.spec && <QuoteBlock quote={p.spec} label="Specified in" />}
                  {p.note && <div className="text-xs text-warning">{p.note}</div>}
                </td>
                <td className="px-3 py-2.5 whitespace-nowrap">
                  {p.set_by ? (
                    <button onClick={() => onOpenDecision(p.set_by!)} className="text-xs text-primary hover:underline">
                      {p.set_by}
                    </button>
                  ) : (
                    <span className="text-xs text-text-muted">—</span>
                  )}
                </td>
                <td className="px-3 py-2.5">
                  <ParamStatusBadge status={p.status} />
                </td>
                <td className="px-3 py-2.5 whitespace-nowrap text-xs">
                  {p.last_changed ? (
                    p.last_changed.uncommitted ? (
                      <span className="text-warning">uncommitted</span>
                    ) : (
                      <>
                        <span className="font-mono">{p.last_changed.commit}</span>
                        <div className="text-text-muted">{p.last_changed.date}</div>
                      </>
                    )
                  ) : (
                    <span className="text-text-muted">not in git</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
