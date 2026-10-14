import { useMemo, useState } from 'react'
import { Badge } from '@/components/Badge'
import type { ProcessRun, RunsResponse } from '@/types/process'
import { formatTime, formatValue } from '@/lib/processFormat'

const selectClass =
  'px-2.5 py-2 text-sm border border-border-strong rounded-md bg-surface text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border'

function SplitBadge({ split }: { split: string | null }) {
  if (split === 'test') return <Badge tone="warning">test (held-out)</Badge>
  if (split === 'dev') return <Badge tone="primary">dev</Badge>
  return <Badge tone="neutral">{split ?? '—'}</Badge>
}

export function RunsTab({ data }: { data: RunsResponse }) {
  const [condition, setCondition] = useState('all')
  const [split, setSplit] = useState('all')
  const [kind, setKind] = useState('all')

  const rows = useMemo(
    () =>
      data.runs.filter(
        (r: ProcessRun) =>
          (condition === 'all' || r.condition === condition) &&
          (split === 'all' || r.split === split) &&
          (kind === 'all' || r.run_kind === kind),
      ),
    [data.runs, condition, split, kind],
  )

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <select value={condition} onChange={(e) => setCondition(e.target.value)} className={selectClass} aria-label="Condition">
          <option value="all">All conditions</option>
          {['A', 'B', 'C', 'D'].map((c) => (
            <option key={c} value={c}>
              Condition {c}
            </option>
          ))}
        </select>
        <select value={split} onChange={(e) => setSplit(e.target.value)} className={selectClass} aria-label="Split">
          <option value="all">Dev and test</option>
          <option value="dev">Dev only</option>
          <option value="test">Test only</option>
        </select>
        <select value={kind} onChange={(e) => setKind(e.target.value)} className={selectClass} aria-label="Run kind">
          <option value="all">All run kinds</option>
          <option value="generation">Generation</option>
          <option value="retrieval-only">Retrieval only</option>
          <option value="judge">Judge</option>
        </select>
      </div>
      <div className="text-xs text-text-muted mb-2">
        {rows.length} of {data.runs.length} runs · metadata only (no questions, answers or judge outputs) · read from{' '}
        {data.dirs.map((d) => (
          <span key={d} className="font-mono">
            {d}{' '}
          </span>
        ))}
      </div>
      <div className="overflow-x-auto border border-border rounded-lg bg-surface">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-bg border-b border-border-strong text-left text-text-secondary">
              {['Run', 'Condition', 'Split', 'n', 'alpha', 'Grounding', 'C retrieval', 'trustScoreVersion', 'config_hash',
                'Prompt / judge', 'Started', 'Commit', 'Status'].map((h) => (
                <th key={h} className="font-medium px-3 py-2.5 whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={13} className="px-4 py-8 text-center text-text-muted">
                  No runs match these filters.
                </td>
              </tr>
            )}
            {rows.map((r) => (
              <tr
                key={r.file}
                className={`border-b border-border last:border-b-0 align-top ${r.split === 'test' ? 'bg-bg' : ''}`}
              >
                <td className="px-3 py-2 min-w-[260px]">
                  <div className="font-mono text-xs break-all text-text-primary">{r.file}</div>
                  <div className="text-xs text-text-muted">{r.run_kind}</div>
                </td>
                <td className="px-3 py-2">{r.condition}</td>
                <td className="px-3 py-2">
                  <SplitBadge split={r.split} />
                </td>
                <td className="px-3 py-2 font-mono">{formatValue(r.n_sample)}</td>
                <td className="px-3 py-2 font-mono">{r.alpha ?? '—'}</td>
                <td className="px-3 py-2">{r.grounding === null ? '—' : r.grounding ? 'on' : 'off'}</td>
                <td className="px-3 py-2 font-mono">{r.c_retrieval_version ?? '—'}</td>
                <td className="px-3 py-2 font-mono">{r.trust_score_version ?? 'not recorded'}</td>
                <td className="px-3 py-2 font-mono">{r.config_hash ?? '—'}</td>
                <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">
                  {r.prompt_version ?? '—'}
                  {r.judge_version && <div className="text-text-muted">{r.judge_version}</div>}
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-xs">{formatTime(r.started_at_utc)}</td>
                <td className="px-3 py-2 font-mono text-xs whitespace-nowrap">
                  {r.commit ?? '—'}
                  {r.git_dirty && <div className="text-warning">dirty tree</div>}
                </td>
                <td className="px-3 py-2 text-xs">{r.status ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
