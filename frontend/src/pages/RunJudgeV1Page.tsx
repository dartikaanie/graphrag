import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  useCancelRun,
  useContextRelevanceV1Jobs,
  useContextRelevanceV1Plan,
  useCreateContextRelevanceV1Run,
  useCreateJudgeV1Run,
  useCreateJudgeV2Run,
  useHistoryBatches,
  useJudgeV1JudgedStatus,
  useJudgeV1Jobs,
  useJudgeV1Plan,
  useJudgeV1Registry,
  useJudgeV2JudgedStatus,
  useJudgeV2Jobs,
  useJudgeV2Plan,
  useRunStream,
  type ContextRelevanceV1Job,
  type JudgeV1Job,
  type JudgeV1RegistryEntry,
} from '@/api/hooks'
import { JudgeVersionComparisonSection } from '@/components/JudgeVersionComparisonSection'
import { Badge } from '@/components/Badge'
import { ProgressRunPanel } from '@/components/ProgressRunPanel'
import type { HistoryItem } from '@/types/history'

function fmtTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

/** Always Asia/Jakarta (UTC+07:00), regardless of the viewer's own
 * timezone -- e.g. "2026-10-05 10:24". */
function fmtJakarta(iso: string): string {
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Jakarta', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(new Date(iso))
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? ''
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}`
}

function fmtDurationShort(sec: number | undefined): string {
  if (sec === undefined || sec === null) return '—'
  if (sec < 60) return `${sec.toFixed(0)}s`
  return `${Math.floor(sec / 60)}m${Math.round(sec % 60)}s`
}

/** "≈ 6 min" / "< 1 min" -- for wall-clock ESTIMATES (never claims
 * precision a rough per-item-latency average doesn't have). */
function fmtEstTime(sec: number | undefined): string {
  if (sec === undefined || sec === null) return '—'
  if (sec === 0) return '0 min'
  const min = sec / 60
  return min < 1 ? '< 1 min' : `≈ ${Math.round(min)} min`
}

function fmtCost(usd: number | null | undefined, unknown: boolean | undefined): string {
  if (unknown) return 'unknown'
  if (usd == null) return '—'
  return `$${usd.toFixed(4)}`
}

const ROLE_TONE: Record<string, 'primary' | 'neutral' | 'warning' | 'success'> = {
  primary: 'primary', secondary: 'success', fallback: 'warning', exploratory: 'neutral',
}

function JobProgressCard({ jobRunId }: { jobRunId: string }) {
  const { run } = useRunStream(jobRunId)
  const cancelRun = useCancelRun()
  const status = run?.status ?? 'queued'
  const isTerminal = ['completed', 'failed', 'cancelled'].includes(status)

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-text-primary">Judge-v1 job</h2>
          <Badge tone={status === 'completed' ? 'success' : status === 'failed' ? 'danger' : 'neutral'}>{status}</Badge>
        </div>
      </div>
      <ProgressRunPanel
        current={run?.progress.current ?? 0}
        total={run?.progress.total ?? 0}
        results={[]}
        onStop={isTerminal ? undefined : () => cancelRun.mutate(jobRunId)}
        status={status}
        queuePosition={run?.queue_position}
        compact
      />
      {run?.error && <div className="mt-2 text-xs text-danger border border-danger/40 bg-white rounded-md p-2">{run.error}</div>}
      {isTerminal && run?.summary && (
        <pre className="mt-2 text-xs text-text-secondary bg-bg rounded-md p-2 overflow-x-auto">
          {JSON.stringify(run.summary.per_judge, null, 2)}
        </pre>
      )}
    </div>
  )
}

export function RunJudgeV1Page() {
  const { data: registryData } = useJudgeV1Registry()
  const judges = registryData?.judges ?? []

  // Judge prompt version selector (Step 5) -- v1 and v2 hooks are both
  // called unconditionally (rules of hooks), and this just picks which
  // result/mutation feeds the UI below. Default v2 (the fix), v1 kept
  // fully selectable and untouched for reproducibility.
  const [judgeVersion, setJudgeVersion] = useState<'v1' | 'v2'>('v2')

  const { data: jobsDataV1 } = useJudgeV1Jobs()
  const { data: jobsDataV2 } = useJudgeV2Jobs()
  const jobsData = judgeVersion === 'v1' ? jobsDataV1 : jobsDataV2
  const { data: ctxrelJobsData } = useContextRelevanceV1Jobs()

  const [showSuperseded, setShowSuperseded] = useState(false)
  const { data: batchesData } = useHistoryBatches(showSuperseded)
  const batches = useMemo(
    () => (batchesData?.batches ?? []).filter((b) => b.runs.some((r) => r.status === 'success')),
    [batchesData],
  )

  const [expandedBatchIds, setExpandedBatchIds] = useState<Set<string> | null>(null)
  // Newest batch expanded by default -- only applied once batches first load.
  const isExpanded = (batchId: string, index: number) =>
    expandedBatchIds ? expandedBatchIds.has(batchId) : index === 0
  const toggleBatchExpanded = (batchId: string, index: number) => {
    setExpandedBatchIds((prev) => {
      const base = prev ?? new Set(batches.map((b, i) => (i === 0 ? b.batch_id : null)).filter((x): x is string => !!x))
      const next = new Set(base)
      if (next.has(batchId)) next.delete(batchId)
      else next.add(batchId)
      return next
    })
    void index
  }

  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set())
  const [selectedJudges, setSelectedJudges] = useState<Set<string>>(new Set(['primary', 'secondary']))
  const [workers, setWorkers] = useState<Record<string, number>>({ primary: 8, secondary: 4, fallback: 4 })

  const allRunIds = useMemo(() => batches.flatMap((b) => b.runs.map((r) => r.history_id)), [batches])
  const { data: judgedStatusDataV1 } = useJudgeV1JudgedStatus(allRunIds, Array.from(selectedJudges))
  const { data: judgedStatusDataV2 } = useJudgeV2JudgedStatus(allRunIds, Array.from(selectedJudges))
  const judgedStatusData = judgeVersion === 'v1' ? judgedStatusDataV1 : judgedStatusDataV2
  const judgedStatusByRun = useMemo(() => {
    const m = new Map<string, Map<string, { judged: number; total: number }>>()
    for (const row of judgedStatusData?.rows ?? []) {
      if (row.refused_reason || !row.judge_id) continue
      if (!m.has(row.run_id)) m.set(row.run_id, new Map())
      m.get(row.run_id)!.set(row.judge_id, {
        judged: (row.items_total ?? 0) - (row.items_to_judge ?? 0), total: row.items_total ?? 0,
      })
    }
    return m
  }, [judgedStatusData])

  const planV1 = useJudgeV1Plan()
  const planV2 = useJudgeV2Plan()
  const plan = judgeVersion === 'v1' ? planV1 : planV2
  const createRunV1 = useCreateJudgeV1Run()
  const createRunV2 = useCreateJudgeV2Run()
  const createRun = judgeVersion === 'v1' ? createRunV1 : createRunV2
  const [launchedJobId, setLaunchedJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const toggleRun = (id: string) => {
    setSelectedRunIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectBatch = (batchId: string) => {
    const batch = batches.find((b) => b.batch_id === batchId)
    if (!batch) return
    setSelectedRunIds(new Set(batch.runs.filter((r) => r.status === 'success').map((r) => r.history_id)))
  }

  const toggleJudge = (judgeId: string) => {
    setSelectedJudges((prev) => {
      const next = new Set(prev)
      if (next.has(judgeId)) next.delete(judgeId)
      else next.add(judgeId)
      return next
    })
  }

  const handlePlan = () => {
    setError(null)
    plan.mutate({
      run_ids: Array.from(selectedRunIds),
      judge_ids: Array.from(selectedJudges),
      workers,
    })
  }

  const handleLaunch = async () => {
    setError(null)
    try {
      const result = await createRun.mutateAsync({
        run_ids: Array.from(selectedRunIds), judge_ids: Array.from(selectedJudges), workers,
      })
      setLaunchedJobId(result.run_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  if (launchedJobId) {
    return (
      <div className="max-w-2xl">
        <div className="text-sm text-text-secondary mb-2">
          <Link to="/history" className="hover:text-primary">View results in History →</Link>
        </div>
        <h1 className="text-lg font-semibold text-text-primary mb-2">Hallucination Judge ({judgeVersion}) — running</h1>
        <JobProgressCard jobRunId={launchedJobId} />
      </div>
    )
  }

  return (
    <div className="max-w-4xl">
      <h1 className="text-lg font-semibold text-text-primary mb-1">Hallucination Judge</h1>
      <p className="text-sm text-text-secondary mb-4">
        Reference-based hallucination judging (candidate vs. the accepted Stack Overflow answer) -- a different
        metric from the legacy "Faithfulness Judge (context-grounded, legacy)". Invalid and superseded generation
        runs are hidden from the list below.
      </p>

      <div className="flex items-center gap-2 mb-4 text-xs">
        <span className="text-text-secondary">Judge prompt version:</span>
        <div className="flex border border-border-strong rounded-md overflow-hidden">
          {(['v1', 'v2'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setJudgeVersion(v)}
              className={`px-3 py-1 ${judgeVersion === v ? 'bg-primary text-white' : 'text-text-primary hover:bg-bg'}`}
            >
              judge-{v}{v === 'v2' ? ' (default, fixed prompt)' : ' (frozen)'}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-text-primary">1. Select runs</h2>
          <label className="flex items-center gap-1.5 text-xs text-text-secondary">
            <input type="checkbox" checked={showSuperseded} onChange={(e) => setShowSuperseded(e.target.checked)} />
            Show superseded runs
          </label>
        </div>

        {batches.length === 0 && <div className="px-1 py-2 text-sm text-text-muted">No completed runs found.</div>}

        <div className="flex flex-col gap-3">
          {batches.map((batch, batchIndex) => {
            const successRuns = batch.runs.filter((r) => r.status === 'success')
            const sorted = [...successRuns].sort((a, b) => {
              const cmp = (a.run_started_at ?? '').localeCompare(b.run_started_at ?? '')
              return sortDir === 'asc' ? cmp : -cmp
            })
            const expanded = isExpanded(batch.batch_id, batchIndex)
            return (
              <div key={batch.batch_id} className="border border-border rounded-md">
                <div className="flex items-center justify-between px-3 py-2 bg-bg">
                  <button
                    onClick={() => toggleBatchExpanded(batch.batch_id, batchIndex)}
                    className="flex items-center gap-2 text-sm text-text-primary"
                  >
                    <span>{expanded ? '▾' : '▸'}</span>
                    <span className="font-medium">{batch.launched_at ? fmtJakarta(batch.launched_at) : 'Unknown time'}</span>
                    <span className="text-text-secondary">· {successRuns.length} run(s)</span>
                    {batch.inferred && <Badge tone="warning">inferred</Badge>}
                  </button>
                  <button
                    onClick={() => selectBatch(batch.batch_id)}
                    className={`px-2.5 py-1 text-xs border rounded-md ${
                      batchIndex === 0
                        ? 'border-primary bg-primary text-white hover:bg-primary-hover'
                        : 'border-border-strong text-text-primary hover:bg-surface'
                    }`}
                  >
                    Select all {successRuns.length} run(s) of this batch
                  </button>
                </div>

                {expanded && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-left text-text-secondary border-b border-border">
                          <th className="py-1 pl-3 pr-2"></th>
                          <th
                            className="py-1 pr-3 cursor-pointer whitespace-nowrap"
                            onClick={() => setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))}
                          >
                            Run started ({sortDir === 'desc' ? '↓' : '↑'})
                          </th>
                          <th className="py-1 pr-3">Duration</th>
                          <th className="py-1 pr-3">run_label</th>
                          <th className="py-1 pr-3">run_id</th>
                          <th className="py-1 pr-3">n_sample</th>
                          <th className="py-1 pr-3">prompt_version</th>
                          <th className="py-1 pr-3">N processed</th>
                          <th className="py-1 pr-3">config_hash</th>
                          {Array.from(selectedJudges).map((jid) => (
                            <th key={jid} className="py-1 pr-3 whitespace-nowrap">{jid}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sorted.map((r: HistoryItem) => (
                          <tr key={r.history_id} className="border-b border-border/50">
                            <td className="py-1 pl-3 pr-2">
                              <input type="checkbox" checked={selectedRunIds.has(r.history_id)} onChange={() => toggleRun(r.history_id)} />
                            </td>
                            <td className="py-1 pr-3 whitespace-nowrap">{r.run_started_at ? fmtJakarta(r.run_started_at) : '—'}</td>
                            <td className="py-1 pr-3">{fmtDurationShort(r.duration_sec)}</td>
                            <td className="py-1 pr-3 font-medium">{r.run_label ?? r.condition}</td>
                            <td className="py-1 pr-3 text-text-muted">{r.history_id}</td>
                            <td className="py-1 pr-3">{r.n_sample_target}</td>
                            <td className="py-1 pr-3">{r.prompt_version ?? '—'}</td>
                            <td className="py-1 pr-3">{r.n_processed}</td>
                            <td className="py-1 pr-3">{r.config_hash ? r.config_hash.slice(0, 6) : '—'}</td>
                            {Array.from(selectedJudges).map((jid) => {
                              const s = judgedStatusByRun.get(r.history_id)?.get(jid)
                              return (
                                <td key={jid} className="py-1 pr-3 whitespace-nowrap">
                                  {s ? `${jid}: ${s.judged}/${s.total}` : '—'}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <h2 className="text-sm font-semibold text-text-primary mb-2">2. Judge models</h2>
        <div className="flex flex-col gap-2">
          {judges.map((j) => (
            <label key={j.judge_id} className="flex items-center gap-3 text-sm">
              <input type="checkbox" checked={selectedJudges.has(j.judge_id)} onChange={() => toggleJudge(j.judge_id)} />
              <Badge tone={ROLE_TONE[j.role] ?? 'neutral'}>{j.role}</Badge>
              <span className="font-medium">{j.judge_id}</span>
              <span className="text-text-secondary text-xs">{j.model} ({j.precision.split('--')[0].trim()})</span>
              <span className="text-text-muted text-xs">${j.price_per_m_input}/${j.price_per_m_output} per 1M (as of {j.price_date})</span>
              {!j.api_key_configured && <span className="text-danger text-xs">API key not configured</span>}
              <label className="ml-auto flex items-center gap-1 text-xs text-text-secondary">
                workers
                <input
                  type="number" min={1} className="w-14 px-1.5 py-0.5 border border-border-strong rounded-md"
                  value={workers[j.judge_id] ?? 4}
                  onChange={(e) => setWorkers((w) => ({ ...w, [j.judge_id]: Number(e.target.value) }))}
                />
              </label>
            </label>
          ))}
        </div>
      </div>

      <button
        onClick={handlePlan}
        disabled={plan.isPending || selectedRunIds.size === 0 || selectedJudges.size === 0}
        className="px-4 py-2 text-sm border border-border-strong rounded-md text-text-primary hover:bg-bg mb-4 disabled:opacity-50"
      >
        {plan.isPending ? 'Planning…' : 'Build dry-run preview'}
      </button>

      {plan.data && (
        <div className="border border-border rounded-lg bg-surface p-4 mb-4 overflow-x-auto">
          <h2 className="text-sm font-semibold text-text-primary mb-2">3. Confirm — nothing runs until you launch</h2>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-secondary border-b border-border">
                <th className="py-1 pr-3">run_label</th>
                <th className="py-1 pr-3">judge</th>
                <th className="py-1 pr-3">items</th>
                <th className="py-1 pr-3">already judged</th>
                <th className="py-1 pr-3">to judge</th>
                <th className="py-1 pr-3">est. tokens (in / out)</th>
                <th className="py-1 pr-3">est. cost (USD)</th>
                <th className="py-1 pr-3">est. time</th>
                <th className="py-1">source</th>
              </tr>
            </thead>
            <tbody>
              {plan.data.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/50">
                  {row.refused_reason ? (
                    <td colSpan={9} className="py-1 text-danger">{row.run_id}: {row.refused_reason}</td>
                  ) : (
                    <>
                      <td className="py-1 pr-3 font-medium">{row.run_label}</td>
                      <td className="py-1 pr-3">{row.judge_id}</td>
                      <td className="py-1 pr-3">{row.items_total}</td>
                      <td className="py-1 pr-3">{row.already_judged}</td>
                      <td className="py-1 pr-3">{row.items_to_judge}</td>
                      <td className="py-1 pr-3">
                        {row.est_tokens_in != null && row.est_tokens_out != null
                          ? `${row.est_tokens_in} / ${row.est_tokens_out}` : '—'}
                      </td>
                      <td className="py-1 pr-3">{fmtCost(row.est_cost_usd, row.cost_unknown)}</td>
                      <td className="py-1 pr-3">{fmtEstTime(row.est_time_sec)}</td>
                      <td className="py-1 text-text-muted">{row.token_source ?? '—'}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-3 text-xs text-text-secondary border-t border-border pt-2">
            <div className="font-medium text-text-primary mb-1">Totals</div>
            {Object.entries(plan.data.totals.per_judge).map(([judgeId, t]) => (
              <div key={judgeId}>
                {judgeId}: {t.items_to_judge} item(s)
                {t.est_tokens_in != null && t.est_tokens_out != null && ` · ${t.est_tokens_in} / ${t.est_tokens_out} tokens (in/out)`}
                {' · '}{fmtCost(t.est_cost_usd, t.cost_unknown_rows > 0 && t.est_cost_usd == null)}
                {t.cost_unknown_rows > 0 && ` (${t.cost_unknown_rows} row(s) excluded, cost unknown)`}
                {' · '}{fmtEstTime(t.est_time_sec)} ({judgeId})
              </div>
            ))}
            <div className="mt-1">
              Overall: {plan.data.totals.items_to_judge} item(s)
              {plan.data.totals.est_tokens_in != null && plan.data.totals.est_tokens_out != null &&
                ` · ${plan.data.totals.est_tokens_in} / ${plan.data.totals.est_tokens_out} tokens (in/out)`}
              {' · '}{fmtCost(plan.data.totals.est_cost_usd, plan.data.totals.cost_unknown_rows > 0 && plan.data.totals.est_cost_usd == null)}
              {plan.data.totals.cost_unknown_rows > 0 && ` (${plan.data.totals.cost_unknown_rows} row(s) excluded, cost unknown)`}
              {' · '}{fmtEstTime(plan.data.totals.est_time_sec)} wall-clock (judges run in parallel, slowest judge dominates)
            </div>
          </div>

          {error && <div className="mt-2 text-sm text-danger">{error}</div>}

          <button
            onClick={handleLaunch}
            disabled={createRun.isPending}
            className="mt-3 px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {createRun.isPending ? 'Launching…' : 'Confirm & Launch'}
          </button>
        </div>
      )}

      <ContextRelevanceLaunchSection batches={batches} judges={judges} />

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <h2 className="text-sm font-semibold text-text-primary mb-2">Judge results (hallucination, {judgeVersion})</h2>
        <JudgeResultsByBatch jobs={jobsData?.jobs ?? []} />
      </div>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <h2 className="text-sm font-semibold text-text-primary mb-2">Context relevance jobs</h2>
        <CtxrelResultsByBatch jobs={ctxrelJobsData?.jobs ?? []} />
      </div>

      <JudgeVersionComparisonSection selectedRunIds={Array.from(selectedRunIds)} />
    </div>
  )
}

/** Separate launch flow for context relevance (Step 3) -- its own plan/
 * confirm/progress flow and cost estimate, condition-A runs excluded
 * from the picker (Condition A has no retrieved_context at all). Reuses
 * the parent's already-fetched `batches`/`judges` rather than
 * re-fetching, but keeps its own run/judge selection state. */
function ContextRelevanceLaunchSection({
  batches, judges,
}: {
  batches: Array<{ batch_id: string; launched_at: string | null; runs: HistoryItem[] }>
  judges: JudgeV1RegistryEntry[]
}) {
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set())
  const [selectedJudges, setSelectedJudges] = useState<Set<string>>(new Set(['primary']))
  const [workers, setWorkers] = useState<Record<string, number>>({ primary: 8, secondary: 4, fallback: 4 })
  const [error, setError] = useState<string | null>(null)
  const [launchedJobId, setLaunchedJobId] = useState<string | null>(null)

  const plan = useContextRelevanceV1Plan()
  const createRun = useCreateContextRelevanceV1Run()

  const eligibleBatches = useMemo(
    () => batches.map((b) => ({ ...b, runs: b.runs.filter((r) => r.condition !== 'A' && r.status === 'success') }))
      .filter((b) => b.runs.length > 0),
    [batches],
  )

  const toggleRun = (id: string) => {
    setSelectedRunIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const selectBatch = (batchId: string) => {
    const batch = eligibleBatches.find((b) => b.batch_id === batchId)
    if (!batch) return
    setSelectedRunIds(new Set(batch.runs.map((r) => r.history_id)))
  }

  const toggleJudge = (judgeId: string) => {
    setSelectedJudges((prev) => {
      const next = new Set(prev)
      if (next.has(judgeId)) next.delete(judgeId)
      else next.add(judgeId)
      return next
    })
  }

  const handlePlan = () => {
    setError(null)
    plan.mutate({ run_ids: Array.from(selectedRunIds), judge_ids: Array.from(selectedJudges), workers })
  }

  const handleLaunch = async () => {
    setError(null)
    try {
      const result = await createRun.mutateAsync({
        run_ids: Array.from(selectedRunIds), judge_ids: Array.from(selectedJudges), workers,
      })
      setLaunchedJobId(result.run_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  if (launchedJobId) {
    return (
      <div className="border border-border rounded-lg bg-surface p-4 mb-4">
        <h2 className="text-sm font-semibold text-text-primary mb-2">Context relevance — running</h2>
        <JobProgressCard jobRunId={launchedJobId} />
      </div>
    )
  }

  return (
    <div className="border border-border rounded-lg bg-surface p-4 mb-4">
      <h2 className="text-sm font-semibold text-text-primary mb-1">Context relevance</h2>
      <p className="text-xs text-text-secondary mb-3">
        Judges whether retrieved context items directly address each question (reference-free, retrieval-content
        only). Condition A has no retrieved context and is excluded. Identical context shared by a plain/grounded
        pair of the same retrieval method is judged once and reused.
      </p>

      <div className="mb-3">
        <div className="text-xs font-medium text-text-secondary mb-1">Select runs</div>
        {eligibleBatches.length === 0 && <div className="text-sm text-text-muted">No eligible runs (B/C/D) found.</div>}
        <div className="flex flex-col gap-1.5">
          {eligibleBatches.map((batch) => (
            <div key={batch.batch_id} className="flex items-center gap-2 text-xs">
              <button
                onClick={() => selectBatch(batch.batch_id)}
                className="px-2 py-0.5 border border-border-strong rounded-md text-text-primary hover:bg-bg"
              >
                Select batch
              </button>
              <span className="text-text-secondary">
                {batch.launched_at ? fmtJakarta(batch.launched_at) : 'Unknown time'} · {batch.runs.length} eligible run(s)
              </span>
              <div className="flex gap-1">
                {batch.runs.map((r) => (
                  <label key={r.history_id} className="flex items-center gap-0.5">
                    <input type="checkbox" checked={selectedRunIds.has(r.history_id)} onChange={() => toggleRun(r.history_id)} />
                    <span>{r.run_label ?? r.condition}</span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-3">
        <div className="text-xs font-medium text-text-secondary mb-1">Judge models</div>
        <div className="flex flex-col gap-1">
          {judges.map((j) => (
            <label key={j.judge_id} className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={selectedJudges.has(j.judge_id)} onChange={() => toggleJudge(j.judge_id)} />
              <Badge tone={ROLE_TONE[j.role] ?? 'neutral'}>{j.role}</Badge>
              <span className="font-medium">{j.judge_id}</span>
              <label className="ml-auto flex items-center gap-1 text-text-secondary">
                workers
                <input
                  type="number" min={1} className="w-12 px-1 py-0.5 border border-border-strong rounded-md"
                  value={workers[j.judge_id] ?? 4}
                  onChange={(e) => setWorkers((w) => ({ ...w, [j.judge_id]: Number(e.target.value) }))}
                />
              </label>
            </label>
          ))}
        </div>
      </div>

      <button
        onClick={handlePlan}
        disabled={plan.isPending || selectedRunIds.size === 0 || selectedJudges.size === 0}
        className="px-3 py-1.5 text-xs border border-border-strong rounded-md text-text-primary hover:bg-bg mb-3 disabled:opacity-50"
      >
        {plan.isPending ? 'Planning…' : 'Build dry-run preview'}
      </button>

      {plan.data && (
        <div className="overflow-x-auto mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-text-secondary border-b border-border">
                <th className="py-1 pr-3">run_label</th>
                <th className="py-1 pr-3">context items</th>
              </tr>
            </thead>
            <tbody>
              {plan.data.rows.map((row, i) => (
                <tr key={i} className="border-b border-border/50">
                  {row.refused_reason ? (
                    <td colSpan={2} className="py-1 text-danger">{row.run_id}: {row.refused_reason}</td>
                  ) : (
                    <>
                      <td className="py-1 pr-3 font-medium">{row.run_label}</td>
                      <td className="py-1 pr-3">{row.n_context_items}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-2 text-xs text-text-secondary">
            <div>Unique (question, context item) pairs after dedup: {plan.data.totals.unique_items}</div>
            {Object.entries(plan.data.totals.per_judge).map(([judgeId, t]) => (
              <div key={judgeId}>
                {judgeId}: {t.items_to_judge} to judge
                {t.est_tokens_in != null && t.est_tokens_out != null && ` · ${t.est_tokens_in} / ${t.est_tokens_out} tokens (in/out)`}
                {' · '}{fmtCost(t.est_cost_usd, t.cost_unknown)}
                {' · '}{fmtEstTime(t.est_time_sec)}
              </div>
            ))}
          </div>

          {error && <div className="mt-2 text-sm text-danger">{error}</div>}

          <button
            onClick={handleLaunch}
            disabled={createRun.isPending}
            className="mt-3 px-4 py-2 text-sm bg-primary text-white rounded-md hover:bg-primary-hover disabled:opacity-50"
          >
            {createRun.isPending ? 'Launching…' : 'Confirm & Launch'}
          </button>
        </div>
      )}
    </div>
  )
}

/** Completed judge-v1 jobs grouped by batch_id, newest batch first --
 * jobs with no resolvable batch_id (older runs launched before batch_id
 * was recorded, or an unresolvable run_file) fall into an "Unknown
 * batch" bucket at the end rather than being hidden. Each batch groups
 * its jobs' date/time, judge models, run labels covered, items judged/
 * failed, and warnings (missing output file / served-model mismatch),
 * and links to the full results view at /evaluation/judge-v1/results/:batchId. */
function JudgeResultsByBatch({ jobs }: { jobs: JudgeV1Job[] }) {
  const grouped = useMemo(() => {
    const byBatch = new Map<string, JudgeV1Job[]>()
    for (const job of jobs) {
      const key = job.batch_id ?? '__unknown__'
      if (!byBatch.has(key)) byBatch.set(key, [])
      byBatch.get(key)!.push(job)
    }
    const entries = Array.from(byBatch.entries())
    entries.sort(([keyA, jobsA], [keyB, jobsB]) => {
      if (keyA === '__unknown__') return 1
      if (keyB === '__unknown__') return -1
      const latest = (js: JudgeV1Job[]) => js.reduce((m, j) => (j.run_started_at > m ? j.run_started_at : m), '')
      return latest(jobsB).localeCompare(latest(jobsA))
    })
    return entries
  }, [jobs])

  if (grouped.length === 0) {
    return <div className="px-1 py-2 text-sm text-text-muted">No judge-v1 jobs yet.</div>
  }

  return (
    <div className="flex flex-col gap-3">
      {grouped.map(([batchId, batchJobs]) => {
        const allRunLabels = Array.from(new Set(batchJobs.flatMap((j) => j.run_labels.filter(Boolean)))) as string[]
        const latestStartedAt = batchJobs.reduce((m, j) => (j.run_started_at > m ? j.run_started_at : m), '')
        const isUnknown = batchId === '__unknown__'
        return (
          <div key={batchId} className="border border-border rounded-md">
            <div className="flex items-center justify-between px-3 py-2 bg-bg">
              <div className="text-sm text-text-primary">
                <span className="font-medium">{isUnknown ? 'Unknown batch' : fmtJakarta(latestStartedAt)}</span>
                <span className="text-text-secondary"> · {allRunLabels.join(', ') || '—'}</span>
              </div>
              {!isUnknown && (
                <Link
                  to={`/evaluation/judge-v1/results/${batchId}`}
                  className="px-2.5 py-1 text-xs border border-border-strong rounded-md text-text-primary hover:bg-surface"
                >
                  View results →
                </Link>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-text-secondary border-b border-border">
                    <th className="py-1 pl-3 pr-3">Date / time</th>
                    <th className="py-1 pr-3">Judge</th>
                    <th className="py-1 pr-3">Run labels</th>
                    <th className="py-1 pr-3">Judged</th>
                    <th className="py-1 pr-3">Failed</th>
                    <th className="py-1">Warnings</th>
                  </tr>
                </thead>
                <tbody>
                  {batchJobs.map((job) => (
                    <tr key={job.job_id} className="border-b border-border/50 last:border-b-0">
                      <td className="py-1 pl-3 pr-3 whitespace-nowrap">{fmtTimestamp(job.run_started_at)}</td>
                      <td className="py-1 pr-3">{job.judge}</td>
                      <td className="py-1 pr-3">{job.run_labels.filter(Boolean).join(', ') || '—'}</td>
                      <td className="py-1 pr-3">{job.judged ?? '—'}</td>
                      <td className="py-1 pr-3">{job.failed ?? '—'}</td>
                      <td className="py-1">
                        {job.output_file_missing && <span className="text-danger">output file missing</span>}
                        {job.is_other_config && <span className="text-warning ml-2">other config</span>}
                        {job.served_model_mismatch_count > 0 && (
                          <span className="text-warning ml-2">{job.served_model_mismatch_count} served-model mismatch(es)</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}

/** Same batch-grouping shape as JudgeResultsByBatch, for context-
 * relevance jobs -- a SEPARATE job log (ctxrel_v1_lookup_service.py),
 * previously not shown anywhere in the dashboard at all. Links to the
 * SAME batch results page (JudgeV1ResultsPage now renders context
 * relevance via /api/context-relevance/results/batch-summary
 * regardless of whether a hallucination job also exists for that batch). */
function CtxrelResultsByBatch({ jobs }: { jobs: ContextRelevanceV1Job[] }) {
  const grouped = useMemo(() => {
    const byBatch = new Map<string, ContextRelevanceV1Job[]>()
    for (const job of jobs) {
      const key = job.batch_id ?? '__unknown__'
      if (!byBatch.has(key)) byBatch.set(key, [])
      byBatch.get(key)!.push(job)
    }
    const entries = Array.from(byBatch.entries())
    entries.sort(([keyA, jobsA], [keyB, jobsB]) => {
      if (keyA === '__unknown__') return 1
      if (keyB === '__unknown__') return -1
      const latest = (js: ContextRelevanceV1Job[]) => js.reduce((m, j) => (j.run_started_at > m ? j.run_started_at : m), '')
      return latest(jobsB).localeCompare(latest(jobsA))
    })
    return entries
  }, [jobs])

  if (grouped.length === 0) {
    return <div className="px-1 py-2 text-sm text-text-muted">No context-relevance jobs yet.</div>
  }

  return (
    <div className="flex flex-col gap-3">
      {grouped.map(([batchId, batchJobs]) => {
        const allRunLabels = Array.from(new Set(batchJobs.flatMap((j) => j.run_labels.filter(Boolean)))) as string[]
        const latestStartedAt = batchJobs.reduce((m, j) => (j.run_started_at > m ? j.run_started_at : m), '')
        const isUnknown = batchId === '__unknown__'
        return (
          <div key={batchId} className="border border-border rounded-md">
            <div className="flex items-center justify-between px-3 py-2 bg-bg">
              <div className="text-sm text-text-primary">
                <span className="font-medium">{isUnknown ? 'Unknown batch' : fmtJakarta(latestStartedAt)}</span>
                <span className="text-text-secondary"> · {allRunLabels.join(', ') || '—'}</span>
              </div>
              {!isUnknown && (
                <Link
                  to={`/evaluation/judge-v1/results/${batchId}`}
                  className="px-2.5 py-1 text-xs border border-border-strong rounded-md text-text-primary hover:bg-surface"
                >
                  View results →
                </Link>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-text-secondary border-b border-border">
                    <th className="py-1 pl-3 pr-3">Date / time</th>
                    <th className="py-1 pr-3">Judge</th>
                    <th className="py-1 pr-3">Run labels</th>
                    <th className="py-1 pr-3">Judged</th>
                    <th className="py-1 pr-3">Failed</th>
                    <th className="py-1">Warnings</th>
                  </tr>
                </thead>
                <tbody>
                  {batchJobs.map((job) => (
                    <tr key={job.job_id} className="border-b border-border/50 last:border-b-0">
                      <td className="py-1 pl-3 pr-3 whitespace-nowrap">{fmtTimestamp(job.run_started_at)}</td>
                      <td className="py-1 pr-3">{job.judge}</td>
                      <td className="py-1 pr-3">{job.run_labels.filter(Boolean).join(', ') || '—'}</td>
                      <td className="py-1 pr-3">{job.judged ?? '—'}</td>
                      <td className="py-1 pr-3">{job.failed ?? '—'}</td>
                      <td className="py-1">
                        {job.output_file_missing && <span className="text-danger">output file missing</span>}
                        {job.is_other_config && <span className="text-warning ml-2">other config</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )
      })}
    </div>
  )
}
