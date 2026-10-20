import { useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useReport, useReportItem, useReportItems } from '@/api/hooks'
import { API_BASE } from '@/api/client'
import { Badge } from '@/components/Badge'
import { LoadState } from '@/components/process/shared'
import { downloadMarkdown } from '@/lib/markdown'
import type { Report, ReportRun, ThreeNumbers } from '@/types/report'

// End-to-end report of the official evaluation (Amendment 14 draft, plan step 9). Read-only: it shows the run
// registry and stored outputs through GET /api/report; nothing on this page starts a run or calls a model.

const JUDGES = ['deepseek', 'llama', 'qwen3-235b'] as const
const JUDGE_ROLE: Record<string, string> = { deepseek: 'sensitive', llama: 'supporting', 'qwen3-235b': 'supporting' }

const SECTIONS = [
  ['config', 'a. Configuration'],
  ['data', 'b. Data and sampling'],
  ['alpha', 'c. Alpha selection'],
  ['retrieval', 'd. Retrieval'],
  ['generation', 'e. Generation'],
  ['judging', 'f. Judging and budget'],
  ['results', 'g. Results'],
  ['comparisons', 'h. Comparisons'],
  ['reliability', 'i. Judge reliability'],
  ['limitations', 'j. Limitations'],
  ['items', 'k. Item drill-down'],
] as const

const th = 'text-left font-medium text-text-secondary px-3 py-2 whitespace-nowrap'
const td = 'px-3 py-2 align-top'

function pct(x: number | null | undefined): string {
  return x === null || x === undefined ? '—' : `${(x * 100).toFixed(1)}%`
}

function num(x: number | null | undefined, digits = 3): string {
  return x === null || x === undefined ? '—' : x.toFixed(digits)
}

function Section({ id, title, children, note }: { id: string; title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section id={id} className="mb-8 scroll-mt-4">
      <h2 className="text-base font-semibold text-text-primary mb-1">{title}</h2>
      {note && <p className="text-xs text-text-muted mb-2 max-w-4xl">{note}</p>}
      {children}
    </section>
  )
}

function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto border border-border rounded-lg bg-surface">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-bg border-b border-border-strong">
            {head.map((h) => (
              <th key={h} className={th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

function EmptyRow({ cols, text }: { cols: number; text: string }) {
  return (
    <tr>
      <td colSpan={cols} className="px-3 py-6 text-center text-text-muted">{text}</td>
    </tr>
  )
}

const NO_RUNS = 'No official run registered yet (Amendment 14 not approved; no run before "go").'

function ArchiveBadge() {
  return <Badge tone="warning">from the archive</Badge>
}

function runName(r: ReportRun) {
  return `${r.label} · ${r.condition}${r.grounding === null ? '' : r.grounding ? ' grounded' : ' plain'}${r.alpha === null ? '' : ` α=${r.alpha}`}`
}

function Three({ t }: { t: ThreeNumbers }) {
  return (
    <span className="whitespace-nowrap">
      {pct(t.rate_excl_abstain)} <span className="text-text-muted">· {pct(t.abstain_rate)} · {pct(t.rate_all_answers)}</span>
    </span>
  )
}

export function ReportPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const mock = searchParams.get('mock') === '1'
  const { data, isLoading, error } = useReport(mock)

  const setMock = (on: boolean) => {
    const next = new URLSearchParams(searchParams)
    if (on) next.set('mock', '1')
    else next.delete('mock')
    setSearchParams(next)
  }

  const exportJson = () => {
    if (!data) return
    const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `official_report${mock ? '_MOCK' : ''}.json`
    a.click()
    URL.revokeObjectURL(url)
  }
  const exportMd = async () => {
    const res = await fetch(`${API_BASE}/api/report/export.md${mock ? '?mock=true' : ''}`)
    downloadMarkdown(`official_report${mock ? '_MOCK' : ''}.md`, await res.text())
  }

  return (
    <div className="max-w-[1400px]">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
        <div>
          <h1 className="text-lg font-semibold text-text-primary">Official evaluation report</h1>
          <p className="text-sm text-text-secondary max-w-3xl">
            Pipeline order, from the run registry and stored outputs only. Rates are panel-detected hallucination
            rates (panel primary; DeepSeek-only sensitive; Llama and Qwen supporting).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-text-secondary">
            <input type="checkbox" checked={mock} onChange={(e) => setMock(e.target.checked)} />
            Mock data (walkthrough)
          </label>
          <button onClick={exportJson} disabled={!data} className="px-3 py-1.5 text-sm border border-border-strong rounded-md hover:bg-bg disabled:opacity-50">
            Export JSON
          </button>
          <button onClick={exportMd} disabled={!data} className="px-3 py-1.5 text-sm border border-border-strong rounded-md hover:bg-bg disabled:opacity-50">
            Export Markdown
          </button>
        </div>
      </div>
      {mock && (
        <div className="mb-4 rounded-md border border-warning/40 bg-white px-3 py-2 text-sm text-warning">
          MOCK DATA — synthetic numbers for the page walkthrough. Nothing on this view is a result.
        </div>
      )}
      <nav className="flex flex-wrap gap-x-3 gap-y-1 text-xs mb-6">
        {SECTIONS.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="text-primary hover:underline">{label}</a>
        ))}
      </nav>
      <LoadState isLoading={isLoading} error={error} />
      {data && <ReportBody data={data} mock={mock} />}
    </div>
  )
}

function ReportBody({ data, mock }: { data: Report; mock: boolean }) {
  const c = data.configuration
  const runs = data.runs
  const groups = Array.from(new Set(c.frozen_parameters.map((p) => p.group)))
  return (
    <>
      <Section id="config" title="a. Configuration" note={<>Frozen parameter table (<span className="font-mono">llm/frozen_parameters.py</span>), table hash <span className="font-mono">{c.table_hash ?? '—'}</span>; git HEAD <span className="font-mono">{c.git_head ?? '—'}</span>.</>}>
        <div className="flex flex-wrap gap-2 mb-3 text-xs">
          <Badge tone={c.amendment14.approved ? 'success' : 'warning'}>
            Amendment 14: {c.amendment14.approved ? 'approved' : 'draft, not approved'}
          </Badge>
          {c.amendments_in_decision_record.map((a) => (
            <Badge key={a} tone="neutral">{a.split(':')[0]}</Badge>
          ))}
        </div>
        {groups.map((g) => (
          <details key={g} className="mb-2" open={g === 'Judge'}>
            <summary className="text-sm font-medium cursor-pointer">{g}</summary>
            <Table head={['Parameter', 'Value', 'Status', 'Source', 'Decision']}>
              {c.frozen_parameters.filter((p) => p.group === g).map((p) => (
                <tr key={p.parameter} className="border-b border-border last:border-b-0">
                  <td className={td}>{p.parameter}</td>
                  <td className={`${td} font-mono text-xs break-all max-w-md`}>{typeof p.value === 'string' ? p.value : JSON.stringify(p.value)}</td>
                  <td className={td}><Badge tone={p.status.startsWith('frozen') ? 'success' : 'warning'}>{p.status}</Badge></td>
                  <td className={`${td} font-mono text-xs`}>{p.source}</td>
                  <td className={`${td} text-xs text-text-secondary`}>{p.decision || '—'}</td>
                </tr>
              ))}
            </Table>
          </details>
        ))}
      </Section>

      <Section id="data" title="b. Data and sampling">
        <Table head={['Set', 'n', 'Positions', 'Seed', 'Frozen list']}>
          <tr className="border-b border-border"><td className={td}>Test sample (official)</td><td className={td}>{data.data.test_sample.n}</td><td className={td}>{data.data.test_sample.positions}</td><td className={td}>{data.data.test_sample.seed}</td><td className={`${td} font-mono text-xs`}>{data.data.test_sample.frozen_list}</td></tr>
          <tr><td className={td}>Dev set (alpha selection)</td><td className={td}>{data.data.dev_set.n}</td><td className={td}>{data.data.dev_set.positions}</td><td className={td}>42</td><td className={`${td} font-mono text-xs`}>{data.data.dev_set.frozen_list}</td></tr>
        </Table>
        <p className="text-sm mt-2">
          Leakage exclusion <span className="font-mono">{data.data.leakage.mode}</span>: {data.data.leakage.rules}. Pool {data.data.leakage.n_pool ?? '—'} questions; exclusion sets built {data.data.leakage.built_at_utc ?? '—'} (sha256 <span className="font-mono">{data.data.leakage.sha256?.slice(0, 12) ?? '—'}</span>).
        </p>
      </Section>

      <Section id="alpha" title="c. Alpha selection">
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="text-sm font-medium mb-1 flex items-center gap-2">Stage 1 of record (context relevance) <ArchiveBadge /></div>
            <Table head={['alpha', '% questions with a relevant item', 'mean relevance score']}>
              {data.alpha_selection.stage1.table.map((r) => (
                <tr key={r.alpha} className="border-b border-border last:border-b-0">
                  <td className={td}>{r.alpha}{data.alpha_selection.stage1.top_two?.includes(r.alpha) && <> <Badge tone="primary">stage 2</Badge></>}</td>
                  <td className={td}>{r.pct_questions_with_relevant}</td>
                  <td className={td}>{r.mean_relevance_score}</td>
                </tr>
              ))}
            </Table>
            <p className="text-xs text-text-muted mt-1">{data.alpha_selection.stage1.reasoning}</p>
          </div>
          <div>
            <div className="text-sm font-medium mb-1">Stage 2 (panel label, pooled over grounding)</div>
            {data.alpha_selection.stage2 ? (
              <div className="text-sm space-y-1">
                <div>alpha* = <b>{data.alpha_selection.stage2.alpha_star}</b>, alpha_abl = <b>{data.alpha_selection.stage2.alpha_abl}</b></div>
                <div className="text-xs text-text-secondary">Rule applied: {data.alpha_selection.stage2.tie_rule_used}</div>
              </div>
            ) : (
              <div className="text-sm text-text-muted border border-border rounded-lg px-3 py-6 text-center bg-surface">
                Stage 2 not run yet. official_params.json: <span className="font-mono">{data.alpha_selection.official_params_status}</span> (Condition C on the test sample stays locked).
              </div>
            )}
          </div>
        </div>
      </Section>

      <Section id="retrieval" title="d. Retrieval per run">
        <Table head={['Run', 'Context items (mean)', 'Empty context', 'Retrieval latency p50 / p95 (s)']}>
          {runs.length === 0 && <EmptyRow cols={4} text={NO_RUNS} />}
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b border-border last:border-b-0">
              <td className={td}>{runName(r)}</td>
              <td className={td}>{r.retrieval.context_items_mean ?? '—'}</td>
              <td className={td}>{r.retrieval.empty_context ?? '— (A has no retrieval)'}</td>
              <td className={td}>{num(r.retrieval.latency_sec.p50)} / {num(r.retrieval.latency_sec.p95)}</td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="generation" title="e. Generation per run" note="ABSTAIN count by the panel label. Latency per answer (record schema r3): end-to-end = retrieval + generation, generation only = the generator call. Answers without r3 timing (archived runs) show only the run-duration fallback.">
        <Table head={['Run', 'n', 'ABSTAIN', 'Tokens in / out', 'Cost', 'End-to-end p50 / p95 (s)', 'Generation p50 / p95 (s)', 'Failures']}>
          {runs.length === 0 && <EmptyRow cols={8} text={NO_RUNS} />}
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b border-border last:border-b-0">
              <td className={td}>{runName(r)}</td>
              <td className={td}>{r.generation.n}</td>
              <td className={td}>{r.generation.abstain_panel}</td>
              <td className={td}>{r.generation.tokens_in.toLocaleString()} / {r.generation.tokens_out.toLocaleString()}</td>
              <td className={td}>{r.generation.cost_usd === null ? '—' : `$${r.generation.cost_usd.toFixed(3)}`}</td>
              <td className={td}>
                {r.latency.schema_r3_answers > 0
                  ? `${num(r.latency.end_to_end_sec.p50, 2)} / ${num(r.latency.end_to_end_sec.p95, 2)}`
                  : <>{num(r.latency.run_duration_per_answer_sec, 2)} <Badge tone="warning">fallback: run duration / answers</Badge></>}
              </td>
              <td className={td}>{num(r.latency.generation_sec.p50, 2)} / {num(r.latency.generation_sec.p95, 2)}</td>
              <td className={td}>{r.generation.failures ?? '—'}</td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="judging" title="f. Judging and budget">
        <Table head={['Run', 'Panel labels', 'Llama', 'DeepSeek', 'Qwen3-235B', 'UNRESOLVED', 'Incomplete', 'Failures left']}>
          {runs.length === 0 && <EmptyRow cols={8} text={NO_RUNS} />}
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b border-border last:border-b-0 text-xs">
              <td className={`${td} text-sm`}>{runName(r)}</td>
              {['panel', 'llama', 'deepseek', 'qwen3-235b'].map((k) => (
                <td key={k} className={td}>{Object.entries(r.judging.label_distribution[k] ?? {}).map(([l, n]) => `${l} ${n}`).join(', ')}</td>
              ))}
              <td className={td}>{r.judging.unresolved}</td>
              <td className={td}>{r.judging.incomplete_panels}</td>
              <td className={td}>{r.judging.failures_unresolved}</td>
            </tr>
          ))}
        </Table>
        <div className="mt-3 text-sm flex flex-wrap gap-4">
          <span>DeepInfra budget: <b>{data.budget.budget?.deepinfra_budget_usd == null ? 'not set' : `$${data.budget.budget.deepinfra_budget_usd.toFixed(2)}`}</b></span>
          <span>Spent: <b>${data.budget.spent_usd.toFixed(4)}</b></span>
          <span>Remaining: <b>{data.budget.remaining_usd == null ? '—' : `$${data.budget.remaining_usd.toFixed(4)}`}</b></span>
          <span>Judging cost of these runs: ${data.budget.judging_cost_usd.toFixed(4)}</span>
        </div>
        {data.budget.ledger.length > 0 && (
          <ul className="mt-1 text-xs text-text-secondary list-disc pl-5">
            {data.budget.ledger.map((l, i) => <li key={i}>{l.label}: ${l.cost_usd.toFixed(4)}{l.recorded_at ? ` (${l.recorded_at})` : ''}</li>)}
          </ul>
        )}
      </Section>

      <Section id="results" title="g. Results" note="Each cell: hallucination rate excl. ABSTAIN · ABSTAIN rate · hallucination rate over all answers.">
        <Table head={['Run', 'Panel (primary)', ...JUDGES.map((j) => `${j} (${JUDGE_ROLE[j]})`), 'Cosine mean / median', 'NF2 citation validity', `NF3: end-to-end p95 (C ≤ 15 s)`]}>
          {runs.length === 0 && <EmptyRow cols={7} text={NO_RUNS} />}
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b border-border last:border-b-0">
              <td className={td}>{runName(r)}</td>
              <td className={`${td} font-medium`}><Three t={r.results.panel} /></td>
              {JUDGES.map((j) => <td key={j} className={td}><Three t={r.results.judges[j]} /></td>)}
              <td className={td}>{num(r.results.cosine.mean)} / {num(r.results.cosine.median)}</td>
              <td className={td}>{pct(r.results.nf2_citation_validity)}</td>
              <td className={td} title={r.results.latency_vs_nf3.method}>
                {num(r.results.latency_vs_nf3.value_sec, 2)} s{' '}
                {r.results.latency_vs_nf3.fallback && <Badge tone="warning">fallback</Badge>}{' '}
                {r.results.latency_vs_nf3.meets !== null && <Badge tone={r.results.latency_vs_nf3.meets ? 'success' : 'danger'}>{r.results.latency_vs_nf3.meets ? 'meets' : 'misses'}</Badge>}
              </td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="comparisons" title="h. Comparisons" note="Rule (e): consistent only if the panel and DeepSeek-only differences (rate excl. ABSTAIN) have the same sign; otherwise inconclusive.">
        <Table head={['Comparison', 'Panel', 'DeepSeek', 'Llama', 'Qwen3-235B', 'Flag']}>
          {data.comparisons.map((cmp) => (
            <tr key={cmp.comparison} className="border-b border-border last:border-b-0">
              <td className={td}>{cmp.comparison}</td>
              <td className={td}>{cmp.available ? num(cmp.difference_panel, 4) : '—'}</td>
              {JUDGES.map((j) => <td key={j} className={td}>{cmp.available ? num(cmp.difference_by_judge?.[j], 4) : '—'}</td>)}
              <td className={td}>
                {cmp.available ? <Badge tone={cmp.flag === 'consistent' ? 'success' : 'warning'}>{cmp.flag}</Badge> : <span className="text-text-muted">not available</span>}
              </td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="reliability" title="i. Judge reliability">
        <Table head={['Run', "Fleiss' kappa (3 categories, complete items)", 'n complete', "Krippendorff's alpha (nominal)"]}>
          {runs.length === 0 && <EmptyRow cols={4} text={NO_RUNS} />}
          {runs.map((r) => (
            <tr key={r.run_id} className="border-b border-border last:border-b-0">
              <td className={td}>{runName(r)}</td>
              <td className={td}>{num(r.reliability.fleiss_kappa_3cat)}</td>
              <td className={td}>{r.reliability.n_complete}</td>
              <td className={td}>{num(r.reliability.krippendorff_alpha_nominal)}</td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="limitations" title="j. Limitations">
        <div className="flex flex-wrap gap-2 mb-2">
          <Badge tone="warning">Amendment 14: {data.limitations.amendment14_status}</Badge>
          <Badge tone="neutral">wording: {data.limitations.wording}</Badge>
        </div>
        <details className="mb-2">
          <summary className="text-sm font-medium cursor-pointer">Amendment 14 text</summary>
          <pre className="whitespace-pre-wrap text-xs bg-bg border border-border rounded-md p-3 max-h-[60vh] overflow-y-auto">{data.limitations.amendment14_text ?? '—'}</pre>
        </details>
        {Object.entries(data.limitations.archived_evidence).map(([k, e]) => (
          <details key={k} className="mb-2">
            <summary className="text-sm cursor-pointer flex items-center gap-2 font-medium">
              {k.replace(/_/g, ' ')} <ArchiveBadge /> <span className="font-mono text-xs text-text-muted">{e.path}</span>
            </summary>
            <pre className="whitespace-pre-wrap text-xs bg-bg border border-border rounded-md p-3 max-h-[60vh] overflow-y-auto">{e.text ?? 'not found'}</pre>
          </details>
        ))}
      </Section>

      <Section id="items" title="k. Item drill-down">
        <ItemDrilldown runs={runs} mock={mock} />
      </Section>
      <p className="text-xs text-text-muted">l. Export: use “Export JSON” / “Export Markdown” at the top of the page.</p>
    </>
  )
}

function ItemDrilldown({ runs, mock }: { runs: ReportRun[]; mock: boolean }) {
  const [runId, setRunId] = useState<string | null>(null)
  const [qid, setQid] = useState<number | null>(null)
  const items = useReportItems(runId, mock)
  const item = useReportItem(runId, qid, mock)
  if (runs.length === 0) return <div className="text-sm text-text-muted">{NO_RUNS}</div>
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(260px,360px)_1fr]">
      <div className="border border-border rounded-lg bg-surface overflow-hidden">
        <select
          className="w-full px-2.5 py-2 text-sm border-b border-border bg-surface"
          value={runId ?? ''}
          onChange={(e) => { setRunId(e.target.value || null); setQid(null) }}
        >
          <option value="">Choose a run…</option>
          {runs.map((r) => <option key={r.run_id} value={r.run_id}>{runName(r)}</option>)}
        </select>
        <ul className="max-h-[60vh] overflow-y-auto">
          {items.data?.items.map((it) => (
            <li key={it.question_id}>
              <button
                onClick={() => setQid(it.question_id)}
                className={`w-full text-left px-3 py-2 border-b border-border text-xs ${qid === it.question_id ? 'bg-primary-soft' : 'hover:bg-bg'}`}
              >
                <div className="text-text-primary">q{it.question_id} — {it.title}</div>
                <div className="text-text-muted">panel {it.panel_label} · {JUDGES.map((j) => `${j} ${it.judge_labels[j] ?? '—'}`).join(' · ')}</div>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="min-w-0">
        <LoadState isLoading={item.isLoading && qid !== null} error={item.error} />
        {!item.data && <div className="text-sm text-text-muted">Choose a run and an answer.</div>}
        {item.data && (
          <div className="space-y-3 text-sm">
            <div className="font-medium">{item.data.title} <Badge tone="primary">panel {item.data.panel.panel_label}</Badge></div>
            <Block title="Reference (accepted answer)"><div className="text-xs whitespace-pre-wrap">{item.data.reference_html}</div></Block>
            <Block title={`Retrieved context (${item.data.context.length})`}>
              {item.data.context.length === 0 ? <span className="text-xs text-text-muted">none</span> : item.data.context.map((c, i) => (
                <div key={i} className="text-xs whitespace-pre-wrap border-b border-border last:border-b-0 py-1">[K{i + 1}] {c.chunk_text}</div>
              ))}
            </Block>
            <Block title="Candidate answer"><div className="text-xs whitespace-pre-wrap">{item.data.candidate}</div></Block>
            {JUDGES.map((j) => (
              <Block key={j} title={`${j}: ${item.data!.judges[j]?.label ?? 'no record'}`}>
                {(item.data!.judges[j]?.claims ?? []).map((cl, i) => (
                  <div key={i} className="text-xs border-b border-border last:border-b-0 py-1">
                    <Badge tone={cl.verdict === 'SUPPORTED' ? 'success' : cl.verdict === 'UNVERIFIABLE' ? 'neutral' : 'danger'}>{cl.verdict}</Badge>{' '}
                    {cl.severity && <Badge tone="warning">{cl.severity}</Badge>} {cl.claim}
                    {cl.evidence && <div className="text-text-muted mt-0.5">{cl.evidence}</div>}
                  </div>
                ))}
              </Block>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="border border-border rounded-lg bg-surface">
      <div className="px-3 py-1.5 border-b border-border text-xs font-medium text-text-secondary">{title}</div>
      <div className="px-3 py-2">{children}</div>
    </div>
  )
}
