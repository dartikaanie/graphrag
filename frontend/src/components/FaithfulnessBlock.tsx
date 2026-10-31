import type { PanelMetricDefinition, PanelMetricRun } from '@/api/hooks'
import { mdTable } from '@/lib/markdown'

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const TH = 'text-left font-medium px-3 py-2 whitespace-nowrap'
const TD = 'px-3 py-2 whitespace-nowrap'
const BEST = 'bg-success/10 text-success font-semibold'

type Name = (id: string) => string

/** Runs with the highest value of `get` among runs that have a panel verdict
 * (only meaningful with ≥2 candidates; ties all count as best). */
function bestIds(runIds: string[], runs: Record<string, PanelMetricRun | null>, get: (r: PanelMetricRun) => number | null | undefined): Set<string> {
  const vals = runIds.map((id) => [id, runs[id]?.panel ? get(runs[id]!) : null] as const).filter(([, v]) => v != null)
  if (vals.length < 2) return new Set()
  const max = Math.max(...vals.map(([, v]) => v as number))
  return new Set(vals.filter(([, v]) => v === max).map(([id]) => id))
}

export function FaithfulnessBlock({ title, definition, runIds, runs, name }: {
  title: string
  definition?: PanelMetricDefinition
  runIds: string[]
  runs: Record<string, PanelMetricRun | null>
  name: Name
}) {
  const usable = runIds.filter((id) => runs[id] && !runs[id]!.note)
  const bestF = bestIds(runIds, runs, (r) => r.panel?.faithfulness_micro)
  const bestC = bestIds(runIds, runs, (r) => r.panel?.citation_micro)

  return (
    <div className="mb-6">
      <h3 className="text-sm font-medium text-text-primary mb-2">{title}</h3>
      {definition && (
        <div className="text-xs text-text-secondary border border-border rounded-lg px-3 py-2 mb-3 bg-bg">
          <div><span className="font-medium">Unit judged:</span> {definition.unit}</div>
          <div className="mt-1"><span className="font-medium">How calculated:</span> {definition.calculation}</div>
          <ol className="mt-2 list-decimal pl-5 space-y-0.5">
            <li>Each judge extracts the claims of every attempted answer and marks each claim <b>YES</b>/<b>NO</b> (supported by the retrieved context). Null values are skipped.</li>
            <li><b>Micro</b> (per judge) = Σ YES ÷ Σ (YES + NO) over all claims of the run. <b>Macro</b> = mean of each answer's own YES ÷ rated ratio.</li>
            <li><b>Panel</b> micro = the three judges' counts pooled (Σ YES of all judges ÷ Σ rated of all judges); panel macro = mean of the three judges' macro values.</li>
            <li>ABSTAIN answers have no claims, so they are excluded from both; they are not counted as faithful or unfaithful.</li>
          </ol>
        </div>
      )}
      {usable.length === 0 ? (
        <div className="text-sm text-text-muted border border-border rounded-lg px-3 py-2.5">
          {runIds.map((id) => runs[id]?.note).find(Boolean) ?? 'Not run yet (no official panel output).'}
        </div>
      ) : (
        <>
          {bestF.size > 0 && (
            <div className="text-xs border border-success/40 bg-success/10 text-success rounded-lg px-3 py-2 mb-3">
              ★ Most faithful: <b>{[...bestF].map(name).join(', ')}</b> ({pct(runs[[...bestF][0]]?.panel?.faithfulness_micro)} micro
              {bestC.size > 0 && <> · best citation support: <b>{[...bestC].map(name).join(', ')}</b> {pct(runs[[...bestC][0]]?.panel?.citation_micro)}</>})
            </div>
          )}
          <div className="text-xs text-text-secondary mb-1">Panel summary</div>
          <div className="overflow-x-auto border border-border rounded-lg mb-3">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg border-b border-border-strong text-text-secondary">
                  <th className={TH}>Run</th><th className={TH}>Answers</th><th className={TH}>ABSTAIN (excluded)</th>
                  <th className={TH}>Claims rated</th><th className={TH}>YES / NO</th>
                  <th className={TH}>Faithfulness micro</th><th className={TH}>Faithfulness macro</th>
                  <th className={TH}>Citation support micro</th><th className={TH}>Citation support macro</th>
                </tr>
              </thead>
              <tbody>
                {runIds.map((id) => {
                  const r = runs[id]
                  const p = r?.panel
                  if (!r || r.note || !p) {
                    return (
                      <tr key={id} className="border-b border-border last:border-b-0">
                        <td className={`${TD} font-medium text-text-primary`}>{name(id)}</td>
                        <td colSpan={8} className="px-3 py-2 text-text-muted">
                          {r?.note ?? (r ? `Panel verdict withheld — judges so far: ${(r.judges_present ?? []).join(', ')}` : 'not run')}
                        </td>
                      </tr>
                    )
                  }
                  return (
                    <tr key={id} className="border-b border-border last:border-b-0">
                      <td className={`${TD} font-medium text-text-primary`}>{name(id)}{bestF.has(id) && ' ★'}</td>
                      <td className={TD}>{p.n_answers}</td>
                      <td className={TD}>{p.n_abstain}</td>
                      <td className={TD}>{p.claims_rated}</td>
                      <td className={TD}>{p.claims_supported} / {p.claims_unsupported}</td>
                      <td className={`${TD} ${bestF.has(id) ? BEST : 'font-medium'}`}>{pct(p.faithfulness_micro)}</td>
                      <td className={TD}>{pct(p.faithfulness_macro)}</td>
                      <td className={`${TD} ${bestC.has(id) ? BEST : ''}`}>{pct(p.citation_micro)}</td>
                      <td className={TD}>{pct(p.citation_macro)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="text-xs text-text-secondary mb-1">Per-judge breakdown (calculation detail)</div>
          <div className="overflow-x-auto border border-border rounded-lg">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg border-b border-border-strong text-text-secondary">
                  <th className={TH}>Run</th><th className={TH}>Judge</th><th className={TH}>Model</th><th className={TH}>Temp</th>
                  <th className={TH}>Answers</th><th className={TH}>ABSTAIN</th><th className={TH}>Unusable</th>
                  <th className={TH}>Answers rated</th><th className={TH}>YES</th><th className={TH}>NO</th>
                  <th className={TH}>Faithfulness micro (YES ÷ rated)</th><th className={TH}>Macro</th>
                  <th className={TH}>Citation support micro</th><th className={TH}>Macro</th>
                </tr>
              </thead>
              <tbody>
                {runIds.flatMap((id) => {
                  const judges = runs[id]?.per_judge ?? []
                  if (judges.length === 0) return []
                  const bestJ = Math.max(...judges.map((j) => j.faithfulness_micro ?? -1))
                  return judges.map((j, i) => (
                    <tr key={`${id}|${j.judge_id}`} className="border-b border-border last:border-b-0">
                      <td className={`${TD} font-medium text-text-primary`}>{i === 0 ? name(id) : ''}</td>
                      <td className={TD}>{j.judge_id}</td>
                      <td className={`${TD} font-mono`}>{j.judge_model ?? '—'}</td>
                      <td className={TD}>{j.temperature ?? '—'}</td>
                      <td className={TD}>{j.n}</td><td className={TD}>{j.n_abstain}</td><td className={TD}>{j.n_unusable}</td>
                      <td className={TD}>{j.n_answers_rated}</td>
                      <td className={TD}>{j.claims_supported}</td><td className={TD}>{j.claims_unsupported}</td>
                      <td className={`${TD} ${bestF.has(id) && j.faithfulness_micro === bestJ ? BEST : ''}`}>
                        {j.claims_supported} ÷ {j.claims_rated} = {pct(j.faithfulness_micro)}
                      </td>
                      <td className={TD}>{pct(j.faithfulness_macro)}</td>
                      <td className={TD}>{j.cit_supported} ÷ {j.cit_rated} = {pct(j.citation_micro)}</td>
                      <td className={TD}>{pct(j.citation_macro)}</td>
                    </tr>
                  ))
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

export function faithfulnessMarkdown(
  title: string, definition: PanelMetricDefinition | undefined, runIds: string[],
  runs: Record<string, PanelMetricRun | null>, name: Name,
): string[] {
  const lines = [`### ${title}`, '']
  if (definition) lines.push(`- **Unit judged:** ${definition.unit}`, `- **How calculated:** ${definition.calculation}`, '')
  if (!runIds.some((id) => runs[id] && !runs[id]!.note)) {
    return [...lines, runIds.map((id) => runs[id]?.note).find(Boolean) ?? 'Not run yet (no official panel output).', '']
  }
  const bestF = bestIds(runIds, runs, (r) => r.panel?.faithfulness_micro)
  const bestC = bestIds(runIds, runs, (r) => r.panel?.citation_micro)
  if (bestF.size) lines.push(`**Most faithful:** ${[...bestF].map(name).join(', ')}`, '')
  const rows = runIds.map((id) => {
    const r = runs[id]; const p = r?.panel
    if (!r || r.note || !p) return [name(id), r?.note ?? 'not run', '—', '—', '—', '—', '—', '—', '—']
    return [name(id) + (bestF.has(id) ? ' ★' : ''), String(p.n_answers), String(p.n_abstain), String(p.claims_rated),
      `${p.claims_supported} / ${p.claims_unsupported}`, pct(p.faithfulness_micro) + (bestF.has(id) ? ' ★' : ''),
      pct(p.faithfulness_macro), pct(p.citation_micro) + (bestC.has(id) ? ' ★' : ''), pct(p.citation_macro)]
  })
  lines.push('**Panel summary**', '', mdTable(['Run', 'Answers', 'ABSTAIN (excluded)', 'Claims rated', 'YES / NO', 'Faithfulness micro',
    'Faithfulness macro', 'Citation support micro', 'Citation support macro'], rows), '')
  const jrows: string[][] = []
  for (const id of runIds) {
    for (const j of runs[id]?.per_judge ?? []) {
      jrows.push([name(id), j.judge_id, j.judge_model ?? '—', String(j.temperature ?? '—'), String(j.n), String(j.n_abstain),
        String(j.n_unusable), String(j.n_answers_rated), `${j.claims_supported} ÷ ${j.claims_rated} = ${pct(j.faithfulness_micro)}`,
        pct(j.faithfulness_macro), `${j.cit_supported} ÷ ${j.cit_rated} = ${pct(j.citation_micro)}`, pct(j.citation_macro)])
    }
  }
  lines.push('**Per-judge breakdown**', '', mdTable(['Run', 'Judge', 'Model', 'Temp', 'Answers', 'ABSTAIN', 'Unusable', 'Answers rated',
    'Faithfulness micro (YES ÷ rated)', 'Faithfulness macro', 'Citation support micro', 'Citation support macro'], jrows), '')
  return lines
}
