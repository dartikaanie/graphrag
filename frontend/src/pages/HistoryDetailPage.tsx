import { Link, useNavigate, useParams } from 'react-router-dom'
import { useHistoryDetail } from '@/api/hooks'
import { MetricSummaryCards } from '@/components/MetricSummaryCards'
import { DataTable, type Column } from '@/components/DataTable'
import { Badge } from '@/components/Badge'
import { MetricInfoLink } from '@/components/MetricInfoLink'
import type { RunResultItem } from '@/types/run'
import type { JudgeEvaluationSummary } from '@/types/judge'

function kappaTone(kappa: number): 'success' | 'warning' | 'danger' {
  if (kappa > 0.6) return 'success'
  if (kappa >= 0.4) return 'warning'
  return 'danger'
}

export function HistoryDetailPage() {
  const { history_id = '' } = useParams()
  const navigate = useNavigate()
  const { data: run, isLoading, isError } = useHistoryDetail(history_id)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !run) return <div className="text-sm text-danger">History entry not found.</div>

  const columns: Column<RunResultItem>[] = [
    { key: 'question_id', header: 'ID', render: (r) => <span className="font-mono text-xs">{r.question_id}</span> },
    { key: 'similarity', header: 'Similarity', render: (r) => (r.similarity != null ? r.similarity.toFixed(4) : '—') },
  ]

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/history" className="hover:text-primary">
          ← Back to History
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">
        Run {run.run_id} — Condition {run.condition}
      </h1>
      <p className="text-sm text-text-secondary mb-4">
        {new Date(run.created_at).toLocaleString()} · Mode: {run.mode} · Provider: {run.params.provider} · Model:{' '}
        {run.params.model} · Status: <Badge tone={run.status === 'success' || run.status === 'completed' ? 'success' : 'warning'}>{run.status}</Badge>
      </p>

      {run.summary && (
        <div className="mb-6">
          <h2 className="text-sm font-medium text-text-secondary mb-2">Result Summary</h2>
          <MetricSummaryCards summary={run.summary} condition={run.condition} />
        </div>
      )}

      <div className="mb-6">
        <h2 className="text-sm font-medium text-text-secondary mb-2 inline-flex items-center gap-1">
          Hasil LLM-as-Judge
          <MetricInfoLink metricId="hallucination-rate" label="LLM-as-Judge" />
        </h2>
        {run.judge_evaluations && run.judge_evaluations.length > 0 ? (
          <div className="overflow-x-auto border border-border rounded-lg">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg border-b border-border-strong">
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Judge Model</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Temp</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Rounds</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">%Faktual</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">%Sebagian</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">%Penuh</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Mean Faithfulness</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Mean Relevance</th>
                  <th className="text-left font-medium text-text-secondary px-3 py-2">Kappa</th>
                </tr>
              </thead>
              <tbody>
                {run.judge_evaluations.map((ev: JudgeEvaluationSummary, i) => (
                  <tr key={i} className="border-b border-border last:border-b-0">
                    <td className="px-3 py-2">{ev.judge_provider}/{ev.judge_model}</td>
                    <td className="px-3 py-2">{ev.judge_temperature}</td>
                    <td className="px-3 py-2">{ev.majority_rounds}</td>
                    <td className="px-3 py-2 text-success">{ev.pct_faktual ?? '—'}%</td>
                    <td className="px-3 py-2 text-warning">{ev.pct_sebagian ?? '—'}%</td>
                    <td className="px-3 py-2 text-danger">{ev.pct_penuh ?? '—'}%</td>
                    <td className="px-3 py-2">{ev.mean_faithfulness != null ? ev.mean_faithfulness.toFixed(4) : '—'}</td>
                    <td className="px-3 py-2">{ev.mean_answer_relevance != null ? ev.mean_answer_relevance.toFixed(4) : '—'}</td>
                    <td className="px-3 py-2">
                      {ev.kappa_value != null ? (
                        <Badge tone={kappaTone(ev.kappa_value)}>{ev.kappa_value.toFixed(3)}</Badge>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-sm text-text-muted border border-border rounded-lg px-3 py-2.5">
            <span>Belum dinilai LLM-as-Judge.</span>
            {run.output_path && (
              <Link
                to={`/evaluation/judge?prefill_input_path=${encodeURIComponent(run.output_path)}&condition=${run.condition}`}
                className="text-primary hover:underline"
              >
                Nilai sekarang →
              </Link>
            )}
          </div>
        )}
      </div>

      {run.results.length > 0 ? (
        <div>
          <h2 className="text-sm font-medium text-text-secondary mb-2">Per-Question Results</h2>
          <DataTable
            columns={columns}
            rows={run.results}
            rowKey={(r) => r.question_id}
            onRowClick={(r) => navigate(`/history/${history_id}/q/${r.question_id}`)}
          />
        </div>
      ) : (
        <div className="text-sm text-text-muted">
          No per-question results available (the underlying results file may have been moved or deleted).
        </div>
      )}
    </div>
  )
}
