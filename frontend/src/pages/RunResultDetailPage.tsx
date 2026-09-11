import { Link, useParams } from 'react-router-dom'
import { useRunResultDetail } from '@/api/hooks'
import { AnswerComparisonPanel } from '@/components/AnswerComparisonPanel'
import { RetrievalEvidenceList } from '@/components/RetrievalEvidenceList'
import { Badge } from '@/components/Badge'

export function RunResultDetailPage() {
  const { condition = 'a', run_id = '', question_id = '' } = useParams()
  const { data, isLoading, isError } = useRunResultDetail(run_id, question_id)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !data) return <div className="text-sm text-danger">Result not found.</div>

  const conditionUpper = condition.toUpperCase()
  const isRag = conditionUpper === 'B' || conditionUpper === 'C'

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to={`/experiment/${condition}/runs/${run_id}`} className="hover:text-primary">
          ← Back to Run Results
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1 break-words">{data.title}</h1>
      <p className="text-sm text-text-secondary mb-4">Question #{data.question_id}</p>

      <AnswerComparisonPanel
        llmAnswer={data.llm_answer}
        groundTruthAnswer={data.ground_truth_answer}
        llmLabel={`LLM Answer (Condition ${conditionUpper})`}
      />

      <div className="border border-border rounded-lg bg-surface p-4 my-4 flex flex-wrap gap-6 text-sm">
        <div>
          <span className="text-text-secondary">Cosine similarity (0–1): </span>
          <span className="font-medium">{data.cosine_similarity.toFixed(4)}</span>
        </div>
        {conditionUpper === 'C' && (
          <>
            <div>
              <span className="text-text-secondary">Valid citation (NF2): </span>
              {data.has_valid_citation ? <Badge tone="success">Yes</Badge> : <Badge tone="danger">No</Badge>}
            </div>
            {data.retrieval_latency_sec != null && (
              <div>
                <span className="text-text-secondary">Retrieval latency: </span>
                <span className="font-medium">{data.retrieval_latency_sec.toFixed(2)}s</span>
              </div>
            )}
            {data.n_anchors != null && (
              <div>
                <span className="text-text-secondary">Anchors → Graph → Expansion: </span>
                <span className="font-medium">
                  {data.n_anchors} → {data.n_graph_candidates ?? 0} → {data.n_expansion_candidates ?? 0}
                </span>
              </div>
            )}
          </>
        )}
      </div>

      {isRag && (
        <div className="border border-border rounded-lg bg-surface p-4">
          <h2 className="text-sm font-medium text-text-secondary mb-3">Retrieval Evidence</h2>
          <RetrievalEvidenceList items={data.retrieved_context ?? []} />
        </div>
      )}
    </div>
  )
}
