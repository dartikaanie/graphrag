import { Link, useNavigate, useParams } from 'react-router-dom'
import { useRunResultDetail, useRunResultGraph } from '@/api/hooks'
import { AnswerComparisonPanel } from '@/components/AnswerComparisonPanel'
import { RetrievalEvidenceList } from '@/components/RetrievalEvidenceList'
import { PromptTranscript } from '@/components/PromptTranscript'
import { GraphView } from '@/components/GraphView'
import { EdgeLegend } from '@/components/EdgeLegend'
import { Badge } from '@/components/Badge'
import type { GraphNode } from '@/types/graph'

export function RunResultDetailPage() {
  const { condition = 'a', run_id = '', question_id = '' } = useParams()
  const navigate = useNavigate()
  const { data, isLoading, isError } = useRunResultDetail(run_id, question_id)
  const conditionUpper = condition.toUpperCase()
  const isGraphRag = conditionUpper === 'C'
  const isDualLevel = conditionUpper === 'D'
  const { data: graph, isLoading: graphLoading } = useRunResultGraph(run_id, question_id)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !data) return <div className="text-sm text-danger">Result not found.</div>

  const isRag = conditionUpper === 'B' || conditionUpper === 'C' || conditionUpper === 'D'

  const handleNodeClick = (node: GraphNode) => {
    if (node.type === 'Question') navigate(`/questions/${node.properties.id}`)
    else if (node.type === 'Answer') navigate(`/answers/${node.properties.id}`)
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to={`/experiment/${condition}/runs/${run_id}`} className="hover:text-primary">
          ← Back to Run Results
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1 break-words">{data.title}</h1>
      <p className="text-sm text-text-secondary mb-4">
        Question #{data.question_id} · Accepted answer:{' '}
        <Link to={`/answers/${data.accepted_answer_id}`} className="text-primary hover:underline">
          #{data.accepted_answer_id}
        </Link>
      </p>

      <div className="border border-border rounded-lg bg-surface p-4 mb-4 flex flex-wrap gap-6 text-sm">
        <div>
          <span className="text-text-secondary">Cosine similarity (0–1): </span>
          <span className="font-medium">{data.cosine_similarity.toFixed(4)}</span>
        </div>
        {data.has_valid_citation !== undefined && (
          <div>
            <span className="text-text-secondary">Valid citation (NF2): </span>
            {data.has_valid_citation ? <Badge tone="success">Yes</Badge> : <Badge tone="danger">No</Badge>}
          </div>
        )}
        {conditionUpper === 'C' && (
          <>
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
        {isDualLevel && (
          <>
            {data.retrieval_latency_sec != null && (
              <div>
                <span className="text-text-secondary">Retrieval latency: </span>
                <span className="font-medium">{data.retrieval_latency_sec.toFixed(2)}s</span>
              </div>
            )}
            {data.n_low_level_candidates != null && (
              <div>
                <span className="text-text-secondary">Low-Level → High-Level candidates: </span>
                <span className="font-medium">
                  {data.n_low_level_candidates} → {data.n_high_level_candidates ?? 0}
                </span>
              </div>
            )}
            {data.require_grounding !== undefined && (
              <div>
                <span className="text-text-secondary">Grounding constraint: </span>
                <Badge tone={data.require_grounding ? 'success' : 'neutral'}>
                  {data.require_grounding ? 'On' : 'Off'}
                </Badge>
              </div>
            )}
          </>
        )}
      </div>

      <AnswerComparisonPanel
        llmAnswer={data.llm_answer}
        groundTruthAnswer={data.ground_truth_answer}
        llmLabel={`LLM Answer (Condition ${conditionUpper})`}
      />

      <div className="mb-4 mt-4">
        <PromptTranscript messages={data.prompt_messages ?? []} />
      </div>

      {isRag && (
        <div className="border border-border rounded-lg bg-surface p-4 mb-4">
          <h2 className="text-sm font-medium text-text-secondary mb-3">Retrieval Evidence</h2>
          <RetrievalEvidenceList items={data.retrieved_context ?? []} />
        </div>
      )}

      {(isGraphRag || isDualLevel) && (
        <div>
          <h2 className="text-sm font-medium text-text-secondary mb-2">
            {isGraphRag
              ? 'Retrieval Path — Entity Anchoring → Graph Traversal → Semantic Expansion'
              : 'Retrieval Path — Entity Anchoring → Low-Level → High-Level Retrieval'}
          </h2>
          {graphLoading && (
            <div className="border border-border rounded-lg bg-surface p-4 text-sm text-text-muted h-[320px] flex items-center justify-center">
              Loading graph...
            </div>
          )}
          {graph && (
            <>
              <GraphView data={graph} height={320} onNodeClick={handleNodeClick} centerNodeId={`Question-${data.question_id}`} />
              <div className="mt-2 border border-border rounded-lg bg-surface px-3 py-2.5">
                <EdgeLegend present={new Set(graph.links.map((l) => l.type))} />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
