import { Link, useNavigate, useParams } from 'react-router-dom'
import { useAnswerDetail, useNodeSubgraph } from '@/api/hooks'
import { AttributePanel } from '@/components/AttributePanel'
import { GraphView } from '@/components/GraphView'
import { EdgeLegend } from '@/components/EdgeLegend'
import type { GraphNode } from '@/types/graph'

const PRIORITY_FIELDS = ['id', 'score', 'isAccepted', 'authorReputation', 'trustScore']

export function AnswerDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const { data: detail, isLoading, isError } = useAnswerDetail(id)
  const { data: graph, isLoading: graphLoading } = useNodeSubgraph('answer', id, 2)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !detail) return <div className="text-sm text-danger">Answer not found.</div>

  const attrs = detail.attributes
  const body = typeof attrs.body === 'string' ? attrs.body : null
  const questionId = detail.graph_meta?.question_id

  const handleNodeClick = (node: GraphNode) => {
    if (node.type === 'Question') navigate(`/questions/${node.properties.id}`)
    else if (node.type === 'Answer') navigate(`/answers/${node.properties.id}`)
    else if (node.type === 'Tag') navigate(`/tags/${encodeURIComponent(node.properties.name as string)}`)
  }

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2 flex gap-4">
        <Link to="/answers" className="hover:text-primary">
          ← Back to Answers
        </Link>
        {questionId != null && (
          <Link to={`/questions/${questionId}`} className="hover:text-primary">
            ← Back to Question #{String(questionId)}
          </Link>
        )}
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">Answer #{id}</h1>

      <div className="grid grid-cols-2 gap-6">
        <div>
          <div className="border border-border rounded-lg bg-surface p-4 mb-4">
            <h2 className="text-sm font-medium text-text-secondary mb-2">Attributes</h2>
            <AttributePanel attributes={attrs} priorityFields={PRIORITY_FIELDS} hideFields={['body']} />
          </div>
          {body && (
            <div className="border border-border rounded-lg bg-surface p-4">
              <h2 className="text-sm font-medium text-text-secondary mb-2">Body</h2>
              <div
                className="prose prose-sm max-w-none text-text-primary"
                dangerouslySetInnerHTML={{ __html: body }}
              />
            </div>
          )}
        </div>
        <div>
          {graphLoading && (
            <div className="border border-border rounded-lg bg-surface p-4 text-sm text-text-muted h-[280px] flex items-center justify-center">
              Loading graph...
            </div>
          )}
          {graph && (
            <>
              <GraphView data={graph} height={280} onNodeClick={handleNodeClick} centerNodeId={`Answer-${id}`} />
              <div className="mt-2 border border-border rounded-lg bg-surface px-3 py-2.5">
                <EdgeLegend present={new Set(graph.links.map((l) => l.type))} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
