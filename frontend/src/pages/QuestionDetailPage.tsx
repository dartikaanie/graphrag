import { useParams, Link, useNavigate } from 'react-router-dom'
import { useNodeSubgraph, useQuestionAnswers, useQuestionDetail } from '@/api/hooks'
import { AttributePanel } from '@/components/AttributePanel'
import { Badge } from '@/components/Badge'
import { GraphView } from '@/components/GraphView'
import { EdgeLegend } from '@/components/EdgeLegend'
import type { GraphNode } from '@/types/graph'

const PRIORITY_FIELDS = ['id', 'domainTag', 'trustScore', 'title', 'score', 'viewCount', 'creationDate']

export function QuestionDetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const { data: detail, isLoading, isError } = useQuestionDetail(id)
  const { data: answers } = useQuestionAnswers(id)
  const { data: graph, isLoading: graphLoading } = useNodeSubgraph('question', id, 2)

  const handleNodeClick = (node: GraphNode) => {
    if (node.type === 'Question') navigate(`/questions/${node.properties.id}`)
    else if (node.type === 'Answer') navigate(`/answers/${node.properties.id}`)
    else if (node.type === 'Tag') navigate(`/tags/${encodeURIComponent(node.properties.name as string)}`)
  }

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !detail) return <div className="text-sm text-danger">Question not found.</div>

  const attrs = detail.attributes
  const body = typeof attrs.body === 'string' ? attrs.body : null
  const acceptedAnswer = answers?.find((a) => Boolean((a as Record<string, unknown>).isAccepted)) as
    | Record<string, unknown>
    | undefined

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/questions" className="hover:text-primary">
          ← Back to Questions
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1 break-words">
        {typeof attrs.title === 'string' ? attrs.title : `Question #${id}`}
      </h1>
      <p className="text-sm text-text-secondary mb-4">
        {acceptedAnswer ? (
          <>
            Accepted answer:{' '}
            <Link to={`/answers/${acceptedAnswer.id}`} className="text-primary hover:underline">
              #{String(acceptedAnswer.id)}
            </Link>
          </>
        ) : (
          <span className="text-text-muted">No accepted answer</span>
        )}
      </p>

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
          <div className="mb-4">
            {graphLoading && (
              <div className="border border-border rounded-lg bg-surface p-4 text-sm text-text-muted h-[280px] flex items-center justify-center">
                Loading graph...
              </div>
            )}
            {graph && (
              <>
                <GraphView data={graph} height={280} onNodeClick={handleNodeClick} centerNodeId={`Question-${id}`} />
                <div className="mt-2 border border-border rounded-lg bg-surface px-3 py-2.5">
                  <EdgeLegend present={new Set(graph.links.map((l) => l.type))} />
                </div>
              </>
            )}
          </div>
          <div className="border border-border rounded-lg bg-surface p-4">
            <h2 className="text-sm font-medium text-text-secondary mb-3">
              Answers {answers ? `(${answers.length})` : ''}
            </h2>
            <div className="flex flex-col gap-2">
              {answers?.map((a) => {
                const answer = a as Record<string, unknown>
                return (
                  <Link
                    key={String(answer.id)}
                    to={`/answers/${answer.id}`}
                    className="flex items-center justify-between border border-border rounded-md px-3 py-2 hover:bg-bg"
                  >
                    <span className="text-sm text-text-primary">Answer #{String(answer.id)}</span>
                    <span className="flex items-center gap-2">
                      {Boolean(answer.isAccepted) && <Badge tone="success">Accepted</Badge>}
                      <span className="text-xs text-text-secondary">score: {String(answer.score ?? '—')}</span>
                    </span>
                  </Link>
                )
              })}
              {answers?.length === 0 && <div className="text-sm text-text-muted">No answers found.</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
