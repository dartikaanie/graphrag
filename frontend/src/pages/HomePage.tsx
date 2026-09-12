import { Link, useNavigate } from 'react-router-dom'
import { useHistory, usePartialGraph, useStatsSummary } from '@/api/hooks'
import { StatCard } from '@/components/StatCard'
import { GraphView } from '@/components/GraphView'
import { EdgeLegend } from '@/components/EdgeLegend'
import { Badge } from '@/components/Badge'
import type { GraphNode } from '@/types/graph'
import type { RunCondition } from '@/types/run'

const CONDITIONS: { key: RunCondition; label: string }[] = [
  { key: 'A', label: 'A — Pure LLM' },
  { key: 'B', label: 'B — LLM + RAG' },
  { key: 'C', label: 'C — LLM + GraphRAG' },
  { key: 'D', label: 'D — Dual-Level Retrieval (LightRAG-adapted)' },
]

function LatestRunCard({ condition, label }: { condition: RunCondition; label: string }) {
  const { data, isLoading } = useHistory(condition, 1, 1)
  const latest = data?.items[0]

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <div className="text-sm font-medium text-text-primary mb-2">{label}</div>
      {isLoading && <div className="text-xs text-text-muted">Loading...</div>}
      {!isLoading && !latest && <div className="text-xs text-text-muted">No runs yet.</div>}
      {latest && (
        <>
          <div className="text-xl font-semibold text-text-primary mb-1">
            {latest.cosine_similarity_mean != null ? latest.cosine_similarity_mean.toFixed(4) : '—'}
          </div>
          <div className="text-xs text-text-secondary mb-2">
            n={latest.n_processed}/{latest.n_sample_target} · <Badge tone={latest.status === 'success' ? 'success' : 'warning'}>{latest.status}</Badge>
          </div>
          <Link to={`/history/${latest.history_id}`} className="text-xs text-primary hover:underline">
            View details →
          </Link>
        </>
      )}
    </div>
  )
}

export function HomePage() {
  const { data, isLoading, isError } = useStatsSummary()
  const { data: graph, isLoading: graphLoading, isError: graphError } = usePartialGraph(100)
  const navigate = useNavigate()

  const handleNodeClick = (node: GraphNode) => {
    if (node.type === 'Question') navigate(`/questions/${node.properties.id}`)
    else if (node.type === 'Answer') navigate(`/answers/${node.properties.id}`)
    else if (node.type === 'Tag') navigate(`/tags/${encodeURIComponent(node.properties.name as string)}`)
  }

  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">Knowledge Graph Summary</h1>
      <p className="text-sm text-text-secondary mb-6">
        SORD dataset overview, from the Neo4j knowledge graph used by Condition C.
      </p>

      {isLoading && <div className="text-sm text-text-muted">Loading stats...</div>}
      {isError && <div className="text-sm text-danger">Failed to load stats — is the backend running?</div>}

      {data && (
        <div className="grid grid-cols-4 gap-4 mb-8">
          <StatCard label="Questions" value={data.total_questions} />
          <StatCard label="Answers" value={data.total_answers} />
          <StatCard label="Tags" value={data.total_tags} />
          <StatCard label="Edges" value={data.total_edges} />
        </div>
      )}

      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-medium text-text-secondary">
          Partial Knowledge Graph
          <span className="text-text-muted font-normal"> — top 100 questions by score, with their answers and tags</span>
        </h2>
      </div>
      {graphLoading && (
        <div className="border border-border rounded-lg bg-surface p-6 text-sm text-text-muted">
          Loading graph...
        </div>
      )}
      {graphError && (
        <div className="border border-border rounded-lg bg-surface p-6 text-sm text-danger">
          Failed to load graph.
        </div>
      )}
      {graph && (
        <>
          <GraphView data={graph} height={420} onNodeClick={handleNodeClick} />
          <div className="mt-2 border border-border rounded-lg bg-surface px-3 py-2.5">
            <EdgeLegend />
          </div>
        </>
      )}

      <h2 className="text-sm font-medium text-text-secondary mt-8 mb-2">Latest Runs</h2>
      <div className="grid grid-cols-4 gap-4">
        {CONDITIONS.map((c) => (
          <LatestRunCard key={c.key} condition={c.key} label={c.label} />
        ))}
      </div>
    </div>
  )
}
