import { Link, useNavigate, useParams } from 'react-router-dom'
import { useNodeSubgraph, useTagDetail } from '@/api/hooks'
import { DataTable, type Column } from '@/components/DataTable'
import { GraphView } from '@/components/GraphView'
import { EdgeLegend } from '@/components/EdgeLegend'
import type { QuestionListItem } from '@/types/api'
import type { GraphNode } from '@/types/graph'

export function TagDetailPage() {
  const { name = '' } = useParams()
  const navigate = useNavigate()
  const { data, isLoading, isError } = useTagDetail(name)
  const { data: graph, isLoading: graphLoading } = useNodeSubgraph('tag', name, 2)

  if (isLoading) return <div className="text-sm text-text-muted">Loading...</div>
  if (isError || !data) return <div className="text-sm text-danger">Tag not found.</div>

  const handleNodeClick = (node: GraphNode) => {
    if (node.type === 'Question') navigate(`/questions/${node.properties.id}`)
    else if (node.type === 'Answer') navigate(`/answers/${node.properties.id}`)
    else if (node.type === 'Tag') navigate(`/tags/${encodeURIComponent(node.properties.name as string)}`)
  }

  const columns: Column<QuestionListItem>[] = [
    { key: 'id', header: 'ID', render: (q) => <span className="font-mono text-xs">{q.id}</span> },
    { key: 'title', header: 'Title', className: 'max-w-md', render: (q) => <span className="line-clamp-2">{q.title}</span> },
    { key: 'score', header: 'Score', render: (q) => q.score ?? '—' },
    { key: 'view_count', header: 'Views', render: (q) => q.view_count?.toLocaleString() ?? '—' },
  ]

  return (
    <div>
      <div className="text-sm text-text-secondary mb-2">
        <Link to="/tags" className="hover:text-primary">
          ← Back to Tags
        </Link>
      </div>
      <h1 className="text-lg font-semibold text-text-primary mb-1">{data.name}</h1>
      <p className="text-sm text-text-secondary mb-4">
        {data.question_count?.toLocaleString() ?? '—'} questions tagged — showing top {data.questions.length} by score
      </p>

      <div className="mb-6">
        {graphLoading && (
          <div className="border border-border rounded-lg bg-surface p-4 text-sm text-text-muted h-[280px] flex items-center justify-center">
            Loading graph...
          </div>
        )}
        {graph && (
          <>
            <GraphView data={graph} height={280} onNodeClick={handleNodeClick} centerNodeId={`Tag-${name}`} />
            <div className="mt-2 border border-border rounded-lg bg-surface px-3 py-2.5">
              <EdgeLegend present={new Set(graph.links.map((l) => l.type))} />
            </div>
          </>
        )}
      </div>

      <DataTable
        columns={columns}
        rows={data.questions}
        rowKey={(q) => q.id}
        onRowClick={(q) => navigate(`/questions/${q.id}`)}
      />
    </div>
  )
}
