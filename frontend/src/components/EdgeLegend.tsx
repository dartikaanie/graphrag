import type { EdgeType } from '@/types/graph'
import { EDGE_COLORS, EDGE_LABELS } from '@/lib/graphColors'

const ORDER: EdgeType[] = [
  'ANCHOR',
  'GRAPH_TRAVERSAL',
  'SEMANTIC_EXPANSION',
  'RETRIEVED',
  'HAS_ACCEPTED_ANSWER',
  'HAS_ANSWER',
  'IS_RELATED_TO',
  'TAG_COOCCUR',
  'EMBED_SIM',
  'TAGGED_WITH',
  'AUTHOR_TRUST',
]

export function EdgeLegend({ present }: { present?: Set<EdgeType> }) {
  const types = present ? ORDER.filter((t) => present.has(t)) : ORDER
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5">
      {types.map((type) => (
        <div key={type} className="flex items-center gap-1.5 text-xs text-text-secondary">
          <span className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: EDGE_COLORS[type] }} />
          <span>{EDGE_LABELS[type]}</span>
        </div>
      ))}
    </div>
  )
}
