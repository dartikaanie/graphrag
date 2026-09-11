import { Link } from 'react-router-dom'
import { Badge } from './Badge'
import type { RetrievedContextItem } from '@/types/run'

const STAGE_LABEL: Record<string, string> = {
  graph_traversal: 'Graph Traversal',
  semantic_expansion: 'Semantic Expansion',
}

export function RetrievalEvidenceList({ items }: { items: RetrievedContextItem[] }) {
  if (items.length === 0) {
    return <div className="text-sm text-text-muted">No context was retrieved for this question.</div>
  }

  return (
    <div className="flex flex-col gap-2">
      {items.map((item, i) => (
        <div key={i} className="border border-border rounded-md p-3 text-sm">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            {item.question_id != null && (
              <Link
                to={`/questions/${item.question_id}`}
                className="font-mono text-xs bg-primary-soft border border-primary-border text-primary rounded-sm px-1"
              >
                SO-{item.question_id}
              </Link>
            )}
            {item.answer_id != null && (
              <Link to={`/answers/${item.answer_id}`} className="text-xs text-text-secondary hover:text-primary">
                Answer #{item.answer_id}
              </Link>
            )}
            {item.is_accepted && <Badge tone="success">Accepted</Badge>}
            {item.source_stage && <Badge tone="neutral">{STAGE_LABEL[item.source_stage] ?? item.source_stage}</Badge>}
            {item.trust_weight != null && (
              <span className="text-xs text-text-secondary">trust={item.trust_weight.toFixed(2)}</span>
            )}
            {item.combined_score != null && (
              <span className="text-xs text-text-secondary">score={item.combined_score.toFixed(2)}</span>
            )}
            {item.hop != null && <span className="text-xs text-text-secondary">{item.hop}-hop</span>}
          </div>
          <p className="text-text-secondary line-clamp-3">{item.chunk_text}</p>
        </div>
      ))}
    </div>
  )
}
