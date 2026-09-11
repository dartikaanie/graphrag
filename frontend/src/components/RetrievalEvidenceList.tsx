import { Link } from 'react-router-dom'
import type { RetrievedContextItem } from '@/types/run'

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
            {item.trust_weight != null && (
              <span className="text-xs text-text-secondary">trust={item.trust_weight.toFixed(2)}</span>
            )}
            {item.hop != null && <span className="text-xs text-text-secondary">{item.hop}-hop</span>}
          </div>
          <p className="text-text-secondary line-clamp-3">{item.chunk_text}</p>
        </div>
      ))}
    </div>
  )
}
