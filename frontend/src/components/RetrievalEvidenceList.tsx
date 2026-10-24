import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Badge } from './Badge'
import type { ProvenanceStep, RetrievedContextItem, RunResultDetail } from '@/types/run'

const STAGE_LABEL: Record<string, string> = {
  graph_traversal: 'Graph Traversal',
  semantic_expansion: 'Semantic Expansion',
  low_level: 'Low-Level Retrieval',
  high_level: 'High-Level Retrieval',
}

const NOTE = 'The path shows how the source was retrieved from the knowledge graph, not why the model used it.'

/** "Question 111 -(TAG_COOCCUR)-> Question 222 -(HAS_ACCEPTED_ANSWER)-> Answer 333" */
function pathText(path: ProvenanceStep[]): string {
  return path
    .map((s, i) => `${s.node_type} ${s.node_id}${i < path.length - 1 && s.edge_type_to_next ? ` -(${s.edge_type_to_next})->` : ''}`)
    .join(' ')
}

function citedQuestionIds(record?: Partial<RunResultDetail>): Set<number> {
  return new Set((record?.valid_cited_source_ids ?? []).map((x) => Number(x)))
}

function IdList({ label, ids, total }: { label: string; ids: number[]; total?: number }) {
  return (
    <div>
      <span className="font-medium">{label}</span> ({total ?? ids.length}
      {total != null && total > ids.length ? `, first ${ids.length} shown` : ''}):{' '}
      <span className="font-mono">{ids.length ? ids.join(', ') : '—'}</span>
    </div>
  )
}

export function RetrievalEvidenceList({
  items,
  record,
  condition,
}: {
  items: RetrievedContextItem[]
  /** The stored run record; with it the "Show provenance" toggle appears. */
  record?: Partial<RunResultDetail>
  condition?: string
}) {
  const [showProv, setShowProv] = useState(false)
  const cited = citedQuestionIds(record)
  const hasPaths = items.some((it) => it.path && it.path.length > 0)
  if (items.length === 0) {
    return <div className="text-sm text-text-muted">No context was retrieved for this question.</div>
  }

  // Items already arrive in final ranked order (fuse_and_rank/fuse_dual_level
  // sort before capping to top_k), so position in this list IS the rank --
  // shown explicitly here rather than making the reader infer it from score
  // order alone.
  return (
    <div className="flex flex-col gap-2">
      {record && (
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <button
            type="button"
            onClick={() => setShowProv((v) => !v)}
            className="border border-border-strong rounded-sm px-2 py-1 hover:bg-bg"
            aria-pressed={showProv}
          >
            {showProv ? 'Hide provenance' : 'Show provenance'}
          </button>
          {record.provenance_version && <Badge tone="neutral">provenance {record.provenance_version}</Badge>}
        </div>
      )}
      {showProv && !hasPaths && (
        <div className="text-xs text-text-muted border border-border rounded-md p-2">
          {condition?.toUpperCase() === 'A'
            ? 'Condition A has no retrieval, so there is nothing to trace.'
            : 'This run was written before per-item provenance existed (no path stored), so no path can be shown. Sources are listed without paths.'}
        </div>
      )}
      {items.map((item, i) => (
        <div key={i} className="border border-border rounded-md p-3 text-sm">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <Badge tone="neutral">Rank #{i + 1}</Badge>
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
              <span className="text-xs font-medium text-text-primary">ranking score={item.combined_score.toFixed(3)}</span>
            )}
            {item.relevance_score != null && (
              <span className="text-xs font-medium text-text-primary">ranking score={item.relevance_score.toFixed(3)}</span>
            )}
            {item.hop != null && <span className="text-xs text-text-secondary">{item.hop}-hop</span>}
          </div>
          <p className="text-text-secondary line-clamp-3">{item.chunk_text}</p>
          {showProv && (
            <div className="mt-2 border-t border-border pt-2 text-xs text-text-secondary flex flex-col gap-1">
              <div className="flex items-center gap-2 flex-wrap">
                {item.question_id != null && (
                  <Badge tone={cited.has(item.question_id) ? 'success' : 'neutral'}>
                    {cited.has(item.question_id) ? 'Cited' : 'Retrieved, not cited'}
                  </Badge>
                )}
                {item.source_stage === 'vector_only' && <span>No knowledge-graph path (vector search hit).</span>}
              </div>
              {item.path && item.path.length > 0 && item.source_stage !== 'vector_only' && (
                <div className="font-mono break-words">{pathText(item.path)}</div>
              )}
              {item.alt_paths && item.alt_paths.length > 0 && (
                <details>
                  <summary className="cursor-pointer">Other paths to this answer ({item.alt_paths.length})</summary>
                  <ul className="mt-1 flex flex-col gap-0.5 font-mono break-words">
                    {item.alt_paths.map((alt, j) => (
                      <li key={j}>
                        {pathText(alt.path)}
                        {alt.source_stage ? ` [${STAGE_LABEL[alt.source_stage] ?? alt.source_stage}]` : ''}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {item.path?.some((s) => s.edge_type_to_next === 'SEMANTIC_SIM') && (
                <div className="text-text-muted">SEMANTIC_SIM is a similarity link from the vector index, not a knowledge-graph edge.</div>
              )}
            </div>
          )}
        </div>
      ))}
      {showProv && record && (record.not_selected || (record.below_cutoff && record.below_cutoff.length > 0)) && (
        <div className="border border-border rounded-md p-3 text-xs text-text-secondary flex flex-col gap-1">
          <div className="font-medium text-text-primary">Retrieved but not selected</div>
          {record.not_selected && (
            <>
              <IdList label="Answers" ids={record.not_selected.answer_ids} total={record.not_selected.n_answer_ids} />
              <IdList label="Questions visited" ids={record.not_selected.question_ids} total={record.not_selected.n_question_ids} />
            </>
          )}
          {record.below_cutoff && record.below_cutoff.length > 0 && (
            <div>
              <span className="font-medium">Just below the cutoff</span> (answer, rank, score):{' '}
              <span className="font-mono">{record.below_cutoff.map(([a, r, sc]) => `${a} (#${r}, ${sc})`).join('; ')}</span>
            </div>
          )}
          <div className="text-text-muted">
            Candidates cut inside the graph query (e.g. the 2-hop LIMIT 50) are not listed.
          </div>
        </div>
      )}
      {showProv && <div className="text-xs text-text-muted">{NOTE}</div>}
    </div>
  )
}
