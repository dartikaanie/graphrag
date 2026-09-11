import type { EdgeType, GraphNodeType } from '@/types/graph'

// Mirrors the --color-edge-* tokens in index.css. Duplicated as plain hex here
// because react-force-graph paints on a <canvas>, which can't read CSS custom
// properties directly. Keep these two in sync if the palette ever changes.
export const EDGE_COLORS: Record<EdgeType, string> = {
  HAS_ACCEPTED_ANSWER: '#2563eb',
  HAS_ANSWER: '#93c5fd',
  IS_RELATED_TO: '#0ea5e9',
  TAG_COOCCUR: '#64748b',
  EMBED_SIM: '#94a3b8',
  TAGGED_WITH: '#cbd5e1',
  AUTHOR_TRUST: '#c4b5fd',
}

export const EDGE_LABELS: Record<EdgeType, string> = {
  HAS_ACCEPTED_ANSWER: 'Accepted answer',
  HAS_ANSWER: 'Has answer',
  IS_RELATED_TO: 'Related / duplicate',
  TAG_COOCCUR: 'Tag co-occurrence',
  EMBED_SIM: 'Embedding similarity',
  TAGGED_WITH: 'Tagged with',
  AUTHOR_TRUST: 'Author trust',
}

export const NODE_COLORS: Record<GraphNodeType, string> = {
  Question: '#2563eb',
  Answer: '#16a34a',
  Tag: '#94a3b8',
  User: '#c4b5fd',
}
