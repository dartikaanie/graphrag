export type GraphNodeType = 'Question' | 'Answer' | 'Tag' | 'User'

export interface GraphNode {
  id: string
  type: GraphNodeType
  label: string
  properties: Record<string, unknown>
}

export type EdgeType =
  | 'HAS_ACCEPTED_ANSWER'
  | 'HAS_ANSWER'
  | 'TAGGED_WITH'
  | 'IS_RELATED_TO'
  | 'TAG_COOCCUR'
  | 'EMBED_SIM'
  | 'AUTHOR_TRUST'

export interface GraphLink {
  source: string
  target: string
  type: EdgeType
  weight: number | null
}

export interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}
