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
  // Synthetic edge types for a single run result's retrieval-path graph
  // (not real KG edges) -- see get_run_result_graph on the backend.
  | 'ANCHOR'
  | 'GRAPH_TRAVERSAL'
  | 'SEMANTIC_EXPANSION'
  | 'RETRIEVED'

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
