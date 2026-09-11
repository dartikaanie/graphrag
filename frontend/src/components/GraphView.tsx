import { useCallback, useRef } from 'react'
import ForceGraph2D, { type NodeObject, type LinkObject, type ForceGraphMethods } from 'react-force-graph-2d'
import type { GraphData, GraphLink, GraphNode } from '@/types/graph'
import { EDGE_COLORS, NODE_COLORS } from '@/lib/graphColors'

interface GraphViewProps {
  data: GraphData
  height?: number
  onNodeClick?: (node: GraphNode) => void
  centerNodeId?: string
}

const NODE_RADIUS: Record<string, number> = { Question: 5, Answer: 4, Tag: 3.5, User: 3.5 }

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Short on-canvas label -- full detail (title, score, etc.) shows on hover instead. */
function shortLabel(node: GraphNode): string {
  switch (node.type) {
    case 'Question':
      return `Q#${node.properties.id}`
    case 'Answer':
      return `A#${node.properties.id}`
    case 'User':
      return `U#${node.properties.id}`
    case 'Tag':
    default:
      return node.label
  }
}

function hoverLabel(node: GraphNode): string {
  const p = node.properties
  switch (node.type) {
    case 'Question':
      return `<div style="max-width:240px;white-space:normal">
        <strong>Q#${p.id}</strong> ${escapeHtml(node.label)}<br/>
        score: ${p.score ?? '—'} · trust: ${typeof p.trustScore === 'number' ? p.trustScore.toFixed(2) : '—'}${p.domainTag ? ` · ${escapeHtml(String(p.domainTag))}` : ''}
      </div>`
    case 'Answer':
      return `<div>
        <strong>A#${p.id}</strong>${p.isAccepted ? ' · ✓ accepted' : ''}<br/>
        score: ${p.score ?? '—'} · trust: ${typeof p.trustScore === 'number' ? p.trustScore.toFixed(2) : '—'}
      </div>`
    case 'Tag':
      return `<div><strong>${escapeHtml(node.label)}</strong> · ${p.questionCount ?? '—'} questions</div>`
    case 'User':
      return `<div><strong>U#${p.id}</strong> · reputation: ${p.reputation ?? '—'}</div>`
    default:
      return escapeHtml(node.label)
  }
}

export function GraphView({ data, height = 360, onNodeClick, centerNodeId }: GraphViewProps) {
  const fgRef = useRef<ForceGraphMethods<NodeObject<GraphNode>, LinkObject<GraphNode, GraphLink>> | undefined>(
    undefined,
  )

  const handleNodeClick = useCallback(
    (node: NodeObject<GraphNode>) => {
      onNodeClick?.(node as unknown as GraphNode)
    },
    [onNodeClick],
  )

  const nodeCanvasObject = useCallback(
    (node: NodeObject<GraphNode>, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as unknown as GraphNode & { x: number; y: number }
      const radius = NODE_RADIUS[n.type] ?? 4
      const isCenter = centerNodeId === n.id
      ctx.beginPath()
      ctx.arc(n.x, n.y, isCenter ? radius * 1.6 : radius, 0, 2 * Math.PI)
      ctx.fillStyle = NODE_COLORS[n.type] ?? '#94a3b8'
      ctx.fill()
      if (isCenter) {
        ctx.lineWidth = 1.5 / globalScale
        ctx.strokeStyle = '#0f172a'
        ctx.stroke()
      }

      // Short id-only label always on -- full title/score/etc. shows in the
      // native hover tooltip (nodeLabel below) instead of cluttering the canvas.
      const label = shortLabel(n)
      const fontSize = 11 / globalScale
      ctx.font = `${fontSize}px Inter, system-ui, sans-serif`
      ctx.fillStyle = '#0f172a'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(label, n.x, n.y + radius + 2)
    },
    [centerNodeId],
  )

  return (
    <div className="rounded-lg border border-border overflow-hidden bg-surface" style={{ height }}>
      <ForceGraph2D
        ref={fgRef}
        graphData={data as unknown as { nodes: NodeObject<GraphNode>[]; links: LinkObject<GraphNode, GraphLink>[] }}
        height={height}
        nodeId="id"
        nodeLabel={(node) => hoverLabel(node as unknown as GraphNode)}
        nodeCanvasObject={nodeCanvasObject}
        nodePointerAreaPaint={(node, color, ctx) => {
          const n = node as unknown as GraphNode & { x: number; y: number }
          const radius = (NODE_RADIUS[n.type] ?? 4) + 2
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(n.x, n.y, radius, 0, 2 * Math.PI)
          ctx.fill()
        }}
        linkColor={(link) => EDGE_COLORS[(link as unknown as GraphLink).type] ?? '#cbd5e1'}
        linkWidth={1}
        linkDirectionalParticles={0}
        onNodeClick={handleNodeClick}
        cooldownTicks={100}
        enableNodeDrag={true}
      />
    </div>
  )
}
