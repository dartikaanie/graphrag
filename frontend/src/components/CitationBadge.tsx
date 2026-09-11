import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

const CITATION_RE = /\[SO-(\d+)\]/g

export function CitationBadge({ questionId }: { questionId: string }) {
  return (
    <Link
      to={`/questions/${questionId}`}
      className="inline-flex items-center rounded-sm bg-primary-soft border border-primary-border px-1 font-mono text-xs text-primary hover:bg-primary hover:text-white"
      title={`Jump to source question #${questionId}`}
    >
      SO-{questionId}
    </Link>
  )
}

/** Splits text on [SO-<id>] citations and renders each as a CitationBadge link. */
export function renderWithCitations(text: string): ReactNode[] {
  const parts: ReactNode[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null
  CITATION_RE.lastIndex = 0
  while ((match = CITATION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index))
    parts.push(<CitationBadge key={`${match.index}-${match[1]}`} questionId={match[1]} />)
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return parts
}
