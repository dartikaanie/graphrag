import { useState, type ReactNode } from 'react'
import { Badge } from '@/components/Badge'
import type { SourceQuote } from '@/types/process'

export function ParamStatusBadge({ status }: { status: string }) {
  const tone = status === 'frozen' ? 'success' : status === 'pending' ? 'warning' : status === 'candidate' ? 'primary' : 'neutral'
  return <Badge tone={tone}>{status}</Badge>
}

export function StepStatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase()
  const tone = s.startsWith('done') || s === 'implemented' ? 'success' : s.startsWith('not started') ? 'neutral' : 'warning'
  return <Badge tone={tone}>{status}</Badge>
}

export function SourceRef({ file, line, lineEnd }: { file: string; line?: number | null; lineEnd?: number | null }) {
  const suffix = line ? (lineEnd && lineEnd !== line ? `:${line}-${lineEnd}` : `:${line}`) : ''
  return (
    <span className="font-mono text-xs text-text-muted break-all">
      {file}
      {suffix}
    </span>
  )
}

/** A verbatim passage from a repo document or code comment, with its source. */
export function QuoteBlock({ quote, label, collapsed = true }: { quote: SourceQuote; label?: string; collapsed?: boolean }) {
  const [open, setOpen] = useState(!collapsed)
  if (quote.error || !quote.text) {
    return (
      <div className="text-xs text-danger">
        Source not found: <SourceRef file={quote.file} /> ({quote.error})
      </div>
    )
  }
  const long = quote.text.length > 220
  const shown = open || !long ? quote.text : `${quote.text.slice(0, 220)}…`
  return (
    <div className="border-l-2 border-border-strong pl-3">
      {label && <div className="text-xs font-medium text-text-secondary mb-0.5">{label}</div>}
      <div className="text-sm text-text-primary whitespace-pre-wrap break-words">{shown}</div>
      <div className="flex items-center gap-2 mt-0.5">
        <SourceRef file={quote.file} line={quote.line} lineEnd={quote.line_end} />
        {long && (
          <button onClick={() => setOpen(!open)} className="text-xs text-primary hover:underline">
            {open ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
    </div>
  )
}

export function NotDocumented() {
  return <Badge tone="warning">Rationale not documented</Badge>
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-1.5">{children}</h3>
}

export function LoadState({ isLoading, error }: { isLoading: boolean; error: unknown }) {
  if (isLoading) return <div className="text-sm text-text-muted py-6">Loading…</div>
  if (error) return <div className="text-sm text-danger py-6">Could not load: {(error as Error).message}</div>
  return null
}
