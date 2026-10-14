import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import type { DecisionEntry, DecisionsResponse } from '@/types/process'
import { SectionTitle, SourceRef } from './shared'
import { formatTime } from '@/lib/processFormat'

function Entry({ entry, focused }: { entry: DecisionEntry; focused: boolean }) {
  // Remounted (via key) when it becomes the focused entry, so it opens expanded.
  const [open, setOpen] = useState(focused)
  useEffect(() => {
    if (focused) document.getElementById(`decision-${entry.id}`)?.scrollIntoView({ block: 'start' })
  }, [focused, entry.id])

  return (
    <li id={`decision-${entry.id}`} className="relative pl-6">
      <span className="absolute left-0 top-1.5 h-2.5 w-2.5 rounded-full border-2 border-primary bg-surface" />
      <div className={`border rounded-lg bg-surface p-4 ${focused ? 'border-primary-border' : 'border-border'}`}>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="font-mono text-sm text-text-secondary">{entry.date ?? '—'}</span>
          <h2 className="text-sm font-semibold text-text-primary">
            {entry.number === 0 ? 'Decision record' : `Amendment ${entry.number}`}: {entry.title}
          </h2>
        </div>
        {entry.intro && <p className="text-sm text-text-secondary mt-1 whitespace-pre-wrap">{entry.intro}</p>}

        {entry.why && (
          <div className="mt-3">
            <SectionTitle>{entry.why.heading}</SectionTitle>
            <div className="text-sm text-text-primary whitespace-pre-wrap break-words">{entry.why.text}</div>
          </div>
        )}

        {entry.applied.length > 0 && (
          <div className="mt-3">
            <SectionTitle>Applied</SectionTitle>
            <ul className="text-sm text-text-primary space-y-1">
              {entry.applied.map((a) => (
                <li key={a} className="break-words">
                  {a}
                </li>
              ))}
            </ul>
          </div>
        )}

        {entry.commits.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-text-secondary">Commits cited:</span>
            {entry.commits.map((c) => (
              <span key={c} className="font-mono text-xs border border-border rounded-sm px-1.5 py-0.5">
                {c.slice(0, 7)}
              </span>
            ))}
          </div>
        )}

        <div className="mt-3 flex items-center justify-between gap-2">
          <SourceRef file="docs/DECISION_C_SCORING.md" line={entry.line} lineEnd={entry.line_end} />
          <button onClick={() => setOpen(!open)} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            {open ? 'Hide full text' : 'Show full text'}
          </button>
        </div>

        {open && (
          <div className="mt-3 pt-3 border-t border-border space-y-3">
            {entry.sections.map((s) => (
              <div key={`${s.line}-${s.heading}`}>
                {s.heading && <SectionTitle>{s.heading}</SectionTitle>}
                <div className="text-sm text-text-primary whitespace-pre-wrap break-words">{s.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </li>
  )
}

export function DecisionsTab({ data, focus }: { data: DecisionsResponse; focus: string | null }) {
  return (
    <div>
      <div className="text-xs text-text-muted mb-3">
        Parsed from <span className="font-mono">{data.source}</span> (last modified {formatTime(data.as_of)}). Text is
        shown as written in the document.
      </div>
      <ol className="relative space-y-4 before:absolute before:left-[4px] before:top-2 before:bottom-2 before:w-px before:bg-border">
        {data.decisions.map((d) => (
          <Entry key={focus === d.id ? `${d.id}-focused` : d.id} entry={d} focused={focus === d.id} />
        ))}
      </ol>
    </div>
  )
}
