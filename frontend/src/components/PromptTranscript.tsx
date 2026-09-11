import { useState } from 'react'
import type { PromptMessage } from '@/types/run'

const ROLE_LABEL: Record<string, string> = { system: 'System', user: 'User', assistant: 'Assistant' }

/**
 * Shows the EXACT 4-turn message array actually sent to the LLM for this
 * question -- for verifying the pipeline did what's expected, not just
 * trusting the final answer/score. Collapsed by default since it's long
 * (especially the final user turn for B/C, which embeds the full retrieved
 * context) and isn't needed on first glance at a result.
 */
export function PromptTranscript({ messages }: { messages: PromptMessage[] }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border border-border rounded-lg bg-surface p-4">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center justify-between w-full text-sm font-medium text-text-secondary"
      >
        <span>Full Prompt Sent to LLM ({messages.length} turns)</span>
        <span className="text-text-muted">{open ? '▲ Hide' : '▼ Show'}</span>
      </button>
      {open && (
        <div className="mt-3 flex flex-col gap-3">
          {messages.map((m, i) => (
            <div key={i} className="border border-border rounded-md p-3">
              <div className="text-xs font-medium text-primary mb-1">{ROLE_LABEL[m.role] ?? m.role}</div>
              <pre className="text-xs text-text-primary whitespace-pre-wrap font-mono leading-relaxed">
                {m.content}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
