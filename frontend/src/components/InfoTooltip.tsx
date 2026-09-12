import { useState, type ReactNode } from 'react'

/**
 * Small "(?)" hint icon with a hover/focus tooltip. Used next to
 * configuration field labels across the Run pages so the same term (e.g.
 * "seed", "top_k") is always explained the same way, everywhere it appears.
 */
export function InfoTooltip({ text }: { text: ReactNode }) {
  const [open, setOpen] = useState(false)

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        tabIndex={0}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="w-3.5 h-3.5 inline-flex items-center justify-center rounded-full border border-border-strong text-[9px] leading-none text-text-secondary hover:bg-primary-soft hover:text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-border"
        aria-label="More info"
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute left-1/2 bottom-full -translate-x-1/2 mb-1.5 w-56 z-20 rounded-md border border-border-strong bg-surface px-2.5 py-1.5 text-xs font-normal text-text-primary shadow-lg"
        >
          {text}
        </span>
      )}
    </span>
  )
}
