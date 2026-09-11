import type { ReactNode } from 'react'

type BadgeTone = 'neutral' | 'primary' | 'success' | 'warning' | 'danger'

const toneClasses: Record<BadgeTone, string> = {
  neutral: 'bg-bg text-text-secondary border-border-strong',
  primary: 'bg-primary-soft text-primary border-primary-border',
  success: 'bg-white text-success border-success/40',
  warning: 'bg-white text-warning border-warning/40',
  danger: 'bg-white text-danger border-danger/40',
}

export function Badge({ tone = 'neutral', children }: { tone?: BadgeTone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-xs font-medium ${toneClasses[tone]}`}
    >
      {children}
    </span>
  )
}
