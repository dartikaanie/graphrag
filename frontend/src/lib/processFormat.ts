// Formatting helpers and labels for the Process & Parameters page.

export const STEP_LABELS: Record<string, string> = {
  data: '1. Data',
  sampling: '2. Sampling',
  kg_build: '3. Knowledge graph',
  trust: '4. Trust scoring',
  conditions: '5. Conditions',
  c_ranking: '6. C ranking',
  alpha_selection: '7. Alpha selection',
  judge: '8. Judge',
  official: '9. Official run',
  metrics: '10. Metrics',
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return 'not set'
  if (typeof value === 'number') return Number.isInteger(value) && Math.abs(value) >= 10000 ? value.toLocaleString() : String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return value.map(formatValue).join(', ')
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => `${k}: ${formatValue(v)}`)
      .join(', ')
  }
  return String(value)
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
