interface AttributePanelProps {
  attributes: Record<string, unknown>
  /** Field order to prioritize before falling back to remaining keys, per PLAN_UI_UX.md §7.3. */
  priorityFields?: string[]
  /** Fields to hide entirely (e.g. long body text rendered separately). */
  hideFields?: string[]
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

export function AttributePanel({ attributes, priorityFields = [], hideFields = [] }: AttributePanelProps) {
  const hidden = new Set(hideFields)
  const remainingKeys = Object.keys(attributes).filter(
    (k) => !priorityFields.includes(k) && !hidden.has(k),
  )
  const orderedKeys = [...priorityFields.filter((k) => k in attributes && !hidden.has(k)), ...remainingKeys]

  return (
    <dl className="divide-y divide-border">
      {orderedKeys.map((key) => (
        <div key={key} className="flex justify-between gap-4 py-2 text-sm">
          <dt className="text-text-secondary shrink-0">{key}</dt>
          <dd className="text-text-primary text-right break-words">{formatValue(attributes[key])}</dd>
        </div>
      ))}
    </dl>
  )
}
