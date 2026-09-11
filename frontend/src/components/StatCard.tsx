interface StatCardProps {
  label: string
  value: string | number
}

export function StatCard({ label, value }: StatCardProps) {
  return (
    <div className="border border-border rounded-lg bg-surface px-4 py-3">
      <div className="text-xs text-text-secondary">{label}</div>
      <div className="text-2xl font-semibold text-text-primary mt-1">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  )
}
