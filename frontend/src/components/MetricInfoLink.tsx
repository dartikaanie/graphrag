import { Info } from 'lucide-react'
import { Link } from 'react-router-dom'

/**
 * Small inline "i" link next to a metric's label/value, pointing to its
 * full definition on /docs/metrics (MetricsReferencePage) -- the single
 * source of truth for metric definitions. Deliberately just an icon + a
 * link, never a duplicated explanation, so there is exactly one place a
 * metric's definition can drift out of date.
 */
export function MetricInfoLink({ metricId, label = 'metric definition' }: { metricId: string; label?: string }) {
  return (
    <Link
      to={`/docs/metrics#${metricId}`}
      title={`What is this? (${label})`}
      className="inline-flex text-text-muted hover:text-primary align-middle"
    >
      <Info className="h-3.5 w-3.5" />
    </Link>
  )
}
