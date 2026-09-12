/** "12.3s" for short runs, "2m 05s" once a run crosses a minute -- shared
 * between MetricSummaryCards and HistoryComparePage so "usage time" reads
 * consistently everywhere it's compared. */
export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return `${minutes}m ${String(rest).padStart(2, '0')}s`
}
