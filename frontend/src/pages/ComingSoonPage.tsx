export function ComingSoonPage({ title, phase }: { title: string; phase: string }) {
  return (
    <div>
      <h1 className="text-lg font-semibold text-text-primary mb-4">{title}</h1>
      <div className="border border-border rounded-lg bg-surface p-6 text-sm text-text-muted">
        Built in {phase} of PLAN_UI_UX.md.
      </div>
    </div>
  )
}
