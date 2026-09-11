interface PaginationProps {
  page: number
  totalPages: number
  onChange: (page: number) => void
}

export function Pagination({ page, totalPages, onChange }: PaginationProps) {
  return (
    <div className="flex items-center justify-between mt-4">
      <button
        className="px-3 py-1.5 text-sm border border-border-strong rounded-md text-text-primary disabled:opacity-40 disabled:cursor-not-allowed hover:bg-bg"
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
      >
        Previous
      </button>
      <span className="text-sm text-text-secondary">
        Page {page} of {totalPages.toLocaleString()}
      </span>
      <button
        className="px-3 py-1.5 text-sm border border-border-strong rounded-md text-text-primary disabled:opacity-40 disabled:cursor-not-allowed hover:bg-bg"
        disabled={page >= totalPages}
        onClick={() => onChange(page + 1)}
      >
        Next
      </button>
    </div>
  )
}
