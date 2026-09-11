import type { ReactNode } from 'react'

export interface Column<T> {
  key: string
  header: string
  render: (row: T) => ReactNode
  className?: string
}

interface DataTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T) => string | number
  onRowClick?: (row: T) => void
  emptyLabel?: string
}

export function DataTable<T>({ columns, rows, rowKey, onRowClick, emptyLabel = 'No results.' }: DataTableProps<T>) {
  return (
    <div className="overflow-x-auto border border-border rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-bg border-b border-border-strong">
            {columns.map((col) => (
              <th
                key={col.key}
                className="text-left font-medium text-text-secondary px-4 py-2.5 whitespace-nowrap"
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-4 py-8 text-center text-text-muted">
                {emptyLabel}
              </td>
            </tr>
          )}
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={() => onRowClick?.(row)}
              className={`border-b border-border last:border-b-0 ${onRowClick ? 'cursor-pointer hover:bg-bg' : ''}`}
            >
              {columns.map((col) => (
                <td key={col.key} className={`px-4 py-2.5 align-top ${col.className ?? ''}`}>
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
