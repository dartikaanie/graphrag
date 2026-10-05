// Shared markdown-export helpers -- used by History Compare's "Export as
// Markdown" and the Judge-v1 results page's "Download report (.md)" so
// both produce tables/files the same way instead of each reimplementing
// escaping/table/download logic.

// Markdown table cells can't contain a raw "|" or newline -- escape/strip
// so a stray value never breaks the table structure.
export function mdEscape(v: string): string {
  return v.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ')
}

export function mdTable(header: string[], rows: string[][]): string {
  const lines = [
    `| ${header.map(mdEscape).join(' | ')} |`,
    `|${header.map(() => '---').join('|')}|`,
    ...rows.map((row) => `| ${row.map(mdEscape).join(' | ')} |`),
  ]
  return lines.join('\n')
}

export function downloadMarkdown(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
