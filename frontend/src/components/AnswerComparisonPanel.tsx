import { renderWithCitations } from './CitationBadge'

interface AnswerComparisonPanelProps {
  llmAnswer: string
  groundTruthAnswer: string
  llmLabel: string
}

export function AnswerComparisonPanel({ llmAnswer, groundTruthAnswer, llmLabel }: AnswerComparisonPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="border border-border rounded-lg bg-surface p-4">
        <h2 className="text-sm font-medium text-text-secondary mb-2">{llmLabel}</h2>
        <div className="text-sm text-text-primary whitespace-pre-wrap leading-relaxed">
          {renderWithCitations(llmAnswer)}
        </div>
      </div>
      <div className="border border-border rounded-lg bg-surface p-4">
        <h2 className="text-sm font-medium text-text-secondary mb-2">Accepted Answer (Ground Truth)</h2>
        <div
          className="prose prose-sm max-w-none text-text-primary"
          dangerouslySetInnerHTML={{ __html: groundTruthAnswer }}
        />
      </div>
    </div>
  )
}
