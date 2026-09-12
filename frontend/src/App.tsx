import { Route, Routes } from 'react-router-dom'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { HomePage } from '@/pages/HomePage'
import { MethodologyPage } from '@/pages/MethodologyPage'
import { MetricsReferencePage } from '@/pages/MetricsReferencePage'
import { QuestionListPage } from '@/pages/QuestionListPage'
import { QuestionDetailPage } from '@/pages/QuestionDetailPage'
import { AnswerListPage } from '@/pages/AnswerListPage'
import { AnswerDetailPage } from '@/pages/AnswerDetailPage'
import { TagListPage } from '@/pages/TagListPage'
import { TagDetailPage } from '@/pages/TagDetailPage'
import { RunConditionPage } from '@/pages/RunConditionPage'
import { RunAllConditionsPage } from '@/pages/RunAllConditionsPage'
import { RunAllResultsPage } from '@/pages/RunAllResultsPage'
import { RunResultPage } from '@/pages/RunResultPage'
import { RunResultDetailPage } from '@/pages/RunResultDetailPage'
import { HistoryPage } from '@/pages/HistoryPage'
import { HistoryComparePage } from '@/pages/HistoryComparePage'
import { HistoryDetailPage } from '@/pages/HistoryDetailPage'
import { HistoryResultDetailPage } from '@/pages/HistoryResultDetailPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { RunJudgePage } from '@/pages/RunJudgePage'
import { NotFoundPage } from '@/pages/NotFoundPage'

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/methodology" element={<MethodologyPage />} />
        <Route path="/docs/metrics" element={<MetricsReferencePage />} />
        <Route path="/questions" element={<QuestionListPage />} />
        <Route path="/questions/:id" element={<QuestionDetailPage />} />
        <Route path="/answers" element={<AnswerListPage />} />
        <Route path="/answers/:id" element={<AnswerDetailPage />} />
        <Route path="/tags" element={<TagListPage />} />
        <Route path="/tags/:name" element={<TagDetailPage />} />

        <Route path="/experiment/all" element={<RunAllConditionsPage />} />
        <Route path="/experiment/all/runs" element={<RunAllResultsPage />} />
        <Route path="/experiment/:condition" element={<RunConditionPage />} />
        <Route path="/experiment/:condition/runs/:run_id" element={<RunResultPage />} />
        <Route path="/experiment/:condition/runs/:run_id/q/:question_id" element={<RunResultDetailPage />} />

        <Route path="/evaluation/judge" element={<RunJudgePage />} />

        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/compare" element={<HistoryComparePage />} />
        <Route path="/history/:history_id" element={<HistoryDetailPage />} />
        <Route path="/history/:history_id/q/:question_id" element={<HistoryResultDetailPage />} />

        <Route path="/settings" element={<SettingsPage />} />

        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
