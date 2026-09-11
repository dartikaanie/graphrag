import { Route, Routes } from 'react-router-dom'
import { DashboardLayout } from '@/layouts/DashboardLayout'
import { HomePage } from '@/pages/HomePage'
import { QuestionListPage } from '@/pages/QuestionListPage'
import { QuestionDetailPage } from '@/pages/QuestionDetailPage'
import { AnswerListPage } from '@/pages/AnswerListPage'
import { AnswerDetailPage } from '@/pages/AnswerDetailPage'
import { TagListPage } from '@/pages/TagListPage'
import { TagDetailPage } from '@/pages/TagDetailPage'
import { ComingSoonPage } from '@/pages/ComingSoonPage'

export default function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/questions" element={<QuestionListPage />} />
        <Route path="/questions/:id" element={<QuestionDetailPage />} />
        <Route path="/answers" element={<AnswerListPage />} />
        <Route path="/answers/:id" element={<AnswerDetailPage />} />
        <Route path="/tags" element={<TagListPage />} />
        <Route path="/tags/:name" element={<TagDetailPage />} />

        <Route path="/experiment/a" element={<ComingSoonPage title="Condition A — Pure LLM" phase="Phase 4/5" />} />
        <Route path="/experiment/b" element={<ComingSoonPage title="Condition B — LLM + RAG" phase="Phase 4/5" />} />
        <Route
          path="/experiment/c"
          element={<ComingSoonPage title="Condition C — LLM + GraphRAG" phase="Phase 4/5" />}
        />
        <Route path="/history" element={<ComingSoonPage title="History" phase="Phase 6" />} />
        <Route path="/settings" element={<ComingSoonPage title="Settings" phase="Phase 7" />} />
      </Route>
    </Routes>
  )
}
