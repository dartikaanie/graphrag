import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  MessageSquare,
  MessageSquareText,
  Tag,
  FlaskConical,
  Layers,
  History,
  Settings,
  BookOpen,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { ErrorBoundary } from '@/components/ErrorBoundary'

interface NavItem {
  to: string
  label: string
  icon: ReactNode
  end?: boolean
}

const mainNav: NavItem[] = [
  { to: '/', label: 'Home', icon: <LayoutDashboard className="h-4 w-4" />, end: true },
  { to: '/methodology', label: 'Methodology', icon: <BookOpen className="h-4 w-4" /> },
  { to: '/docs/metrics', label: 'Metrik Evaluasi', icon: <BookOpen className="h-4 w-4" /> },
]

const masterDataNav: NavItem[] = [
  { to: '/questions', label: 'Questions', icon: <MessageSquare className="h-4 w-4" /> },
  { to: '/answers', label: 'Answers', icon: <MessageSquareText className="h-4 w-4" /> },
  { to: '/tags', label: 'Tags', icon: <Tag className="h-4 w-4" /> },
]

const experimentNav: NavItem[] = [
  { to: '/experiment/all', label: 'Run All (A + B + C + D)', icon: <Layers className="h-4 w-4" /> },
  { to: '/experiment/a', label: 'A — Pure LLM', icon: <FlaskConical className="h-4 w-4" /> },
  { to: '/experiment/b', label: 'B — LLM + RAG', icon: <FlaskConical className="h-4 w-4" /> },
  { to: '/experiment/c', label: 'C — LLM + GraphRAG', icon: <FlaskConical className="h-4 w-4" /> },
  { to: '/experiment/d', label: 'D — Dual-Level Retrieval (LightRAG-adapted)', icon: <FlaskConical className="h-4 w-4" /> },
  { to: '/evaluation/judge', label: 'LLM-as-Judge Evaluation', icon: <FlaskConical className="h-4 w-4" /> },
]

const footerNav: NavItem[] = [
  { to: '/history', label: 'History', icon: <History className="h-4 w-4" /> },
  { to: '/settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
]

function NavGroup({ title, items }: { title?: string; items: NavItem[] }) {
  return (
    <div className="mb-4">
      {title && <div className="px-3 mb-1 text-xs font-medium text-text-muted uppercase tracking-wide">{title}</div>}
      <div className="flex flex-col gap-0.5">
        {items.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-1.5 rounded-md text-sm ${
                isActive
                  ? 'bg-primary-soft text-primary font-medium'
                  : 'text-text-secondary hover:bg-bg hover:text-text-primary'
              }`
            }
          >
            {item.icon}
            <span className="truncate">{item.label}</span>
          </NavLink>
        ))}
      </div>
    </div>
  )
}

export function DashboardLayout() {
  const location = useLocation()
  return (
    <div className="flex min-h-screen">
      <aside className="w-60 shrink-0 border-r border-border bg-surface px-2 py-4 flex flex-col">
        <div className="px-3 mb-6">
          <div className="text-sm font-semibold text-text-primary">GraphRAG Dashboard</div>
          <div className="text-xs text-text-muted">Thesis research console</div>
        </div>
        <NavGroup items={mainNav} />
        <NavGroup title="Master Data" items={masterDataNav} />
        <NavGroup title="Experiments" items={experimentNav} />
        <div className="mt-auto">
          <NavGroup items={footerNav} />
        </div>
      </aside>
      <main className="flex-1 min-w-0 px-8 py-6 overflow-x-auto">
        {/* key={pathname} remounts the boundary on every navigation, so an
            error on one page never persists onto the next page visited. */}
        <ErrorBoundary key={location.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  )
}
