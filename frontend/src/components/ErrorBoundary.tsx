import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Catches render-time errors in any page so a single bad state (unexpected
 * API shape, null-ref, etc.) shows a recoverable message instead of a blank
 * white screen -- important during a live defense demo where a silent crash
 * would otherwise be hard to explain or recover from without a hard reload.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Unhandled error in page:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="max-w-lg mx-auto mt-16 border border-danger/40 bg-white rounded-lg p-6 text-center">
          <div className="text-sm font-medium text-danger mb-2">Something went wrong on this page.</div>
          <div className="text-xs text-text-secondary mb-4 font-mono">{this.state.error.message}</div>
          <button
            onClick={() => this.setState({ error: null })}
            className="px-3 py-1.5 text-sm bg-primary text-white rounded-md hover:bg-primary-hover"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
