import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <div className="max-w-md mx-auto mt-16 text-center">
      <div className="text-2xl font-semibold text-text-primary mb-2">Page not found</div>
      <p className="text-sm text-text-secondary mb-4">The page you're looking for doesn't exist.</p>
      <Link to="/" className="text-sm text-primary hover:underline">
        ← Back to Home
      </Link>
    </div>
  )
}
