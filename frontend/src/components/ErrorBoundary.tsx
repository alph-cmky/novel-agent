import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800 text-sm">
          <p className="font-medium">Something went wrong</p>
          <p className="mt-1 text-red-600">{this.state.error?.message}</p>
        </div>
      )
    }
    return this.props.children
  }
}
