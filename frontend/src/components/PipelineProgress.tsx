interface Step {
  node: string
  label: string
  status: 'idle' | 'running' | 'done' | 'error'
  score?: number
  detail?: string
}

export default function PipelineProgress({ steps }: { steps: Step[] }) {
  if (steps.length === 0) {
    return <p className="text-xs text-gray-400 py-2">等待开始...</p>
  }

  return (
    <div className="space-y-1">
      {steps.map((step, i) => {
        const status = step.status
        const isLast = i === steps.length - 1

        return (
          <div key={step.node} className="flex items-start gap-2">
            <div className="flex flex-col items-center">
              <div
                className={`w-3 h-3 rounded-full border-2 ${
                  status === 'done' ? 'bg-green-500 border-green-500' :
                  status === 'running' ? 'bg-blue-500 border-blue-500 animate-pulse' :
                  status === 'error' ? 'bg-red-500 border-red-500' :
                  'bg-gray-200 border-gray-300'
                }`}
              />
              {!isLast && <div className="w-0.5 h-6 bg-gray-200" />}
            </div>
            <div className="flex-1 pb-1">
              <div className="flex items-center gap-2">
                <span
                  className={`text-xs font-medium ${
                    status === 'done' ? 'text-green-700' :
                    status === 'running' ? 'text-blue-700' :
                    status === 'error' ? 'text-red-700' :
                    'text-gray-400'
                  }`}
                >
                  {step.label || step.node}
                </span>
                {step.score != null && (
                  <span className="text-xs text-gray-500">{step.score}/100</span>
                )}
              </div>
              {step.detail && (
                <p className="text-xs text-gray-400 mt-0.5">{step.detail}</p>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
