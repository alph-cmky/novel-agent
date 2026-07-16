type Size = 'sm' | 'md' | 'lg'

const DIMENSIONS: Record<Size, string> = {
  sm: 'h-4 w-4',
  md: 'h-6 w-6',
  lg: 'h-9 w-9',
}

export function Spinner({
  size = 'md',
  className = '',
}: {
  size?: Size
  className?: string
}) {
  return (
    <svg
      className={`animate-spin ${DIMENSIONS[size]} ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        className="opacity-20"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

export function Loading({
  label = '加载中...',
  size = 'lg',
  className = '',
}: {
  label?: string
  size?: Size
  className?: string
}) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-3 ${className}`}
    >
      <Spinner size={size} className="text-blue-600" />
      <p className="text-sm text-gray-400">{label}</p>
    </div>
  )
}
