import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loading } from './Spinner'

const FORMAT_OPTIONS = [
  { value: 'md', label: 'Markdown', ext: '.md' },
  { value: 'txt', label: '纯文本', ext: '.txt' },
] as const

interface ExportData {
  title: string
  content: string
  format: string
  chapter_count: number
}

export default function ExportModal({
  projectId,
  onClose,
}: {
  projectId: string
  onClose: () => void
}) {
  const [fmt, setFmt] = useState<'md' | 'txt'>('md')

  const { data, isLoading, error } = useQuery<ExportData>({
    queryKey: ['export-preview', projectId, fmt],
    queryFn: () =>
      fetch(`/api/projects/${projectId}/export?format=${fmt}&preview=true`).then(
        (r) => {
          if (!r.ok) throw new Error(r.statusText)
          return r.json()
        }
      ),
    staleTime: 0,
  })

  const handleDownload = () => {
    const a = document.createElement('a')
    a.href = `/api/projects/${projectId}/export?format=${fmt}`
    a.download = ''
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-bold text-gray-900">导出全文</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-lg leading-none"
          >
            ✕
          </button>
        </div>

        {/* Format tabs */}
        <div className="flex gap-0 border-b border-gray-200 px-6">
          {FORMAT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setFmt(opt.value)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
                fmt === opt.value
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Content preview */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <Loading label="加载中..." className="py-16" />
          ) : error ? (
            <p className="text-center py-16 text-red-500 text-sm">
              加载失败: {(error as Error).message}
            </p>
          ) : data ? (
            <div>
              <div className="flex items-center gap-3 mb-3 text-xs text-gray-500">
                <span>共 {data.chapter_count} 章</span>
                <span>
                  {data.content.length.toLocaleString()} 字符
                </span>
                <span className="bg-gray-100 px-1.5 py-0.5 rounded">
                  .{fmt}
                </span>
              </div>
              <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono bg-gray-50 rounded-lg p-4 max-h-[50vh] overflow-y-auto leading-relaxed">
                {data.content}
              </pre>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex gap-2 px-6 py-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="flex-1 border border-gray-300 text-gray-700 rounded-lg py-2 text-sm hover:bg-gray-50"
          >
            取消
          </button>
          <button
            onClick={handleDownload}
            disabled={!data}
            className="flex-1 bg-blue-600 text-white rounded-lg py-2 text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            下载 .{fmt}
          </button>
        </div>
      </div>
    </div>
  )
}
