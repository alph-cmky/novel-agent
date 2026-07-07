import { useState } from 'react'

interface ContextData {
  chapterOutline?: string
  characterContext?: string
  worldContext?: string
  recentSummary?: string
  storyLength?: string
  targetWords?: number
}

const STORY_LENGTH_LABELS: Record<string, string> = {
  short: '短篇',
  novella: '中篇',
  long: '长篇',
}

function CollapsibleCard({ title, children, defaultOpen = true }: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-gray-200 rounded-lg bg-white">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-3 py-2 text-left text-xs font-medium text-gray-600 hover:bg-gray-50 flex justify-between items-center"
      >
        {title}
        <span className="text-gray-400">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="px-3 pb-3 text-xs text-gray-700 whitespace-pre-wrap leading-relaxed max-h-[200px] overflow-y-auto">{children}</div>}
    </div>
  )
}

export default function ContextPanel({ data }: { data: ContextData }) {
  const hasContent = data.chapterOutline || data.characterContext || data.worldContext || data.recentSummary

  return (
    <div className="space-y-2">
      {/* Writing config */}
      {(data.storyLength || data.targetWords) && (
        <CollapsibleCard title="写作设置" defaultOpen={false}>
          <div className="space-y-1">
            {data.storyLength && (
              <div className="flex justify-between">
                <span className="text-gray-500">篇幅</span>
                <span>{STORY_LENGTH_LABELS[data.storyLength] || data.storyLength}</span>
              </div>
            )}
            {data.targetWords && (
              <div className="flex justify-between">
                <span className="text-gray-500">目标字数</span>
                <span>{data.targetWords.toLocaleString()} 字/章</span>
              </div>
            )}
          </div>
        </CollapsibleCard>
      )}

      {data.chapterOutline && (
        <CollapsibleCard title="本章大纲">
          {data.chapterOutline}
        </CollapsibleCard>
      )}
      {data.characterContext && (
        <CollapsibleCard title="相关角色" defaultOpen={false}>
          {data.characterContext}
        </CollapsibleCard>
      )}
      {data.worldContext && (
        <CollapsibleCard title="世界观设定" defaultOpen={false}>
          {data.worldContext}
        </CollapsibleCard>
      )}
      {data.recentSummary && (
        <CollapsibleCard title="前文提要" defaultOpen={false}>
          {data.recentSummary}
        </CollapsibleCard>
      )}
      {!hasContent && (
        <p className="text-xs text-gray-400 p-4 text-center">暂无上下文信息</p>
      )}
    </div>
  )
}
