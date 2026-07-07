import { useState, useRef, useCallback, useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import cytoscape from 'cytoscape'
// @ts-expect-error cose-bilkent has no types
import coseBilkent from 'cytoscape-cose-bilkent'
import { api, type OutlineItem } from '../lib/api'
import ExportModal from '../components/ExportModal'
import ErrorBoundary from '../components/ErrorBoundary'

cytoscape.use(coseBilkent)

/* ── Chapter Tree ──────────────────────────────────── */

function ChapterTree({
  chapters,
  selected,
  onSelect,
  onDelete,
  onMove,
}: {
  chapters: OutlineItem[]
  selected: number | null
  onSelect: (n: number) => void
  onDelete: (n: number) => void
  onMove: (from: number, to: number) => void
}) {
  const [dragIdx, setDragIdx] = useState<number | null>(null)

  const statusBadge = (s: string) => {
    const map: Record<string, { cls: string; label: string }> = {
      pending: { cls: 'bg-gray-100 text-gray-600', label: '待写' },
      writing: { cls: 'bg-blue-100 text-blue-600', label: '写作中' },
      drafted: { cls: 'bg-yellow-100 text-yellow-600', label: '已生成' },
      approved: { cls: 'bg-green-100 text-green-600', label: '已审批' },
    }
    const info = map[s] || map.pending
    return (
      <span className={`text-[10px] px-1.5 py-0.5 rounded ${info.cls}`}>
        {info.label}
      </span>
    )
  }

  return (
    <div className="space-y-0.5">
      {chapters.map((ch, idx) => (
        <div
          key={ch.chapter_number}
          draggable
          onDragStart={() => setDragIdx(idx)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={() => {
            if (dragIdx != null && dragIdx !== idx) {
              onMove(dragIdx, idx)
            }
            setDragIdx(null)
          }}
          onClick={() => onSelect(ch.chapter_number)}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer text-sm
            ${selected === ch.chapter_number ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50 border border-transparent'}`}
        >
          <span className="text-gray-400 text-xs w-6">
            第{ch.chapter_number}章
          </span>
          <span className="flex-1 truncate text-gray-700">
            {ch.title || '(无标题)'}
          </span>
          {statusBadge(ch.status)}
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(ch.chapter_number) }}
            className="text-gray-300 hover:text-red-500 text-xs"
          >
            ✕
          </button>
        </div>
      ))}
      {chapters.length === 0 && (
        <p className="text-xs text-gray-400 text-center py-8">
          还没有章节，点击"AI 生成大纲"或"添加章节"
        </p>
      )}
    </div>
  )
}

/* ── Chapter Detail ────────────────────────────────── */

function ChapterDetail({
  chapter,
  onSave,
  projectId,
}: {
  chapter: OutlineItem | null
  onSave: (data: Partial<OutlineItem>) => void
  projectId: string
}) {
  const navigate = useNavigate()

  // Check if chapter has saved content
  const { data: chapterData } = useQuery({
    queryKey: ['chapter', projectId, chapter?.chapter_number],
    queryFn: () => api.getChapter(projectId, chapter!.chapter_number),
    enabled: !!chapter,
    retry: false,
  })
  const hasDraft = chapterData && !!(chapterData as Record<string, unknown>).draft_content as boolean
  const chStatus = chapter?.status || 'pending'

  if (!chapter) {
    return (
      <div className="text-center py-16 text-gray-400 text-sm">
        选择左侧章节查看详情
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">章节标题</label>
        <input
          value={chapter.title}
          onChange={(e) => onSave({ title: e.target.value })}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">章节概要</label>
        <textarea
          value={chapter.summary}
          onChange={(e) => onSave({ summary: e.target.value })}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          rows={6}
        />
      </div>
      <div className="flex gap-2">
        {hasDraft || chStatus === 'approved' || chStatus === 'drafted' ? (
          <button
            onClick={() => navigate(`/projects/${projectId}/chapters/${chapter.chapter_number}`)}
            className="bg-white border border-blue-600 text-blue-600 px-4 py-2 rounded-lg text-sm hover:bg-blue-50"
          >
            阅读章节
          </button>
        ) : null}
        <button
          onClick={() => navigate(`/projects/${projectId}/chapters/${chapter.chapter_number}`)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
        >
          {chStatus === 'pending' ? '开始写作' : hasDraft ? '重新生成' : '开始写作'}
        </button>
      </div>
    </div>
  )
}

/* ── Graph Components ──────────────────────────────── */

function GraphTimeline({ maxChapter, value, onChange }: {
  maxChapter: number
  value: number
  onChange: (v: number) => void
}) {
  if (maxChapter <= 1) return null
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="text-xs text-gray-500">前</span>
      <input
        type="range"
        min={1}
        max={maxChapter}
        value={value || maxChapter}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1"
      />
      <span className="text-xs text-gray-500">{value || maxChapter} 章</span>
    </div>
  )
}

function GraphCanvas({ projectId }: { projectId: string }) {
  const [untilChapter, setUntilChapter] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['graph', projectId, untilChapter],
    queryFn: () => api.getGraph(projectId, untilChapter || undefined),
  })

  // Initialize / update Cytoscape when data changes
  useEffect(() => {
    if (!data || !containerRef.current || data.nodes.length === 0) return

    // Destroy previous instance
    if (cyRef.current) {
      cyRef.current.destroy()
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: [
        ...data.nodes.map((n) => ({
          data: {
            id: n.id,
            label: n.label,
            type: n.type,
            color: n.color,
            importance: n.importance,
            hasConflict: n.has_conflict,
            firstChapter: n.first_chapter,
          },
          // Set shape class for per-shape styling
          classes: `shape-${n.shape}`,
        })),
        ...data.edges.map((e) => ({
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.label,
            relationshipType: e.relationship_type,
          },
        })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'shape': 'ellipse',
            'label': 'data(label)',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-halign': 'center',
            'color': '#374151',
            'width': 'mapData(importance, 1, 10, 24, 56)',
            'height': 'mapData(importance, 1, 10, 24, 56)',
            'border-width': 2,
            'border-color': '#fff',
          },
        },
        {
          selector: 'node.shape-diamond',
          style: { 'shape': 'diamond' },
        },
        {
          selector: 'node.shape-rectangle',
          style: { 'shape': 'rectangle' },
        },
        {
          selector: 'node.shape-hexagon',
          style: { 'shape': 'hexagon' },
        },
        {
          selector: 'node.shape-triangle',
          style: { 'shape': 'triangle' },
        },
        {
          selector: 'node.shape-ellipse',
          style: { 'shape': 'ellipse' },
        },
        {
          selector: 'node[hasConflict=true]',
          style: {
            'border-color': '#f56c6c',
            'border-width': 3,
            'border-style': 'dashed',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 1.5,
            'line-color': '#cbd5e1',
            'target-arrow-color': '#94a3b8',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '8px',
            'color': '#94a3b8',
            'text-rotation': 'autorotate',
          },
        },
      ],
      layout: {
        name: 'cose-bilkent',
        animate: false,
        gravity: 0.4,
        nodeRepulsion: 6000,
        idealEdgeLength: 100,
      } as cytoscape.LayoutOptions,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      minZoom: 0.3,
      maxZoom: 3,
    })

    // Double-click node → navigate to first chapter
    cy.on('dblclick', 'node', (evt) => {
      const node = evt.target
      const ch = node.data('firstChapter') as number
      if (ch > 0) {
        window.open(`/projects/${projectId}/chapters/${ch}`, '_blank')
      }
    })

    // Tooltip on hover
    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target
      const typeLabel = { character: '角色', location: '地点', item: '物品', organization: '组织', event: '事件' }[node.data('type') as string] || node.data('type')
      const conflict = node.data('hasConflict') ? ' ⚠冲突' : ''
      document.title = `${typeLabel}: ${node.data('label')}${conflict}`
    })

    cyRef.current = cy

    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [data, projectId])

  // Determine max chapter for timeline slider
  const maxChapter = untilChapter || 10

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <GraphTimeline
        maxChapter={maxChapter}
        value={untilChapter}
        onChange={setUntilChapter}
      />
      {isLoading ? (
        <p className="text-center py-16 text-gray-400 text-sm">加载图谱数据...</p>
      ) : data && data.nodes.length > 0 ? (
        <div className="relative">
          <div
            ref={containerRef}
            className="w-full h-[500px] border border-gray-100 rounded"
          />
          <div className="flex gap-3 mt-2 text-xs text-gray-400">
            <span>共 {data.nodes.length} 个实体，{data.edges.length} 条关系</span>
          </div>
          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-2 text-xs text-gray-500">
            {[
              { type: 'character', label: '角色', color: '#4a90d9' },
              { type: 'location', label: '地点', color: '#67c23a' },
              { type: 'item', label: '物品', color: '#e6a23c' },
              { type: 'organization', label: '组织', color: '#7b4dd3' },
              { type: 'event', label: '事件', color: '#f56c6c' },
            ].map((item) => (
              <span key={item.type} className="flex items-center gap-1">
                <span
                  className="inline-block w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                {item.label}
              </span>
            ))}
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-3 rounded-full border-2 border-dashed border-red-400" />
              冲突
            </span>
          </div>
        </div>
      ) : (
        <div className="text-center py-16 text-gray-400 text-sm">
          暂无图谱数据，写完章节后实体将自动展示
        </div>
      )}
    </div>
  )
}

/* ── Main Page ─────────────────────────────────────── */

export default function OutlinePage() {
  const { id } = useParams<{ id: string }>()
  const projectId = id!
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'outline' | 'graph'>('outline')
  const [showExport, setShowExport] = useState(false)
  const [selectedChapter, setSelectedChapter] = useState<number | null>(null)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const chaptersRef = useRef<OutlineItem[]>([])

  const { data: chapters = [], isLoading } = useQuery({
    queryKey: ['outline', projectId],
    queryFn: () => api.getOutline(projectId),
  })
  chaptersRef.current = chapters

  const { data: project } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
  })

  const saveMutation = useMutation({
    mutationFn: (chapters: Partial<OutlineItem>[]) => api.saveOutline(projectId, chapters),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['outline', projectId] }),
  })

  const generateMutation = useMutation({
    mutationFn: () => api.generateOutline(projectId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['outline', projectId] }),
  })

  const handleAddChapter = () => {
    const maxNum = chapters.length > 0
      ? Math.max(...chapters.map((c) => c.chapter_number))
      : 0
    const updated = [...chapters, {
      project_id: projectId,
      chapter_number: maxNum + 1,
      title: `第${maxNum + 1}章`,
      summary: '',
      status: 'pending' as const,
      sort_order: maxNum + 1,
    }]
    saveMutation.mutate(updated)
  }

  const handleDeleteChapter = (n: number) => {
    const updated = chapters.filter((c) => c.chapter_number !== n)
    saveMutation.mutate(updated)
    if (selectedChapter === n) setSelectedChapter(null)
  }

  const handleMove = (fromIdx: number, toIdx: number) => {
    const updated = [...chapters]
    const [moved] = updated.splice(fromIdx, 1)
    updated.splice(toIdx, 0, moved)
    const reordered = updated.map((ch, i) => ({ ...ch, sort_order: i + 1 }))
    saveMutation.mutate(reordered)
  }

  const handleSaveDetail = useCallback((data: Partial<OutlineItem>) => {
    if (!selectedChapter) return
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    const chNum = selectedChapter
    saveTimerRef.current = setTimeout(() => {
      const latest = chaptersRef.current.map((ch) =>
        ch.chapter_number === chNum ? { ...ch, ...data } : ch
      )
      saveMutation.mutate(latest)
    }, 400)
  }, [selectedChapter, saveMutation])

  const selected = chapters.find((c) => c.chapter_number === selectedChapter) || null

  return (
    <ErrorBoundary>
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <Link to="/" className="text-sm text-gray-500 hover:text-blue-600">
              &larr; 返回看板
            </Link>
            <h1 className="text-2xl font-bold text-gray-900 mt-1">
              {project?.title || '大纲管理'}
            </h1>
          </div>
          <Link
            to={`/projects/${projectId}/settings`}
            className="text-sm text-gray-500 hover:text-blue-600"
          >
            项目设置
          </Link>
        </div>

        {/* Tabs */}
        <div className="flex items-center justify-between border-b border-gray-200 mb-6">
          <div className="flex gap-0">
            {(['outline', 'graph'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
                  activeTab === tab
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'outline' ? '大纲' : '关系图谱'}
              </button>
            ))}
          </div>
          <button
            onClick={() => setShowExport(true)}
            className="text-sm text-gray-500 hover:text-blue-600 px-3 py-1.5 -mt-1"
          >
            导出
          </button>
        </div>

        {activeTab === 'outline' ? (
          <div className="flex gap-6">
            {/* Left: Chapter tree */}
            <div className="w-72 flex-shrink-0">
              <div className="flex gap-2 mb-3">
                <button
                  onClick={handleAddChapter}
                  className="flex-1 border border-gray-300 text-gray-700 rounded-lg py-1.5 text-xs hover:bg-gray-50"
                >
                  添加章节
                </button>
                <button
                  onClick={() => generateMutation.mutate()}
                  disabled={generateMutation.isPending}
                  className="flex-1 bg-blue-600 text-white rounded-lg py-1.5 text-xs hover:bg-blue-700 disabled:opacity-50"
                >
                  {generateMutation.isPending ? '生成中...' : 'AI 生成大纲'}
                </button>
              </div>
              {isLoading ? (
                <p className="text-xs text-gray-400 text-center py-8">加载中...</p>
              ) : (
                <ChapterTree
                  chapters={chapters}
                  selected={selectedChapter}
                  onSelect={setSelectedChapter}
                  onDelete={handleDeleteChapter}
                  onMove={handleMove}
                />
              )}
            </div>

            {/* Right: Chapter detail */}
            <div className="flex-1 bg-white rounded-lg border border-gray-200 p-6">
              <ChapterDetail
                chapter={selected}
                onSave={handleSaveDetail}
                projectId={projectId}
              />
            </div>
          </div>
        ) : (
          <GraphCanvas projectId={projectId} />
        )}
      </div>

      {showExport && (
        <ExportModal
          projectId={projectId}
          onClose={() => setShowExport(false)}
        />
      )}
    </ErrorBoundary>
  )
}
