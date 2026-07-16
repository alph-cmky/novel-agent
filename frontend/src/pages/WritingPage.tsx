import { useState, useRef, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import ErrorBoundary from '../components/ErrorBoundary'
import ContextPanel from '../components/ContextPanel'
import PipelineProgress from '../components/PipelineProgress'
import ScoreBadge from '../components/ScoreBadge'
import { Spinner } from '../components/Spinner'

interface PipelineStep {
  node: string
  label: string
  status: 'idle' | 'running' | 'done' | 'error'
  score?: number
  detail?: string
}

interface ReviewData {
  draft_preview?: string
  draft_full?: string
  editor_score: number
  continuity_score: number
  editor_issues?: Array<{ category: string; description: string; severity: string }>
  continuity_issues?: Array<{ category: string; description: string }>
  wb_new_entities: number
  wb_conflicts: number
  retry_count: number
}

type ViewMode = 'read' | 'edit'

function wordCount(text: string): number {
  // Chinese character count + word count for English
  const chinese = (text.match(/[一-鿿]/g) || []).length
  const english = (text.match(/[a-zA-Z]+/g) || []).length
  return chinese + english
}

export default function WritingPage() {
  const { id, n } = useParams<{ id: string; n: string }>()
  const projectId = id!
  const chapterNumber = Number(n)
  const queryClient = useQueryClient()

  const [draftContent, setDraftContent] = useState('')
  const [editedContent, setEditedContent] = useState('')
  const [viewMode, setViewMode] = useState<ViewMode>('read')
  const [isStreaming, setIsStreaming] = useState(false)
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([])
  const [reviewData, setReviewData] = useState<ReviewData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [rejectComments, setRejectComments] = useState('')
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [showRejectBox, setShowRejectBox] = useState(false)
  const [notice, setNotice] = useState<{ type: 'approved' | 'saved_draft'; message: string } | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { data: chapter } = useQuery({
    queryKey: ['chapter', projectId, chapterNumber],
    queryFn: () => api.getChapter(projectId, chapterNumber),
    retry: false,
  })

  const { data: project } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
  })

  const { data: outline } = useQuery({
    queryKey: ['outline', projectId],
    queryFn: () => api.getOutline(projectId),
  })

  const hasNextChapter = !!outline?.some((o) => o.chapter_number === chapterNumber + 1)

  // Load existing chapter content
  useEffect(() => {
    if (chapter && !draftContent) {
      const content = (chapter as Record<string, unknown>).draft_content as string || ''
      if (content) {
        setDraftContent(content)
        setEditedContent(content)
      }
    }
  }, [chapter, draftContent])

  // Scroll to bottom when content streams in
  useEffect(() => {
    if (isStreaming && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [draftContent, isStreaming])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  const startWriting = useCallback(() => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsStreaming(true)
    setDraftContent('')
    setEditedContent('')
    setViewMode('read')
    setPipelineSteps([])
    setReviewData(null)
    setError(null)
    setNotice(null)
    setShowRejectBox(false)

    const url = api.writeChapterUrl(projectId, chapterNumber)
    fetch(url, { method: 'POST', signal: controller.signal }).then(async (response) => {
      if (!response.ok || !response.body) {
        setError(`HTTP ${response.status}: ${response.statusText}`)
        setIsStreaming(false)
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (controller.signal.aborted) return

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (controller.signal.aborted) return
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            const raw = line.slice(6)
            try {
              const data = JSON.parse(raw)

              if (currentEvent === 'chunk' || (typeof data === 'string')) {
                setDraftContent((prev) => prev + (typeof data === 'string' ? data : ''))
              } else if (currentEvent === 'review_required' || data.type === 'review_required') {
                setReviewData(data)
                if (data.draft_full) setDraftContent(data.draft_full)
                setPipelineSteps((prev) => [
                  ...prev,
                  { node: 'human_review', label: '人工审批', status: 'running' },
                ])
                setIsStreaming(false)
              } else if (currentEvent === 'done') {
                if (data.chapter_content) {
                  setDraftContent(data.chapter_content)
                }
                setPipelineSteps((prev) =>
                  prev.map((s) => s.status === 'running' ? { ...s, status: 'done' } : s)
                )
                setIsStreaming(false)
                queryClient.invalidateQueries({ queryKey: ['chapter', projectId, chapterNumber] })
              } else if (currentEvent === 'progress') {
                setPipelineSteps((prev) => {
                  const existing = prev.find((s) => s.node === data.node)
                  if (existing) {
                    return prev.map((s) =>
                      s.node === data.node ? { ...s, status: data.status, score: data.score, detail: data.detail } : s
                    )
                  }
                  return [...prev, {
                    node: data.node,
                    label: data.label || data.node,
                    status: data.status,
                    score: data.score,
                    detail: data.detail,
                  }]
                })
              } else if (currentEvent === 'error') {
                setError(data.message)
                setPipelineSteps((prev) => {
                  const node = data.node && data.node !== 'unknown' ? (data.node as string) : null
                  const next = prev.map((s) =>
                    s.status === 'running' ? { ...s, status: 'error' as const } : s
                  )
                  if (node && !next.some((s) => s.node === node)) {
                    return [...next, { node, label: node, status: 'error' as const }]
                  }
                  return next
                })
                setIsStreaming(false)
              }
            } catch {
              setDraftContent((prev) => prev + raw)
            }
          }
        }
      }
      setIsStreaming(false)
    }).catch((err) => {
      if (err.name === 'AbortError') return
      setError(err.message)
      setIsStreaming(false)
    })
  }, [projectId, chapterNumber, queryClient])

  const handleApprove = useCallback(() => {
    fetch(`/api/projects/${projectId}/chapters/${chapterNumber}/approve`, { method: 'POST' })
      .then(async (res) => {
        if (!res.ok || !res.body) return
        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let currentEvent = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                if (currentEvent === 'done') {
                  setPipelineSteps((prev) =>
                    prev.map((s) => ({ ...s, status: 'done' as const }))
                  )
                  setReviewData(null)
                  setShowRejectBox(false)
                  setNotice({ type: 'approved', message: '本章已批准并保存' })
                  queryClient.invalidateQueries({ queryKey: ['chapter', projectId, chapterNumber] })
                  queryClient.invalidateQueries({ queryKey: ['outline', projectId] })
                  queryClient.invalidateQueries({ queryKey: ['projects'] })
                } else if (currentEvent === 'progress') {
                  setPipelineSteps((prev) => {
                    const existing = prev.find((s) => s.node === data.node)
                    if (existing) {
                      return prev.map((s) =>
                        s.node === data.node ? { ...s, status: data.status, score: data.score } : s
                      )
                    }
                    return [...prev, { node: data.node, label: data.label || data.node, status: data.status, score: data.score }]
                  })
                }
              } catch {}
            }
          }
        }
      })
  }, [projectId, chapterNumber, queryClient])

  const handleReject = useCallback(() => {
    const url = `/api/projects/${projectId}/chapters/${chapterNumber}/reject`
    setIsStreaming(true)
    setDraftContent('')
    setEditedContent('')
    setViewMode('read')
    setPipelineSteps([])
    setReviewData(null)
    setError(null)
    setNotice(null)
    setShowRejectBox(false)

    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comments: rejectComments }),
    }).then(async (res) => {
      if (!res.ok || !res.body) {
        setError(`HTTP ${res.status}`)
        setIsStreaming(false)
        return
      }
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (currentEvent === 'chunk' || typeof data === 'string') {
                setDraftContent((prev) => prev + (typeof data === 'string' ? data : ''))
              } else if (currentEvent === 'review_required') {
                setReviewData(data)
                if (data.draft_full) setDraftContent(data.draft_full)
                setIsStreaming(false)
              } else if (currentEvent === 'done') {
                if (data.chapter_content) {
                  setDraftContent(data.chapter_content)
                }
                setIsStreaming(false)
                queryClient.invalidateQueries({ queryKey: ['chapter', projectId, chapterNumber] })
              } else if (currentEvent === 'progress') {
                setPipelineSteps((prev) => {
                  const existing = prev.find((s) => s.node === data.node)
                  if (existing) {
                    return prev.map((s) =>
                      s.node === data.node ? { ...s, status: data.status, score: data.score, detail: data.detail } : s
                    )
                  }
                  return [...prev, { node: data.node, label: data.label || data.node, status: data.status, score: data.score, detail: data.detail }]
                })
              } else if (currentEvent === 'error') {
                setError(data.message)
                setPipelineSteps((prev) => {
                  const node = data.node && data.node !== 'unknown' ? (data.node as string) : null
                  const next = prev.map((s) =>
                    s.status === 'running' ? { ...s, status: 'error' as const } : s
                  )
                  if (node && !next.some((s) => s.node === node)) {
                    return [...next, { node, label: node, status: 'error' as const }]
                  }
                  return next
                })
                setIsStreaming(false)
              }
            } catch {
              setDraftContent((prev) => prev + line.slice(6))
            }
          }
        }
      }
      setIsStreaming(false)
    }).catch((err) => {
      setError(err.message)
      setIsStreaming(false)
    })
  }, [projectId, chapterNumber, queryClient, rejectComments])

  // Switch to edit mode: copy current draft to editor
  const enterEditMode = useCallback(() => {
    setEditedContent(draftContent)
    setViewMode('edit')
    setSaveState('idle')
    setNotice(null)
  }, [draftContent])

  // Cancel editing and revert to read mode
  const cancelEdit = useCallback(() => {
    setEditedContent(draftContent)
    setViewMode('read')
    setSaveState('idle')
  }, [draftContent])

  // Unified "save draft" — saves whatever the user is currently looking at:
  // edit mode → editedContent (their manual changes), read mode → the AI's draftContent.
  const handleSaveDraft = useCallback(async () => {
    const content = viewMode === 'edit' ? editedContent : draftContent
    if (!content.trim()) return
    setSaveState('saving')
    try {
      await api.saveDraft(projectId, chapterNumber, content)
      setDraftContent(content)
      setSaveState('saved')
      setViewMode('read')
      if (reviewData) {
        setReviewData(null)
        setShowRejectBox(false)
        setPipelineSteps((prev) => prev.map((s) => ({ ...s, status: 'done' as const })))
        setNotice({ type: 'saved_draft', message: '草稿已保存，可稍后继续编辑或重新生成本章' })
      } else {
        setNotice(null)
      }
      queryClient.invalidateQueries({ queryKey: ['chapter', projectId, chapterNumber] })
      queryClient.invalidateQueries({ queryKey: ['outline', projectId] })
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    } catch (err) {
      setSaveState('error')
      setError((err as Error).message)
    }
  }, [projectId, chapterNumber, viewMode, editedContent, draftContent, reviewData, queryClient])

  const hasExistingChapter = chapter && (chapter as Record<string, unknown>).draft_content as string
  const showContent = draftContent || hasExistingChapter
  const charCount = draftContent.length
  const wc = wordCount(draftContent)

  return (
    <ErrorBoundary>
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <Link to={`/projects/${projectId}`} className="text-sm text-gray-500 hover:text-blue-600">
              &larr; 返回大纲
            </Link>
            <h1 className="text-2xl font-bold text-gray-900 mt-1">
              第 {chapterNumber} 章写作
            </h1>
          </div>
          {/* Export link */}
          <a
            href={`/api/projects/${projectId}/export`}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-gray-400 hover:text-blue-600 underline"
          >
            导出全文
          </a>
        </div>

        {/* Three-column layout */}
        <div className="flex gap-4">
          {/* Left: Context Panel */}
          <div className="w-[250px] flex-shrink-0">
            <ContextPanel
              data={{
                chapterOutline: chapter
                  ? (chapter as Record<string, unknown>).chapter_outline as string || `第${chapterNumber}章`
                  : `第${chapterNumber}章`,
                recentSummary: chapter
                  ? (chapter as Record<string, unknown>).recent_summary as string || undefined
                  : undefined,
                characterContext: chapter
                  ? (chapter as Record<string, unknown>).character_context as string || undefined
                  : undefined,
                worldContext: chapter
                  ? (chapter as Record<string, unknown>).world_context as string || undefined
                  : undefined,
                storyLength: project?.story_length,
                targetWords: project?.target_chapter_words,
              }}
            />
          </div>

          {/* Center: Writing Area */}
          <div className="flex-1 min-w-0">
            {/* View mode tabs - only show when content exists and not streaming */}
            {showContent && !isStreaming && (
              <div className="flex items-center justify-between mb-2">
                <div className="flex bg-gray-100 rounded-lg p-0.5">
                  <button
                    onClick={() => { setViewMode('read'); setSaveState('idle') }}
                    className={`px-3 py-1 text-xs rounded-md transition-colors ${
                      viewMode === 'read' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    阅读
                  </button>
                  <button
                    onClick={enterEditMode}
                    className={`px-3 py-1 text-xs rounded-md transition-colors ${
                      viewMode === 'edit' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    编辑
                  </button>
                </div>
                {/* Stats */}
                <div className="flex gap-3 text-xs text-gray-400">
                  <span>{charCount.toLocaleString()} 字</span>
                  <span>~{wc.toLocaleString()} 词</span>
                </div>
              </div>
            )}

            {/* Read Mode */}
            {(viewMode === 'read' || isStreaming) && (
              <div
                ref={contentRef}
                className="bg-white rounded-lg border border-gray-200 p-6 min-h-[500px] max-h-[70vh] overflow-y-auto"
              >
                {showContent ? (
                  <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap leading-relaxed font-serif text-[15px]">
                    {draftContent}
                    {hasExistingChapter && !draftContent && (chapter as Record<string, unknown>).draft_content as string}
                    {isStreaming && <span className="inline-block w-2 h-4 bg-blue-600 animate-pulse ml-0.5 align-middle" />}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full min-h-[400px]">
                    {isStreaming ? (
                      <div className="text-center">
                        <div className="flex gap-1 justify-center mb-2">
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.1s]" />
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                        </div>
                        <p className="text-gray-400 text-sm">AI 正在创作...</p>
                      </div>
                    ) : error ? (
                      <div className="text-center">
                        <p className="text-red-500 mb-2">{error}</p>
                        <button onClick={startWriting} className="text-blue-600 text-sm hover:underline">
                          重试
                        </button>
                      </div>
                    ) : (
                      <div className="text-center">
                        {hasExistingChapter ? (
                          <div className="flex flex-col items-center gap-3 mb-4">
                            <Spinner size="md" className="text-blue-600" />
                            <p className="text-gray-400 text-sm">加载中...</p>
                          </div>
                        ) : (
                          <p className="text-gray-400 mb-4">点击按钮开始生成草稿</p>
                        )}
                        {!hasExistingChapter && (
                          <button
                            onClick={startWriting}
                            className="bg-blue-600 text-white px-6 py-2 rounded-lg text-sm hover:bg-blue-700 transition-colors"
                          >
                            生成草稿
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Edit Mode */}
            {viewMode === 'edit' && !isStreaming && (
              <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                <textarea
                  ref={textareaRef}
                  value={editedContent}
                  onChange={(e) => { setEditedContent(e.target.value); setSaveState('idle') }}
                  className="w-full min-h-[500px] max-h-[70vh] p-6 text-gray-800 text-[15px] leading-relaxed font-serif resize-y focus:outline-none"
                  placeholder="在此编辑章节内容..."
                />
                <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200">
                  <div className="flex gap-2 text-xs text-gray-400">
                    <span>{editedContent.length.toLocaleString()} 字</span>
                    <span>~{wordCount(editedContent).toLocaleString()} 词</span>
                    {draftContent !== editedContent && (
                      <span className="text-yellow-600">未保存</span>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={cancelEdit}
                      className="px-3 py-1 text-xs text-gray-500 hover:text-gray-700 border border-gray-300 rounded"
                    >
                      取消
                    </button>
                    <button
                      onClick={handleSaveDraft}
                      disabled={saveState === 'saving' || !editedContent.trim()}
                      className="px-4 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors"
                    >
                      {saveState === 'saving' ? (
                        <span className="flex items-center justify-center gap-1.5">
                          <Spinner size="sm" className="text-white" />
                          保存中...
                        </span>
                      ) : saveState === 'saved' ? '已保存' : '保存草稿'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Action bar: regenerate for existing chapters */}
            {hasExistingChapter && !draftContent && !isStreaming && !reviewData && (
              <button
                onClick={startWriting}
                className="mt-2 text-sm text-blue-600 hover:text-blue-700 hover:underline"
              >
                重新生成本章
              </button>
            )}

            {/* Success notice: post-approve / post-save-draft feedback + navigation */}
            {notice && !isStreaming && !reviewData && (
              <div className="mt-3 bg-green-50 border border-green-200 rounded-lg p-3 flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-green-800">
                  <span className="text-green-600">✓</span>
                  <span>{notice.message}</span>
                </div>
                <div className="flex gap-3 text-sm">
                  {notice.type === 'approved' && hasNextChapter && (
                    <Link
                      to={`/projects/${projectId}/chapters/${chapterNumber + 1}`}
                      className="text-blue-600 hover:underline"
                    >
                      写下一章 →
                    </Link>
                  )}
                  <Link to={`/projects/${projectId}`} className="text-gray-500 hover:underline">
                    返回大纲
                  </Link>
                </div>
              </div>
            )}

            {/* Approval bar */}
            {reviewData && (
              <div className="mt-3 bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-gray-900 text-sm">审阅本章</h3>
                  <div className="flex gap-2">
                    <ScoreBadge score={reviewData.editor_score} />
                    <ScoreBadge score={reviewData.continuity_score} />
                  </div>
                </div>

                <p className="mb-3 text-xs text-gray-400">
                  满意则批准，暂缓可保存草稿，需修改可切换到上方「编辑」标签
                </p>

                {reviewData.retry_count > 0 && (
                  <p className="text-xs text-yellow-600 mb-2">
                    第 {reviewData.retry_count + 1} 次尝试 {reviewData.retry_count >= 2 ? '(最后一次)' : ''}
                  </p>
                )}

                <div className="flex gap-2">
                  <button
                    onClick={handleApprove}
                    className="flex-1 bg-green-600 text-white rounded-lg py-2 text-sm hover:bg-green-700 transition-colors"
                  >
                    批准并保存
                  </button>
                  <button
                    onClick={handleSaveDraft}
                    disabled={saveState === 'saving'}
                    className="flex-1 border border-blue-300 text-blue-700 rounded-lg py-2 text-sm hover:bg-blue-50 transition-colors disabled:opacity-50"
                  >
                    {saveState === 'saving' ? (
                      <span className="flex items-center justify-center gap-1.5">
                        <Spinner size="sm" className="text-blue-700" />
                        保存中…
                      </span>
                    ) : '保存草稿'}
                  </button>
                  <button
                    onClick={() => setShowRejectBox((v) => !v)}
                    className="flex-1 border border-red-300 text-red-600 rounded-lg py-2 text-sm hover:bg-red-50 transition-colors"
                  >
                    拒绝并重写
                  </button>
                </div>

                {showRejectBox && (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <textarea
                      value={rejectComments}
                      onChange={(e) => setRejectComments(e.target.value)}
                      placeholder="拒绝理由（可选），将指导 AI 重新创作..."
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-red-400"
                      rows={2}
                    />
                    <button
                      onClick={handleReject}
                      className="mt-2 w-full bg-red-600 text-white rounded-lg py-2 text-sm hover:bg-red-700 transition-colors"
                    >
                      确认拒绝并重新生成
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right: Review Panel */}
          <div className="w-[300px] flex-shrink-0">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">流水线进度</h3>
              <PipelineProgress steps={pipelineSteps} />
              {reviewData && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  {reviewData.editor_issues && reviewData.editor_issues.length > 0 && (
                    <div className="mb-3">
                      <p className="text-xs font-medium text-gray-600 mb-1">编辑问题</p>
                      {reviewData.editor_issues.slice(0, 5).map((issue, i) => (
                        <p key={i} className="text-xs text-gray-500 mt-0.5">
                          [{issue.severity}] {issue.category}: {issue.description}
                        </p>
                      ))}
                    </div>
                  )}
                  {reviewData.continuity_issues && reviewData.continuity_issues.length > 0 && (
                    <div className="mb-3">
                      <p className="text-xs font-medium text-gray-600 mb-1">一致性问题</p>
                      {reviewData.continuity_issues.slice(0, 5).map((issue, i) => (
                        <p key={i} className="text-xs text-gray-500 mt-0.5">
                          [{issue.category}] {issue.description}
                        </p>
                      ))}
                    </div>
                  )}
                  {reviewData.wb_new_entities > 0 && (
                    <p className="text-xs text-gray-500">
                      新增实体: {reviewData.wb_new_entities}
                      {reviewData.wb_conflicts > 0 && ` | 冲突: ${reviewData.wb_conflicts}`}
                    </p>
                  )}
                </div>
              )}

              {/* Show saved chapter info */}
              {hasExistingChapter && !reviewData && !isStreaming && (
                <div className="mt-4 pt-4 border-t border-gray-100">
                  <p className="text-xs text-gray-500">
                    状态: {(chapter as Record<string, unknown>)?.status as string || '未知'}
                  </p>
                  {chapter && !!(chapter as Record<string, unknown>).editor_report && (
                    <div className="mt-2">
                      <p className="text-xs font-medium text-gray-600 mb-1">评分</p>
                      {/* Try parsing editor_report JSON */}
                      {(() => {
                        try {
                          const report = JSON.parse((chapter as Record<string, unknown>).editor_report as string)
                          return <ScoreBadge score={report.overall_score || 0} />
                        } catch { return null }
                      })()}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  )
}
