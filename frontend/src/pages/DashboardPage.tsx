import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { api, type Project } from '../lib/api'
import ExportModal from '../components/ExportModal'
import ErrorBoundary from '../components/ErrorBoundary'
import { Loading, Spinner } from '../components/Spinner'

const LENGTH_LABELS: Record<string, string> = {
  long: '长篇',
}

function ProjectCard({ project }: { project: Project }) {
  const queryClient = useQueryClient()
  const total = (project as any).total_chapters || 0
  const done = (project as any).chapter_count || 0
  const progress = total > 0 ? Math.round((done / total) * 100) : 0
  const [showExport, setShowExport] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(false)

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteProject(project.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  return (
    <>
      <div className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
        <Link
          to={`/projects/${project.id}`}
          className="block"
        >
          <h3 className="font-semibold text-gray-900 truncate">
            {project.title || project.name}
          </h3>
          <div className="flex items-center gap-2 mt-1">
            {project.genre && (
              <span className="text-xs text-gray-500">{project.genre}</span>
            )}
            <span className="text-xs bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">
              {LENGTH_LABELS[project.story_length] || project.story_length}
            </span>
          </div>
          {total > 0 && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>进度</span>
                <span>{done}/{total} 章</span>
              </div>
              <div className="w-full bg-gray-100 rounded-full h-1.5">
                <div
                  className="bg-blue-600 h-1.5 rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
          <p className="text-xs text-gray-400 mt-3">
            最后写作: {project.updated_at.slice(0, 10)}
          </p>
        </Link>
        <div className="flex gap-2 mt-3 pt-3 border-t border-gray-100">
          <Link
            to={`/projects/${project.id}`}
            className="flex-1 text-center text-xs text-gray-500 hover:text-blue-600 py-1"
          >
            大纲
          </Link>
          <button
            onClick={(e) => { e.preventDefault(); setShowExport(true) }}
            className="flex-1 text-xs text-gray-500 hover:text-blue-600 py-1"
          >
            导出
          </button>
          {confirmingDelete ? (
            <>
              <button
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                className="flex-1 text-xs text-white bg-red-600 hover:bg-red-700 rounded py-1 disabled:opacity-50"
              >
                {deleteMutation.isPending ? (
                  <span className="flex items-center justify-center gap-1.5">
                    <Spinner size="sm" className="text-white" />
                    删除中…
                  </span>
                ) : '确认删除'}
              </button>
              <button
                onClick={() => setConfirmingDelete(false)}
                className="flex-1 text-xs text-gray-500 hover:text-gray-700 py-1"
              >
                取消
              </button>
            </>
          ) : (
            <button
              onClick={() => setConfirmingDelete(true)}
              className="flex-1 text-xs text-gray-400 hover:text-red-600 py-1"
            >
              删除
            </button>
          )}
        </div>
        {deleteMutation.isError && (
          <p className="text-xs text-red-500 mt-2">
            {(deleteMutation.error as Error).message}
          </p>
        )}
      </div>
      {showExport && (
        <ExportModal
          projectId={project.id}
          onClose={() => setShowExport(false)}
        />
      )}
    </>
  )
}

function CreateProjectModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [name, setName] = useState('')
  const [title, setTitle] = useState('')
  const [genre, setGenre] = useState('')
  const [storyLength, setStoryLength] = useState('long')
  const [targetWords, setTargetWords] = useState(3000)
  const [outlineText, setOutlineText] = useState('')

  const mutation = useMutation({
    mutationFn: () => api.createProject({
      name, title, genre, story_length: storyLength,
      target_chapter_words: targetWords, outline_text: outlineText,
    }),
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      navigate(`/projects/${project.id}`)
    },
  })

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-lg font-bold text-gray-900 mb-4">新建项目</h2>
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">项目名称 *</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="如：修仙模拟器"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">书名</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="同项目名可不填"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">题材</label>
            <input
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="如：都市、玄幻、科幻"
            />
          </div>
          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-600 mb-1">篇幅</label>
              <select
                value={storyLength}
                onChange={(e) => setStoryLength(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              >
                <option value="long">长篇 (50-100+章)</option>
              </select>
            </div>
            <div className="w-24">
              <label className="block text-xs font-medium text-gray-600 mb-1">每章字数</label>
              <input
                type="number"
                value={targetWords}
                onChange={(e) => setTargetWords(Number(e.target.value))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">
              梗概（可选，一行一章）
            </label>
            <textarea
              value={outlineText}
              onChange={(e) => setOutlineText(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={4}
              placeholder="主角穿越到异世界，发现拥有金手指&#10;遇见第一个伙伴，开始冒险&#10;..."
            />
          </div>
          {mutation.isError && (
            <p className="text-red-600 text-xs">
              {(mutation.error as Error).message}
            </p>
          )}
        </div>
        <div className="flex gap-2 mt-4">
          <button
            onClick={onClose}
            className="flex-1 border border-gray-300 text-gray-700 rounded-lg py-2 text-sm"
          >
            取消
          </button>
          <button
            onClick={() => mutation.mutate()}
            disabled={!name || mutation.isPending}
            className="flex-1 bg-blue-600 text-white rounded-lg py-2 text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {mutation.isPending ? (
              <span className="flex items-center justify-center gap-2">
                <Spinner size="sm" className="text-white" />
                创建中...
              </span>
            ) : '创建'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function DashboardPage() {
  const [showModal, setShowModal] = useState(false)
  const { data: projects, isLoading, error } = useQuery({
    queryKey: ['projects'],
    queryFn: api.getProjects,
    refetchInterval: 10_000,
  })

  return (
    <ErrorBoundary>
      <div>
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900">项目看板</h1>
          <button
            onClick={() => setShowModal(true)}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
          >
            新建项目
          </button>
        </div>

        {isLoading && <Loading label="加载中..." className="py-20" />}
        {error && (
          <div className="text-center py-20 text-red-500">
            加载失败: {(error as Error).message}
          </div>
        )}
        {!isLoading && !error && projects && (
          <>
            {projects.length === 0 ? (
              <div className="text-center py-20 text-gray-500">
                <p className="text-lg mb-2">还没有项目</p>
                <p className="text-sm">点击"新建项目"开始你的第一部小说</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {projects.map((p) => (
                  <ProjectCard key={p.id} project={p} />
                ))}
              </div>
            )}
          </>
        )}

        {showModal && <CreateProjectModal onClose={() => setShowModal(false)} />}
      </div>
    </ErrorBoundary>
  )
}
