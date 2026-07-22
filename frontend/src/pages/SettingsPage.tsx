import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import ErrorBoundary from '../components/ErrorBoundary'
import { Loading, Spinner } from '../components/Spinner'

export default function SettingsPage() {
  const { id } = useParams<{ id: string }>()
  const projectId = id!
  const queryClient = useQueryClient()

  const { data: project, isLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => api.getProject(projectId),
  })

  const [storyLength, setStoryLength] = useState('')
  const [targetWords, setTargetWords] = useState(0)
  const [worldSetting, setWorldSetting] = useState('')

  useEffect(() => {
    if (project) {
      setStoryLength(project.story_length || 'long')
      setTargetWords(project.target_chapter_words || 3000)
      setWorldSetting((project as any).world_setting || '')
    }
  }, [project])

  const mutation = useMutation({
    mutationFn: () => api.updateProject(projectId, {
      story_length: storyLength,
      target_chapter_words: targetWords,
      world_setting: worldSetting,
    }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['project', projectId] }),
  })

  return (
    <ErrorBoundary>
      <div>
        <div className="flex items-center justify-between mb-6">
          <div>
            <Link to={`/projects/${projectId}`} className="text-sm text-gray-500 hover:text-blue-600">
              &larr; 返回大纲
            </Link>
            <h1 className="text-2xl font-bold text-gray-900 mt-1">项目设置</h1>
          </div>
        </div>

        {isLoading ? (
          <Loading label="加载中..." className="py-16" />
        ) : (
          <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-2xl space-y-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">基本设置</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">篇幅</label>
                  <select
                    value={storyLength}
                    onChange={(e) => setStoryLength(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="long">长篇 (50-100+章, 3000字/章)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">每章字数</label>
                  <input
                    type="number"
                    value={targetWords}
                    onChange={(e) => setTargetWords(Number(e.target.value))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">世界观设定</h3>
              <textarea
                value={worldSetting}
                onChange={(e) => setWorldSetting(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={8}
                placeholder="手动编辑基础世界观设定，会在写作时提供给 AI..."
              />
            </div>

            <div className="pt-2">
              <button
                onClick={() => mutation.mutate()}
                disabled={mutation.isPending}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {mutation.isPending ? (
                  <span className="flex items-center justify-center gap-2">
                    <Spinner size="sm" className="text-white" />
                    保存中...
                  </span>
                ) : '保存设置'}
              </button>
              {mutation.isSuccess && (
                <span className="ml-3 text-green-600 text-sm">已保存</span>
              )}
              {mutation.isError && (
                <span className="ml-3 text-red-600 text-sm">
                  {(mutation.error as Error).message}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  )
}
