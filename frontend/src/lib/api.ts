import type { OutlineStatus } from './status'

const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }))
    throw new Error(err.message || `HTTP ${res.status}`)
  }
  return res.json()
}

export interface Project {
  id: string
  name: string
  title: string
  genre: string
  story_length: string
  target_chapter_words: number
  created_at: string
  updated_at: string
  chapter_count?: number
  total_chapters?: number
}

export interface OutlineItem {
  project_id: string
  chapter_number: number
  title: string
  summary: string
  status: OutlineStatus
  sort_order: number
}

export interface GraphData {
  nodes: Array<{
    id: string
    label: string
    type: string
    shape: string
    color: string
    properties: Record<string, unknown>
    first_chapter: number
    importance: number
    has_conflict: boolean
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    label: string
    description: string
    relationship_type: string
  }>
}

export const api = {
  // Projects
  getProjects: () => request<Project[]>('/projects'),
  createProject: (data: {
    name: string
    title?: string
    genre?: string
    story_length?: string
    target_chapter_words?: number
    outline_text?: string
  }) => request<Project>('/projects', { method: 'POST', body: JSON.stringify(data) }),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  updateProject: (id: string, data: Record<string, unknown>) =>
    request<Project>(`/projects/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteProject: (id: string) =>
    request<void>(`/projects/${id}`, { method: 'DELETE' }),

  // Outline
  getOutline: (id: string) => request<OutlineItem[]>(`/projects/${id}/outline`),
  saveOutline: (id: string, chapters: Array<Partial<OutlineItem>>) =>
    request<void>(`/projects/${id}/outline`, { method: 'PUT', body: JSON.stringify({ chapters }) }),
  generateOutline: (id: string) =>
    request<OutlineItem[]>(`/projects/${id}/outline/generate`, { method: 'POST' }),

  // Graph
  getGraph: (id: string, untilChapter?: number) =>
    request<GraphData>(`/projects/${id}/graph${untilChapter ? `?until_chapter=${untilChapter}` : ''}`),

  // Chapters
  getChapters: (id: string) => request<Array<{ chapter_number: number; status: string }>>(`/projects/${id}/chapters`),
  getChapter: (id: string, n: number) => request<Record<string, unknown>>(`/projects/${id}/chapters/${n}`),
  approveChapter: (id: string, n: number) =>
    request<void>(`/projects/${id}/chapters/${n}/approve`, { method: 'POST' }),
  rejectChapter: (id: string, n: number, comments: string) =>
    request<void>(`/projects/${id}/chapters/${n}/reject`, {
      method: 'POST',
      body: JSON.stringify({ comments }),
    }),
  saveDraft: (id: string, n: number, content: string) =>
    request<void>(`/projects/${id}/chapters/${n}/draft`, {
      method: 'PUT',
      body: JSON.stringify({ content }),
    }),

  // Export
  exportNovel: (id: string) => request<string>(`/projects/${id}/export`),

  // SSE writing stream
  writeChapterUrl: (id: string, n: number) => `${BASE}/projects/${id}/chapters/${n}/write`,
}
