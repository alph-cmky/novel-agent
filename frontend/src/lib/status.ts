// 章节/大纲状态枚举 —— 与后端 novel_agent/schema/enums.py 保持一致。
// 后端是唯一契约（SQLite 持久化 + API 返回值），此处是前端镜像 + 展示文案。

export type ChapterStatus = 'draft' | 'approved'

export type OutlineStatus = 'pending' | 'writing' | 'drafted' | 'approved'

export const OUTLINE_STATUS_META: Record<OutlineStatus, { label: string; cls: string }> = {
  pending: { label: '待写', cls: 'bg-gray-100 text-gray-600' },
  writing: { label: '写作中', cls: 'bg-blue-100 text-blue-600' },
  drafted: { label: '已生成', cls: 'bg-yellow-100 text-yellow-600' },
  approved: { label: '已审批', cls: 'bg-green-100 text-green-600' },
}
