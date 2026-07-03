# novel-agent Web 重设计 Spec

## 目标

将 novel-agent 从 CLI 命令行工具重设计为 Web 应用，提供完整的项目管理、大纲规划、流式写作、关系图谱可视化和 Human Review 交互体验。

## 架构概览

```
React + Tailwind (SPA)
        │
FastAPI REST + SSE
        │
LangGraph StateGraph (不变)
        │
ChromaDB + SQLite + SqliteSaver
```

- 前端：React + Tailwind CSS + Cytoscape.js（关系图谱）
- 后端：FastAPI（新增 REST + SSE API 层）
- Agent 流水线：现有 LangGraph StateGraph 不变，Human Review 节点通过 API 端点驱动 `Command(resume=...)`
- 流式：SSE（Server-Sent Events），Writer 生成文本逐 chunk 推送

## 页面结构

### 1. 项目看板 `/`

- 已有项目卡片列表，每张卡片显示：书名、篇幅标签、进度（3/50 章）、最后写作时间
- 点击卡片进入大纲页
- "新建项目"按钮 → 弹出向导表单（书名、题材、篇幅、可选梗概）
- 空状态：引导文案 + 新建入口

### 2. 大纲页 `/projects/:id`

两个 Tab：

**大纲 Tab**
- 左侧：章节树，可拖拽排序、增删、右键菜单（上移/下移/删除）
- 右侧：选中章节的大纲详情，可编辑
- 顶部工具栏：
  - "AI 生成全书大纲" → Orchestrator 根据篇幅+梗概生成
  - "添加章节" → 手动新增
  - 每章状态标签：待写 / 已生成 / 已审批
- 点击某章"开始写作" → 进入章节写作页

**关系图谱 Tab**
- Cytoscape.js 力导向图渲染
- 节点类型：角色（圆形）、地点（菱形）、物品（方形）、组织（六边形）、事件（三角形）
- 边：关系连线，hover 显示关系描述
- 时间轴滑块：拖动只显示前 N 章的实体和关系
- 点击节点：弹出侧面板，显示属性、首次出现章节、属性变化历史
- 冲突节点：红色边框+脉冲动画标记
- 工具栏：筛选节点类型、放大/缩小/重置

### 3. 章节写作页 `/projects/:id/chapters/:n`

三栏布局（左 250px / 中 flex-1 / 右 300px）：

**左栏（上下文面板，可折叠）**
- 本章大纲
- 相关角色列表
- 世界观设定
- 前文提要
- 以卡片形式展示，可展开/折叠

**中栏（写作区）**
- 生成前：空白区域 + "生成草稿" 按钮
- 生成中：流式文字渲染，光标跟随最新内容
- 生成后：可滚动阅读，顶部出现审批栏

**右栏（审查面板）**
- 流水线进度指示器（类似 CI/CD pipeline 可视化）
- 依次显示：Writer → Editor → Continuity → Worldbuilding → Review
- 每步完成时展开：评分、问题列表
- Review 步：显示审批按钮

### 4. 项目设置页 `/projects/:id/settings`

- 篇幅和每章字数（可改）
- 模型配置（QUALITY_MODEL / BUDGET_MODEL）
- 世界观设定文本（手动编辑基础世界观）

## 核心写作流程

```
用户点击"生成草稿"
  → POST /api/projects/:id/chapters/:n/write
  → 后端启动 StateGraph，建立 SSE 连接
  → SSE 事件流:

    event: progress   data: {node: "orchestrator", status: "running"}
    event: progress   data: {node: "orchestrator", status: "done", ...}

    event: progress   data: {node: "writer", status: "running"}
    event: chunk       data: "主角睁开眼，发现自己..."
    event: chunk       data: "躺在一片陌生的草地上..."
    event: progress   data: {node: "writer", status: "done", char_count: 3200}

    event: progress   data: {node: "editor", status: "running"}
    event: progress   data: {node: "editor", status: "done", score: 82, issues: [...]}

    event: progress   data: {node: "continuity", status: "done", score: 90, ...}

    event: progress   data: {node: "worldbuilding", status: "done", entities: 3, conflicts: 0}

    event: review_required  data: {draft_preview: "...", scores: {...}, issues: [...]}

  → 前端显示审批栏

用户操作:
  - 批准: POST /api/projects/:id/chapters/:n/approve
    → Command(resume={action:"approve"}) → 章节保存 → 关系图更新消息
    → SSE event: done

  - 拒绝: POST /api/projects/:id/chapters/:n/reject {comments: "角色语气不一致"}
    → Command(resume={action:"reject", comments:"..."})
    → 图继续执行: Orchestrator Review → Writer → Editor → ... → 再次 review_required

  - 编辑草稿: 中栏变可编辑 textarea → PUT /api/projects/:id/chapters/:n/draft 保存
```

## Human Review 适配

当前 `human_review_node` 使用 `interrupt()` 挂起。Web 模式流程：

1. `interrupt()` 挂起时，LangGraph 抛出 `GraphInterrupt`（同 CLI 模式）
2. 后端 SSE handler 捕获 `GraphInterrupt`，将 interrupt 数据作为 `review_required` 事件推给前端
3. 前端显示审批 UI，等待用户操作（不阻塞图——图已在挂起状态）
4. 用户操作后前端调 `/approve` 或 `/reject`
5. 后端 handler 用 `graph.ainvoke(Command(resume=feedback), config)` 恢复执行
6. 图继续执行，新事件沿同一 SSE 连接推送
7. 最终 `done` 事件关闭 SSE

## 关系图谱

### 数据源

- 已有 `world_entities` SQLite 表 + WorldbuildingReport 的 `relationships` 字段
- 新增 API：`GET /api/projects/:id/graph?until_chapter=N`
- 按时间轴参数过滤章节，支持渐进展开

### 可视化

- Cytoscape.js 渲染，使用 `cose-bilkent` 力导向布局
- 节点颜色/形状按类型区分
- 节点大小 = `importance`（出现次数 + 关联边数）
- 冲突检测：Worldbuilding 输出的 conflicts 列表标注到对应节点/边
- 交互：hover → tooltip，click → 侧面板详情，drag → 重新布局
- 双击节点 → 跳转到该实体首次出现章节
- 时间轴滑块：`<input type="range">` 绑定 `until_chapter` 参数，拖动时重新请求数据

## API 设计

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/projects` | GET | 项目列表 |
| `/api/projects` | POST | 创建项目 |
| `/api/projects/:id` | GET | 项目详情 |
| `/api/projects/:id` | PATCH | 更新项目设置 |
| `/api/projects/:id/outline` | GET | 获取大纲 |
| `/api/projects/:id/outline` | PUT | 保存大纲（章节列表） |
| `/api/projects/:id/outline/generate` | POST | AI 生成全书大纲 |
| `/api/projects/:id/graph` | GET | 关系图谱数据（?until_chapter=N） |
| `/api/projects/:id/chapters` | GET | 已写章节列表 |
| `/api/projects/:id/chapters/:n` | GET | 章节详情 |
| `/api/projects/:id/chapters/:n/write` | POST | 触发写作流水线（SSE） |
| `/api/projects/:id/chapters/:n/approve` | POST | 批准章节 |
| `/api/projects/:id/chapters/:n/reject` | POST | 拒绝章节 {comments} |
| `/api/projects/:id/chapters/:n/draft` | PUT | 保存手动编辑的草稿 |
| `/api/projects/:id/export` | GET | 导出完整小说（纯文本/Markdown） |

### SSE 事件类型

| event | data | 说明 |
|-------|------|------|
| `progress` | `{node: "orchestrator"\|"writer"\|"editor"\|"continuity"\|"orchestrator_review"\|"worldbuilding", status: "running"\|"done", ...result}` | 流水线节点状态，done 时附带该节点输出 |
| `chunk` | 纯文本字符串 | Writer 输出的增量文本 |
| `review_required` | `{draft_preview, editor_score, continuity_score, editor_issues, continuity_issues, new_entities, conflicts, retry_count}` | Human Review 挂起，等待用户决策 |
| `error` | `{message, node}` | 某节点执行失败 |
| `done` | `{chapter_content, editor_report, continuity_report, worldbuilding_report}` | 章节完成（批准或重试耗尽），关闭 SSE 连接 |

所有事件结束后（`done` 或 `error` 后没有后续节点），SSE 连接自动关闭。如中途断开，可用 `GET /api/projects/:id/chapters/:n` 查询最新状态。

## 需要新增/改动的后端代码

### 新增

- `novel_agent/api/routes.py` — FastAPI Router，上述所有 API 端点
- `novel_agent/api/sse.py` — SSE 流工具函数，包装 StateGraph 执行和事件推送
- `novel_agent/api/graph_data.py` — 从 SQLite 聚合图谱数据（节点+边+冲突）
- `novel_agent/api/outline.py` — 大纲 CRUD + AI 生成大纲逻辑（Orchestrator 新增 `generate_outline()` 方法，根据篇幅+梗概生成章节列表+每章概要）

### 改动

- `novel_agent/api/app.py` — 从 Chainlit 改为 FastAPI + 静态文件 serve
- `novel_agent/graph/chapter.py` — writer_node 通过 `async for chunk in writer.write_stream(...)` 逐 chunk 产出文本；graph 节点间通过状态传递已累积文本，SSE 层从状态变更中提取 chunk 事件推送
- `novel_agent/storage/manager.py` — 新增大纲的 CRUD（outline 表或 chapters 表扩充）

### 不变

- Agent 层（orchestrator/writer/editor/continuity/worldbuilding）完全不动
- 路由逻辑、记忆系统、模型路由、schema 校验完全不动
- CLI 保留不动，作为轻量入口

## 前端技术栈

- React 18 + TypeScript
- Tailwind CSS
- React Router v6（客户端路由）
- Cytoscape.js + cytoscape-cose-bilkent（关系图谱布局）
- 状态管理：React Query（服务端状态）+ Context（UI 状态）
- SSE 消费：EventSource API
- 构建：Vite
- 部署：Vite build → 静态文件，FastAPI serve

## 前端组件树（概要）

```
App
├── Layout (导航栏 + 侧边栏)
├── Pages
│   ├── DashboardPage    // 项目看板
│   │   ├── ProjectCard
│   │   └── CreateProjectModal
│   ├── OutlinePage      // 大纲 + 关系图
│   │   ├── OutlineTab
│   │   │   ├── ChapterTree
│   │   │   └── ChapterDetail
│   │   └── GraphTab
│   │       ├── GraphCanvas (Cytoscape)
│   │       ├── GraphTimeline
│   │       └── NodeDetailPanel
│   ├── WritingPage      // 章节写作
│   │   ├── ContextPanel
│   │   ├── WritingArea (SSE stream)
│   │   ├── ReviewPanel
│   │   └── ApprovalBar
│   └── SettingsPage
└── Shared
    ├── PipelineProgress
    ├── ScoreBadge
    └── ErrorBoundary
```

## 大纲数据模型

```sql
-- 新表或扩充 chapters 表
CREATE TABLE outlines (
    project_id TEXT NOT NULL,
    chapter_number INTEGER NOT NULL,
    title TEXT DEFAULT '',
    summary TEXT DEFAULT '',      -- 本章概要
    status TEXT DEFAULT 'pending', -- pending / writing / drafted / approved
    sort_order INTEGER DEFAULT 0,
    PRIMARY KEY (project_id, chapter_number)
);
```

## 不在本期范围

- 部署方案（Docker/CI 已有基础，后续补充）
- 多用户认证和隔离（当前单用户 SQLite）
- 移动端适配（PC 优先）
- VS Code 插件
- 付费/计费
- 协作写作
