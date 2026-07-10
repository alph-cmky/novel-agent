# novel-agent 设计文档

> 开源多 Agent 小说写作框架。v1.0 — AI vibe coding 最终版。

## 一、项目定位

novel-agent 是一个"主编程写作"工具：人类设定故事方向，AI 在叙事引擎调度下逐章生成正文，经过编辑审查、一致性审计、世界观提取后，交由人类最终审批。

**核心原则：**

- **人机协作，人类决策** — AI 负责创作和执行，人类在每个章节节点做最终审批
- **质量闭环** — 审查不合格自动重写，但始终保留人类否决权
- **长文记忆** — 向量语义搜索 + 结构化存储 + 上下文压缩，支撑百万字级长篇小说
- **模型路由** — 创意任务走强模型，分析任务走弱模型，控制成本

## 二、系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (React + TypeScript)              │
│   Dashboard  │  Outline  │  Writing  │  Settings  │  Graph     │
└─────────────────────────────┬───────────────────────────────────┘
                              │ SSE + REST
┌─────────────────────────────▼───────────────────────────────────┐
│                      FastAPI (novel_agent/api)                   │
│   routes.py (REST)  │  sse.py (SSE streaming)                   │
│   graph_data.py     │  outline.py (AI outline generation)       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                  LangGraph Pipeline (novel_agent/graph)          │
│                                                                  │
│   Orchestrator → Writer → Editor → Continuity                   │
│                    ↑          ↓                                  │
│                    │    [route_after_continuity]                 │
│                    │     pass / fail+retry / fail+exhausted     │
│                    │          ↓                                  │
│              Orchestrator Review   Worldbuilding                │
│                    │               ↓                             │
│                    │          Human Review (interrupt)           │
│                    │          ↓ approve / ↓ reject              │
│                    └──────────┘    END    └──→ Orchestrator     │
│                                                  Review         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Storage Layer                               │
│   SQLite (projects, chapters, entities, outlines, foreshadowings)│
│   ChromaDB (semantic search over chapter content)               │
│   Checkpoints (SqliteSaver per persist_dir)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 三、LangGraph 流水线

### 3.1 图结构

7 节点 StateGraph，使用 `NovelState` TypedDict 作为共享状态。每个节点消费上游字段、产出下游字段，节点间不直接耦合。

```
Entry → orchestrator_node
  → writer_node
  → editor_node
  → continuity_node
  → [条件路由: route_after_continuity]
      ├─ pass → worldbuilding_node → human_review_node → [条件路由: route_after_human]
      │                                                       ├─ approved → END
      │                                                       └─ rejected + retries → orchestrator_review_node
      ├─ fail + retries left → orchestrator_review_node → writer_node (重写)
      └─ fail + retries exhausted → worldbuilding_node → human_review_node (人类最终决定)
```

### 3.2 路由阈值

| 常量 | 值 | 含义 |
|------|-----|------|
| `MAX_RETRIES` | 3 | 进入 human review 前最多自动重试次数 |
| `CONTINUITY_PASS_SCORE` | 80 | Continuity 审计通过分数 |
| `EDITOR_APPROVE_SCORE` | 60 | Editor 审查通过分数 |

### 3.3 反馈闭环

重写不是简单的"重新生成"。当 Editor/Continuity 发现问题时：

1. **Orchestrator Review** 节点分析所有失败报告 + 人类反馈
2. 生成结构化重写指导：`{instructions, constraints: {focus_areas, strategy_override, avoid, reference_chapter, reference_excerpt}}`
3. Writer 收到指导后，以最高优先级注入 prompt 顶部
4. 如果 Orchestrator Review 指定了 `strategy_override`，Writer 将其合并到 chapter_strategy
5. 重写后重新进入 Editor → Continuity 审查循环

### 3.4 Human-in-the-loop

使用 LangGraph `interrupt()` 暂停图执行：

- **CLI (已弃用)**：捕获 `GraphInterrupt` → 交互式 approve/reject
- **Web UI**：SSE 推送 `review_required` 事件 → 用户点击 Approve/Reject → FastAPI 调用 `resume_graph()` 恢复执行
- 人类拒绝 → Orchestrator Review 分析人类意见 → Writer 带指导重写 → 再次进入 Human Review
- 防止无限拒绝：retry 耗尽后直接 END

### 3.5 Checkpoint 持久化

使用 `AsyncSqliteSaver`（基于 `aiosqlite`），支持图执行中断后恢复：
- 数据库：`{persist_dir}/checkpoints.db`
- 线程 ID：`{project_id}:ch{chapter_number}`（确定性，重启后可恢复）
- 实例按 `persist_dir` 缓存
- 每次开始新写作前清除残留 checkpoint

## 四、Agent 设计

所有 Agent 继承 `BaseAgent`（`novel_agent/agents/base.py`），统一 LLM 调用、工具执行和 trace 记录。

### 4.1 BaseAgent — 基础设施

```python
class BaseAgent:
    name: str                          # "orchestrator" / "writer" / "editor" / ...
    config: AgentConfig                # model, api_key, base_url, max_tokens, temperature
    _tools: dict[str, BaseTool]        # 注册的工具

    async call_model(messages, tools) → AIMessage      # 非流式 LLM 调用（支持 tool calling）
    async call_model_stream(messages) → AsyncIterator  # 流式 LLM 调用
    async execute_tool_calls(tool_calls) → list[dict]  # 执行工具调用，返回 ToolMessage 列表
    async run_with_tools(messages, max_rounds) → (str, TraceStep)  # 工具调用循环
```

**LangChain 迁移要点：**
- 使用 `langchain_openai.ChatOpenAI` 替代 `openai.AsyncOpenAI`
- 每次 LLM 调用创建新的 `ChatOpenAI` 实例，确保 `model`/`temperature`/`max_tokens` 取最新配置
- `_to_langchain_messages()` 将 OpenAI dict 格式转为 LangChain `BaseMessage`，正确传递 `tool_calls`
- LangFuse handler 通过 `config={"callbacks": [lf_handler]}` 注入

### 4.2 OrchestratorAgent — 叙事主编

**职责：** 分析叙事位置，制定章节策略，生成重写指导。

**核心方法：**

| 方法 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `analyze()` | chapter_number, outline, previous_chapters, contexts, story_length, narrative_mode, narrative_perspective, arc_summary | `{narrative_stage, chapter_strategy, context_needed}` | 每章写作前调用 |
| `review_feedback()` | draft_content, editor_report, continuity_report, human_feedback | `{instructions, constraints}` | 需要重写时调用 |

**叙事阶段识别：** intro, development, climax, resolution, unit_arc, mini_climax, transition（7 种）

**篇幅感知：** 根据 `StoryLength`（short/novella/long）调整叙事节奏策略

**输出字段三层分类（详见第六章）：**

### 4.3 WriterAgent — 内容创作

**职责：** 生成章节正文，支持流式输出和工具调用。

**写作规则：**
- 对话占比 ≥ 40%
- 每章结尾必须有钩子
- 口语化网文风格，"展示，不要告知"
- 禁用 AI 味短语（"此外"、"值得注意的是"、"他眼中闪过一丝" 等）

**模式支持：**
- `write()` — 完整生成，最多 3 轮工具调用（`search_context`）
- `write_stream()` — 流式生成，逐 chunk 推送，不支持工具调用

**上下文优先级：** 重写指导 > orchestrator 策略 > 角色/世界观 > 近期摘要

### 4.4 EditorAgent — 编辑审查

**职责：** 5 维度审查草稿，检测 AI 写作模式。

| 维度 | 检查项 |
|------|--------|
| 节奏 (rhythm) | 句子长度变化、段落呼吸感、是否有起伏 |
| AI腔 (ai_flavor) | 重复句式、禁用词、模式化表达 |
| 对话 (dialogue) | 自然度、角色语言一致性 |
| 逻辑 (logic) | 情节自洽性、行为合理性 |
| 文笔 (writing) | 口语化程度、感官细节 |

**规则举例：**
- 连续三个长度相近的句子 → 扣分
- 段落以简洁单行结尾 → 加分
- 结尾总结升华 → 严重扣分
- 情感标签堆砌 → 扣分

**工具：** `detect_ai_flavor` — 基于规则的确定性 AI 味检测（非 LLM 驱动）

### 4.5 ContinuityAgent — 一致性审计

**职责：** 跨章节一致性检查。

| 维度 | 检查项 |
|------|--------|
| 角色一致性 | 外貌、性格、能力、人际关系跨章节一致 |
| 时间线一致性 | 事件顺序、时间流逝 |
| 世界观一致性 | 规则体系、势力关系、物品状态 |

**严重程度：** critical（核心设定破坏）、major（明显矛盾）、minor（小瑕疵）

**工具：** `check_continuity` — 跨已写章节语义搜索（3 个域：角色、事件、世界观）

### 4.6 WorldbuildingAgent — 世界观提取

**职责：** 从已批准章节中提取结构化设定。

**实体类型：** character, location, faction, rule, item, event

**冲突检测：** description_mismatch, rule_violation, timeline

**多视角模式：** 在实体属性中注入 `pov_character` 字段

## 五、叙事系统 — 三层字段分类

叙事系统是 v1.0 的核心设计创新，解决"不同叙事模式下编排器应输出什么"的问题。

### 5.1 五种叙事模式

| 模式 | 标识 | 说明 |
|------|------|------|
| Linear | `linear` | 线性叙事，标准章节推进 |
| Unit Arc | `unit_arc` | 单元剧模式，每 N 章构成独立故事弧 |
| Hybrid | `hybrid` | 混合模式，有单元弧但主线在背景推进 |
| Multi-POV | `multi_perspective` | 多视角，每章或每组章切换 POV |
| Ensemble | `ensemble` | 群像剧，无明确主角，多条角色线并进 |

### 5.2 字段三层分类

**GLOBAL（始终注入 Writer）：**
`primary_storyline`, `pacing`, `key_scenes`, `ending_type`, `foreshadowings_to_address`, `suggested_chapter_words`, `storylines`, `storyline_intersection`

**CONDITIONAL（对应叙事模式才注入）：**
`climax_sequence`（高潮阶段）、`stage_boundary`（阶段边界）、`unit_arc`（unit_arc/hybrid 模式）、`pov_config`（multi_perspective/ensemble 模式）、`time_structure`（非线性时）、`ending_tone`（接近结局时）

**AUXILIARY（参考提示，以文学效果为先）：**
`character_arcs`, `character_emotional_state`, `tension_profile`, `foreshadowing_management`, `scene_composition`

### 5.3 Orchestrator → Writer 策略传递

```
Orchestrator.analyze()
  → 根据 narrative_mode 决定输出哪些字段
  → 存入 orchestrator_strategy (dict)

Writer._format_strategy()
  → 读取 self._narrative_mode
  → Tier 1: GLOBAL 字段 → [本章战略]（必须遵循）
  → Tier 2: CONDITIONAL 字段 → [模式指导]（有条件遵循）
  → Tier 3: AUXILIARY 字段 → [创作参考]（建议性）
  → 组装为 Writer prompt 内的结构化指令段
```

## 六、NovelState — 共享状态

定义在 `novel_agent/graph/state.py`，TypedDict，所有字段非必填。

| 字段 | 类型 | 流向 |
|------|------|------|
| `project_id` | `str` | 入口 → 所有节点 |
| `chapter_number` | `int` | 入口 → 所有节点 |
| `chapter_outline` | `str` | 入口 → orchestrator, writer |
| `story_length` | `str` | 项目配置 → orchestrator |
| `target_chapter_words` | `int` | 项目配置 → writer |
| `narrative_mode` | `str\|None` | 项目配置 → orchestrator, writer, editor, continuity, worldbuilding |
| `narrative_perspective` | `str` | 项目配置 → orchestrator |
| `orchestrator_strategy` | `dict` | orchestrator → writer |
| `character_context` | `str` | orchestrator → writer |
| `world_context` | `str` | orchestrator → writer |
| `recent_summary` | `str` | orchestrator → writer |
| `unresolved_foreshadowings` | `list[str]` | orchestrator → writer |
| `rewrite_instructions` | `str\|dict` | orchestrator_review → writer |
| `draft_content` | `str` | writer → editor, continuity, worldbuilding, human_review |
| `editor_report` | `dict` | editor → continuity, orchestrator_review, human_review |
| `continuity_report` | `dict` | continuity → orchestrator_review, human_review |
| `worldbuilding_report` | `dict` | worldbuilding → 存储 |
| `retry_count` | `int` | writer（递增）→ 路由 |
| `human_approved` | `bool` | human_review → 路由 |
| `human_feedback` | `dict` | human_review → orchestrator_review |
| `existing_world_entities` | `list[dict]` | 存储 → worldbuilding |
| `persist_dir` | `str` | 全局配置 |

## 七、模型路由

`ModelRouter`（`novel_agent/routing/__init__.py`）按任务复杂度分配模型：

| 任务类型 | Agent | 模型 | Temperature |
|----------|-------|------|-------------|
| `CREATIVE` | Writer | QUALITY_MODEL (claude-sonnet-4) | 0.85 |
| `STRUCTURAL` | Orchestrator | BUDGET_MODEL (deepseek-chat) | 0.4 |
| `REVIEW` | Editor, Continuity | BUDGET_MODEL | 0.3 |
| `EXTRACTION` | Worldbuilding | BUDGET_MODEL | 0.2 |

**设计原则：** 创意写作需要高品质模型，分析任务（审查、审计、提取）消耗大但要求较低，走弱模型控制成本。每次 `resolve()` 调用时重新读环境变量，支持运行时覆盖。

## 八、输出健壮性

LLM 输出天然不可靠。两层防护确保 JSON 解析失败不中断流水线：

### 8.1 解析层（`schema/parser.py`）

`parse_json_response(text, defaults)` — 4 层回退：
1. `json.loads()` — 直接解析
2. 正则匹配 ` ```json ... ``` ` markdown 代码块
3. 正则匹配 `{...}` JSON 对象
4. 返回 `defaults`，原始文本存入 `raw_output`

**永不抛异常。**

### 8.2 验证层（`schema/validator.py`）

`OutputValidator.validate(agent_type, raw_dict)` — 3 层策略：
1. 直接构造 Pydantic 模型
2. 如果失败，类型强制修复（字符串→整数、标量→列表、缺失嵌套对象补空字典）
3. 仍失败则返回模型默认实例，标记 `valid=False`

**图解：**
```
LLM 输出 → parse_json_response() → raw_dict → OutputValidator.validate() → Pydantic Model
               ↓ 解析失败                         ↓ 验证失败
            返回 defaults                    返回默认实例 + 错误信息
```

## 九、存储层

### 9.1 SQLite（结构化数据）

WAL 模式 + 外键约束。迁移系统基于 `PRAGMA table_info` 检查已有列，`ALTER TABLE ADD COLUMN` 补全缺失列。

| 表 | 用途 | 关键列 |
|---|------|--------|
| `projects` | 项目配置 | name, title, genre, story_length, target_chapter_words, narrative_mode, narrative_perspective, world_setting |
| `chapters` | 章节数据 | project_id, chapter_number, draft_content, editor_report, continuity_report, worldbuilding_report (JSON 字符串), status |
| `outlines` | 大纲 | project_id, chapter_number, title, summary, status, sort_order |
| `world_entities` | 世界观实体 | project_id, entity_type, name, properties (JSON), first_appearance_chapter |
| `foreshadowings` | 伏笔管理 | project_id, description, planted_chapter, expected_resolve_chapter, status, risk_level, action_needed, reader_knows, characters_aware, characters_unaware |

### 9.2 ChromaDB（向量存储）

`ChapterStore` — 对已写章节内容做语义索引。Writer 的 `search_context` 工具和 Continuity 的 `check_continuity` 工具在此之上做检索。每次 `save_chapter()` 时同步索引。

### 9.3 ProjectManager

`novel_agent/storage/manager.py` — 统一数据访问层。封装 SQLite + ChromaDB 操作。所有 Agent 和 API 路由通过 ProjectManager 读写数据，不直接操作 SQLite。

核心方法：
- `init_project()`, `get_project()`, `update_project()`, `list_projects_with_progress()`
- `save_chapter()`, `get_chapter()`, `get_all_chapters()`, `delete_chapter()`
- `save_outline()`, `get_outline()`, `update_outline_item()`, `delete_outline_item()`
- `build_context()` — 构建写作上下文（近期摘要、角色上下文、世界观上下文）
- `save_world_entities()`, `get_all_world_entities()`
- `add_foreshadowing()`, `update_foreshadowing_status()`, `get_foreshadowings()`
- `delete_project()` — 级联删除关联数据

## 十、可观测性

### 10.1 LangFuse 集成

`novel_agent/observability/langfuse.py` — 零配置设计。不设环境变量时为 no-op。

**架构：**
- 使用 Python `contextvars` 在异步任务间传播 handler 和 trace
- `create_trace()` 在 SSE 处理开始时调用，创建 trace 并设置 contextvar
- BaseAgent 在每次 LLM 调用时从 contextvar 读取 handler，注入 `config["callbacks"]`
- 所有 Agent 的 LLM 调用自动归入同一 trace
- 图完成后 `score_trace({"editor_score": x, "continuity_score": y})` 在 trace 上挂接质量评分

**成本：** LangFuse Cloud Free 提供 50,000 events/月，对当前项目规模完全足够。

### 10.2 Trace 系统

`novel_agent/trace/` — 本地 JSON trace 记录 + Rich CLI 查看器。每步记录 agent、action、token 消耗、耗时、工具调用。

## 十一、Web API

### 11.1 REST 端点（`novel_agent/api/routes.py`）

所有端点以 `/api` 为前缀。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/projects` | 列表 / 创建 |
| GET/PATCH/DELETE | `/projects/{id}` | 详情 / 更新 / 删除 |
| GET/PUT | `/projects/{id}/outline` | 大纲读写 |
| POST | `/projects/{id}/outline/generate` | AI 生成大纲 |
| GET | `/projects/{id}/graph` | 关系图谱数据 |
| GET/DELETE | `/projects/{id}/chapters/{n}` | 章节详情 / 删除 |
| POST | `/projects/{id}/chapters/{n}/write` | 触发写作（SSE） |
| POST | `/projects/{id}/chapters/{n}/approve` | 批准章节 |
| POST | `/projects/{id}/chapters/{n}/reject` | 拒绝章节 |
| PUT | `/projects/{id}/chapters/{n}/draft` | 保存手动编辑 |
| GET | `/projects/{id}/export` | 导出小说 |

### 11.2 SSE 流（`novel_agent/api/sse.py`）

写作管线通过 SSE 向前端推送实时事件：

| 事件类型 | 触发时机 | payload |
|----------|----------|---------|
| `start` | 写作开始 | `{message}` |
| `progress` | 节点状态变化 | `{node, label, status, score, detail}` |
| `chunk` | Writer 流式输出 | 文本内容 |
| `review_required` | 图暂停等待审批 | 草稿预览、评分、问题列表 |
| `done` | 图正常完成 | `{status: "completed"}` |
| `error` | 异常 | `{message, node}` |

**SessionStore** — 内存字典存放活跃写作会话，支持暂停/恢复。

## 十二、Web UI

React + TypeScript + Tailwind CSS + Vite，4 个页面：

| 页面 | 路由 | 功能 |
|------|------|------|
| DashboardPage | `/` | 项目卡片列表、进度统计、创建项目 |
| OutlinePage | `/projects/:id` | 章节树编辑 + 关系图谱（Cytoscape.js） |
| WritingPage | `/projects/:id/chapters/:n` | 三栏布局（上下文/写作区/审查面板），SSE 流式接收 |
| SettingsPage | `/projects/:id/settings` | 项目配置（篇幅、模型、世界观设定） |

## 十三、CLI

精简为 3 个命令组，定位是"服务启动 + 运维操作"：

```
novel-agent serve [--host HOST] [--port PORT] [--reload]   # 启动 Web 服务
novel-agent export [-p PROJECT] [-f md|txt] [-o OUTPUT]    # 导出小说
novel-agent trace show <file>                                # 查看 trace
novel-agent trace ls                                         # 列出 trace
```

项目管理、大纲规划、章节写作均在 Web UI 完成。

## 十四、关键技术决策

1. **LangGraph StateGraph** — 图编排是天然正确选择。节点无直接耦合，条件路由可读，interrupt 机制天然支持 HITL。相比自建状态机，LangGraph 提供了检查点持久化、流式事件、错误恢复等基础设施。

2. **LangChain 而非裸 OpenAI SDK** — 为了接入 LangFuse LangChain CallbackHandler，实现零侵入 LLM 调用追踪。LangChain 的 `BaseMessage` 体系对 tool calling 也有更好的抽象。

3. **三层字段分类** — 叙事系统最大的复杂度来自"不同模式需要不同输出"。GLOBAL/CONDITIONAL/AUXILIARY 分类让 Orchestrator 可以输出统一 schema，Writer 按需消费，避免"一个巨型 prompt"或"N 个不同 schema"两个极端。

4. **Pydantic 全默认值** — 所有 schema 模型字段都有默认值。LLM 输出不可靠，部分字段缺失不能导致管线中断。宁可丢信息也不能停。

5. **模型分离** — 创意走强模型，分析走弱模型。一部长篇小说可能有数百次分析调用，全用强模型成本太高。

6. **反馈闭环而非简单重试** — 重写前 Orchestrator Review 分析失败原因，生成具体指导。比"再试一次"有效得多。

7. **零配置可观测性** — LangFuse 不设环境变量时完全无开销。适合开源项目：本地开发零负担，部署时加两行 env 即可。

## 十五、目录结构

```
novel_agent/
├── agents/           # Agent 实现
│   ├── base.py       # BaseAgent 基类
│   ├── orchestrator.py
│   ├── writer.py
│   ├── editor.py
│   ├── continuity.py
│   └── worldbuilding.py
├── api/              # Web API
│   ├── app.py        # FastAPI 应用
│   ├── routes.py     # REST 端点
│   ├── sse.py        # SSE 流处理
│   ├── graph_data.py # 关系图谱聚合
│   └── outline.py    # AI 大纲生成
├── cli/              # 命令行入口
│   └── main.py
├── graph/            # LangGraph 图定义
│   ├── state.py      # NovelState TypedDict
│   └── chapter.py    # 图构建 + 路由
├── memory/           # 记忆系统
│   ├── compressor.py # ContextCompressor
│   └── embeddings.py
├── observability/    # 可观测性
│   └── langfuse.py   # LangFuse 集成
├── routing/          # 模型路由
│   └── __init__.py   # ModelRouter
├── schema/           # 输出健壮性
│   ├── models.py     # Pydantic 模型
│   ├── parser.py     # JSON 解析器
│   └── validator.py  # 输出校验器
├── storage/          # 数据持久化
│   ├── models.py     # SQLite schema
│   └── manager.py    # ProjectManager
├── style/            # 写作风格检测
│   └── ai_flavor.py  # AI 味规则引擎
├── tools/            # Agent 工具
│   ├── base.py       # BaseTool
│   ├── search.py     # SearchContextTool
│   ├── continuity.py # CheckContinuityTool
│   └── style.py      # DetectAiFlavorTool
├── trace/            # 本地 trace
│   ├── collector.py
│   └── viewer.py
└── config.py         # StoryLength 配置

frontend/             # React Web UI
├── src/
│   ├── pages/         # Dashboard, Outline, Writing, Settings
│   ├── components/    # ScoreBadge, PipelineProgress, GraphCanvas, etc.
│   └── api/           # React Query hooks
└── ...

tests/                # 测试
├── eval/              # 评估基准
└── test_*.py          # 单元 + 集成测试
```
