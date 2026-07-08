# novel-agent

开源多 Agent 小说写作框架，支持短篇/中篇/长篇多种篇幅。Writer → Editor → Continuity → Human Review 流水线，支持反馈闭环和 Human-in-the-loop。

## 运行

```
uv sync                    # 安装依赖
uv run pytest              # 运行测试
uv run ruff check .        # lint
uv run python examples/hello_graph.py  # 跑 demo
uv run novel-agent --help  # CLI 入口
```

## 技术栈

Python 3.12+, LangGraph (graph orchestration), LangChain + OpenAI (LLM), ChromaDB (vector memory), FastAPI + Chainlit (web UI), Click (CLI), Pydantic (models).

## 项目约束

- Agent 之间通过 LangGraph StateGraph 传递状态，不直接耦合
- 新增 Agent 继承 `BaseAgent`，放在 `novel_agent/agents/` 下
- 新增 Tool 实现 `BaseTool`，放在 `novel_agent/tools/` 下
- 数据访问走 `ProjectManager`，不直接操作 SQLite
- 测试文件放在 `tests/`，命名 `test_<模块名>.py`
- 数据库 schema 变更需要同时更新 `novel_agent/storage/models.py` 和 `novel_agent/storage/manager.py`
- 全局配置常量放在 `novel_agent/config.py`

## 流水线架构

7 节点 StateGraph（含反馈闭环 + Human-in-the-loop）：

```
Orchestrator → Writer → Editor → Continuity → [
    pass → Worldbuilding → Human Review (interrupt) → [
        approved → Done,
        rejected → Orchestrator Review → Writer (with rewrite guidance)
    ],
    fail + retries → Orchestrator Review → Writer (auto feedback loop),
    fail + no retries → Worldbuilding → Human Review (human final say)
]
```

- **Orchestrator**: 叙事阶段分析 + 篇幅感知策略（短篇快速推进/长篇渐进展开）+ 反馈分析（review_feedback）
- **Writer**: 动态字数目标 + 去 AI 味写作规则 + search_context 工具 + 重写指导接收
- **Editor**: 5 维度审查 + detect_ai_flavor 工具
- **Continuity**: 3 维度一致性审计 + check_continuity 工具
- **Orchestrator Review**: 分析 Editor/Continuity/Human 反馈，生成具体重写指导（反馈闭环核心）
- **Worldbuilding**: 实体提取 + 冲突检测 + 持久化到 SQLite
- **Human Review**: LangGraph `interrupt()` 暂停流水线，等待人类审批（CLI 交互式 / Chainlit 按钮式）

## 反馈闭环

重写不再是"原样重试"。当 Editor/Continuity 发现问题时：

1. **Orchestrator Review** 节点分析失败报告 + 人类反馈
2. 生成具体重写指导（"角色语气不一致，第1章是愤世嫉俗的，这里变成了乐观"）
3. Writer 收到 `rewrite_instructions`，在 prompt 最前面插入指导
4. 重写后重新进入 Editor/Continuity 审查

## Human-in-the-loop

`human_review_node` 使用 LangGraph `interrupt()` 暂停图执行：
- **CLI**: 捕获 `GraphInterrupt` → 显示草稿+评分 → 用户输入 approve/reject → `Command(resume=feedback)`
- **Chainlit**: 捕获 `GraphInterrupt` → 显示草稿+按钮 → 用户点击 Approve/Reject → `Command(resume=feedback)`
- 人类拒绝 → Orchestrator Review 分析人类意见 → Writer 带指导重写 → 再次进入 Human Review

## 路由逻辑

- `route_after_continuity`: score ≥ 80 且无 critical → worldbuilding；有问题且 retry < 3 → orchestrator_review；retry 耗尽 → worldbuilding（人类最终决定）
- `route_after_human`: approved → END；rejected + retries left → orchestrator_review；rejected + no retries → END（防止无限拒绝）
- 阈值常量定义在 `novel_agent/graph/chapter.py`：MAX_RETRIES=3, CONTINUITY_PASS_SCORE=80, EDITOR_APPROVE_SCORE=60

## NovelState 字段

定义在 `novel_agent/graph/state.py`。关键字段：
- `story_length`, `target_chapter_words` — 篇幅控制
- `draft_content`, `editor_report`, `continuity_report`, `worldbuilding_report` — Agent 输出
- `rewrite_instructions` — Orchestrator Review 给 Writer 的重写指导
- `human_feedback`, `human_approved` — Human-in-the-loop 状态

## 篇幅支持

`novel_agent/config.py` 定义 `StoryLength` 枚举：short（短篇 1500字/章）、novella（中篇 3000字/章）、long（长篇 3000字/章）。

- 项目初始化时设定默认篇幅 + 每章字数，写单章时可覆盖
- Orchestrator 根据篇幅调整叙事节奏策略
- Writer 系统提示动态替换字数目标
- AgentConfig.max_tokens 随字数目标自动缩放

## 记忆系统

- **Recent Memory**: ContextCompressor 压缩前文为摘要（40K token 阈值）
- **Long-term Memory**: ChromaDB 向量存储 + SQLite 结构化存储（projects, chapters, world_entities, foreshadowings）
- Worldbuilding 实体跨章节累积，通过 `ProjectManager.save_world_entities()` 持久化

## 模型路由

`novel_agent/routing/__init__.py` — ModelRouter 按 TaskClass 分配模型：
- CREATIVE → QUALITY_MODEL（创作）
- STRUCTURAL / REVIEW / EXTRACTION → BUDGET_MODEL（分析）
- `resolve()` 每次调用时读取 env var，支持运行时覆盖

## 输出健壮性

`novel_agent/schema/parser.py` — 共享 JSON 解析器，3 层策略：直接解析 → markdown 代码块 → 正则提取兜底
`novel_agent/schema/validator.py` — OutputValidator，3 层验证：Pydantic 解析 → 强制转换 → 默认值兜底

## 可观测性

- `novel_agent/trace/collector.py` — JSON trace 记录
- `novel_agent/trace/viewer.py` — Rich CLI trace 查看器

## 测试

```
tests/
├── eval/                          # 评估
│   ├── continuity_benchmark.py    # Continuity 基准测试（注入 bug）
│   └── test_continuity_benchmark.py
├── test_ai_flavor.py              # AI 味检测规则测试
├── test_compressor.py             # ContextCompressor 测试
├── test_config.py                 # 篇幅配置测试
├── test_graph_routing.py          # 路由逻辑测试（含反馈闭环）
├── test_validator.py              # OutputValidator 测试
└── test_*_integration.py          # 集成测试（标记 slow）
```

## CLI 命令

```
novel-agent init -n <name> [-t title] [-g genre] [-l short|novella|long] [-w words]
novel-agent write -c <chapter> -o <outline> [-p project] [-m model] [-w words]
novel-agent quick -c <chapter> -o <outline> [-p project] [-m model] [-w words]
novel-agent list
novel-agent trace show <file>
novel-agent trace ls
```

`write` 命令支持 Human-in-the-loop：流水线运行到 Human Review 节点时暂停，用户输入 approve/reject 决定是否通过。

## Chainlit 命令

```
new <name> [title] [genre] [--length short|novella|long] [--words N]
list
select <project_id>
chapters
write <chapter> <outline> [--words N]
```

`write` 命令支持 Human-in-the-loop：通过 Approve/Reject 按钮进行人工审批。

## Checkpoint 持久化

`build_chapter_graph(persist_dir)` 接受可选的 `persist_dir` 参数：
- 不传 → `MemorySaver`（CLI / Chainlit 兼容）
- 传入 → `SqliteSaver`，checkpoint 写入 `{persist_dir}/checkpoints.db`
- `thread_id` 使用 `{project_id}:ch{chapter_number}` 确定性格式，重启后可恢复
- `_checkpointer_cache` 按 persist_dir 缓存 SqliteSaver 实例

## Web API

FastAPI REST 端点定义在 `novel_agent/api/routes.py`：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/projects | 项目列表 |
| POST | /api/projects | 创建项目 |
| GET | /api/projects/{id} | 项目详情 |
| PATCH | /api/projects/{id} | 更新设置 |
| DELETE | /api/projects/{id} | 删除项目 |
| GET/PUT | /api/projects/{id}/outline | 大纲管理 |
| POST | /api/projects/{id}/outline/generate | AI 生成大纲 |
| GET | /api/projects/{id}/graph | 关系图谱数据 |
| GET/DELETE | /api/projects/{id}/chapters/{n} | 章节详情/删除 |
| POST | /api/projects/{id}/chapters/{n}/write | 触发写作（SSE） |
| POST | /api/projects/{id}/chapters/{n}/approve | 批准章节 |
| POST | /api/projects/{id}/chapters/{n}/reject | 拒绝章节 |

`ProjectManager` 提供 `delete_chapter()` 和 `delete_project()` 方法，级联删除关联数据。
