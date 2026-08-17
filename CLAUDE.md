# novel-agent

开源多 Agent 小说写作框架，支持短篇/中篇/长篇。基于 LangGraph 的递归自进化流水线，支持 Human-in-the-loop。

## 启动

```
uv sync                                    # 安装依赖
uv run novel-agent serve [--reload]        # 后端 http://0.0.0.0:8000
cd frontend && npm run dev                 # 前端 dev（Vite 代理 /api → 8000）
uv run pytest                              # 测试
uv run ruff check .                        # lint
```

数据默认存 `./novel-data`（SQLite + LangGraph checkpoint）。

## 技术栈

Python 3.12+, LangGraph (graph orchestration), LangChain + OpenAI (LLM), ChromaDB (vector memory), FastAPI (web API) + React/Vite/Tailwind (前端), Click (CLI), Pydantic (models)。Chainlit 为 legacy 入口。

## 项目约束

- Agent 之间通过 LangGraph StateGraph 传递状态，不直接耦合
- 新增 Agent 继承 `BaseAgent`，放在 `novel_agent/agents/` 下
- 新增 Tool 实现 `BaseTool`，放在 `novel_agent/tools/` 下
- 数据访问走 `ProjectManager`，不直接操作 SQLite
- 测试文件放在 `tests/`，命名 `test_<模块名>.py`，集成测试标记 `slow`
- 数据库 schema 变更需同时更新 `novel_agent/storage/models.py` 和 `novel_agent/storage/manager.py`
- 全局配置常量放在 `novel_agent/config.py`

## 测试约定

- Agent 层测试不调真实 LLM：`patch.object(agent, "run_with_tools", new=AsyncMock(return_value=("<json>", None)))`（或直接赋值 async 函数），只验证输入组装与输出解析；纯函数（`strip_none`、`_format_strategy` 等）直接测。
- Bug 修复必须配回归测试锁定改动行为（「可验证才算完成」的延伸）。
- 覆盖率用「模块 → `tests/test_<模块名>.py`」映射人工评估，不引入 pytest-cov（避免新增依赖）。

## 流水线架构

v2 递归自进化流水线（默认 `evolution_enabled=True`）：

```
Orchestrator → Evolution Subgraph [
    Writer → Editor → Continuity → EvolutionOrchestrator → [continue|select_best]
] → Worldbuilding → Human Review → [approved → END | rejected → evolution_writer]
```

- **Orchestrator**: 叙事阶段分析 + 篇幅感知策略 + 上下文组装
- **Writer**: 动态字数目标 + 去 AI 味写作规则 + search_context 工具 + 接收结构化 improvement_plan
- **Editor**: 5 维度审查 + detect_ai_flavor 工具
- **Continuity**: 3 维度一致性审计 + check_continuity 工具
- **EvolutionOrchestrator**: 元评估器 — 版本对比 + Delta 分析 + 终止判断 + 改进计划生成（规则层 + LLM 增强）
- **EvolutionSelectBest**: 选择最优版本，一次性写 DB
- **Worldbuilding**: 实体提取 + 冲突检测 + 持久化到 SQLite
- **Human Review**: LangGraph `interrupt()` 暂停流水线，拒绝后触发新进化周期（max 2 轮）

### 兼容参数（`evolution_enabled=False`）

当前 `_build_workflow()` 仍构建同一套递归自进化图；该参数主要影响
`human_review` 拒绝后的处理逻辑。旧版线性反馈节点仍作为兼容代码保留，尚未作为独立
运行模式接线，不应在新功能中依赖。

## 递归自进化

每轮 Writer → Editor → Continuity 产出评估报告，EvolutionOrchestrator 对比上轮计算各维度 Delta，生成 `improvement_plan`（focus_dimensions + preserve + avoid）驱动下一轮创作。

- 7 种终止条件：单维度崩溃、Editor 暴跌、综合退化、天花板、最大轮次、收敛、平台期
- 进化过程中不写 DB（只存 LangGraph checkpoint），结束后一次性落库
- 人类拒绝 → 构建 improvement_plan → 重置计数器 → max 2 轮

规则层（确定性：Delta 计算 / 终止判断）在 `graph/evolution.py`；LLM 层（自然语言改进指导）由 `EvolutionOrchestratorAgent.enrich_plan()` 降级提供，失败不影响流水线。

## NovelState 关键字段

- 进化控制：`evolution_enabled` / `evolution_max_rounds` / `evolution_convergence_threshold`
- 进化状态：`evolution_round` / `evolution_version` / `evolution_history` / `evolution_improvement_plan` / `evolution_termination` / `evolution_best_*`
- 旧版兼容（`evolution_enabled=False`）：`retry_count` / `rewrite_instructions`

完整字段见 `graph/state.py`。

## 记忆系统

- **Recent Memory**: ContextCompressor 压缩前文为摘要（40K token 阈值）
- **Long-term Memory**: ChromaDB 向量 + SQLite 结构化（projects, chapters, world_entities, foreshadowings）；Worldbuilding 实体经 `ProjectManager.save_world_entities()` 跨章节累积

## 模型路由

`novel_agent/routing/__init__.py` — ModelRouter 按 TaskClass 分配模型：
- CREATIVE → QUALITY_MODEL（创作）
- STRUCTURAL / REVIEW / EXTRACTION / META_EVALUATION → BUDGET_MODEL（分析）
- `resolve()` 每次读 env var，支持运行时覆盖

## 输出健壮性

LLM 结构化输出有三类偶发失败：**JSON 语法错误**（漏引号、尾逗号）、**空 content**、**输出截断**。防御分四层：

- `novel_agent/schema/parser.py` — 共享 JSON 解析器。候选文本：直接解析 → markdown 代码块 → 最外层 `{...}`；每层 `json.loads` 失败后先走 `_repair_json`（迭代修复尾逗号、漏左引号的字符串值）再试。所有返回 dict 经 `strip_none` 递归清 None。
- `novel_agent/schema/validator.py` — OutputValidator：Pydantic 解析 → 类型强制转换 → 默认值兜底。
- `novel_agent/agents/base.py` — `call_model` 检测「content 空且无 tool_calls」时重试最多 3 次（step-3.7-flash 偶发空输出）。
- `novel_agent/graph/chapter.py` — `_config_for` 给 EXTRACTION 任务 `max_tokens=8192`（worldbuilding 输出实体+伏笔量大，4096 会截断）。

**易错点**：`dict.get(key, default)` 只在键**不存在**时返回 default；键存在但值为 None 时返回 None。因此所有 LLM 输出必须经过 `strip_none`，否则下游 `None.get(...)` 崩溃。

## CLI 命令

```
novel-agent serve [--host 0.0.0.0] [--port 8000] [--reload]   # 启动 Web 服务
novel-agent export [-p project] [-f md|txt] [-o output]        # 导出小说
```

项目管理、大纲规划、章节写作请在 Web UI 中完成。

## Web API

REST 端点定义在 `novel_agent/api/routes.py`（SSE 流式写作 + Human-in-the-loop）。`ProjectManager` 提供 `delete_chapter()` / `delete_project()` 级联删除。

## Checkpoint 持久化

`build_chapter_graph(persist_dir)` 接受可选 `persist_dir`：
- 不传 → `MemorySaver`（兼容 legacy CLI/Chainlit）
- 传入 → `SqliteSaver`，checkpoint 写 `{persist_dir}/checkpoints.db`
- `thread_id` 用 `{project_id}:ch{chapter_number}` 确定性格式，重启可恢复
- `_checkpointer_cache` 按 persist_dir 缓存

## Chainlit（legacy）

```
new <name> [title] [genre] [--words N]
list
select <project_id>
chapters
write <chapter> <outline> [--words N]
```

`write` 支持 Human-in-the-loop（Approve/Reject）。主入口请用 Web UI。
