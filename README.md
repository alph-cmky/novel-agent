# Novel Agent

基于 LangGraph 的开源多 Agent 小说写作框架，支持短篇/中篇/长篇多种篇幅 — 支持 Human-in-the-loop。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## 架构

```mermaid
graph TD
    O[Orchestrator<br/>叙事策略] --> W[Writer<br/>创作章节]
    W --> E[Editor<br/>5维审查]
    E --> C[Continuity<br/>一致性审计]
    C -->|通过| WB[Worldbuilding<br/>实体提取]
    C -->|失败 + 有重试| OR[Orchestrator Review<br/>反馈分析]
    C -->|失败 + 重试耗尽| WB
    OR -->|重写指导| W
    WB --> H[Human Review<br/>interrupt⏸️]
    H -->|批准| DONE((完成))
    H -->|拒绝| OR

    style O fill:#4a90d9,color:#fff
    style W fill:#7b4dd3,color:#fff
    style E fill:#e6a23c,color:#fff
    style C fill:#e6a23c,color:#fff
    style OR fill:#4a90d9,color:#fff
    style WB fill:#67c23a,color:#fff
    style H fill:#f56c6c,color:#fff
    style DONE fill:#909399,color:#fff
```

**7 节点 + 反馈闭环 + Human-in-the-loop：**

| 节点 | 职责 | 模型层级 |
|------|------|----------|
| Orchestrator | 叙事阶段分析、章节策略、篇幅感知节奏 | Budget |
| Writer | 章节创作 + 去 AI 味规则 + 工具调用 | Quality |
| Editor | 5 维度审查（节奏、AI 味、对话、逻辑、文笔） | Budget |
| Continuity | 跨章节一致性审计（角色、时间线、世界观规则） | Budget |
| Orchestrator Review | 分析失败报告，生成具体重写指导 | Budget |
| Worldbuilding | 实体提取、冲突检测、持久化到 SQLite | Budget |
| Human Review | **LangGraph `interrupt()`** — 暂停流水线，等待人类输入 | — |

### 双层记忆

- **短期记忆**：ContextCompressor 将前文章节压缩为摘要（约 40K token 阈值）
- **长期记忆**：ChromaDB 向量存储 + SQLite 结构化存储（项目、章节、世界观实体、伏笔）

### 篇幅支持

三种篇幅，各有节奏策略，每章字数可配置：

| 篇幅 | 默认字/章 | 叙事节奏 |
|------|----------|---------|
| `short` | 1500 | 快速推进，跳过 intro，3-5 章到达高潮 |
| `novella` | 3000 | 平衡发展，高潮在 60-70% 处 |
| `long` | 3000 | 渐进展开，多线并进，伏笔长线回收 |

## 设计决策

### 为什么用 StateGraph 而非 MessageGraph？

写小说需要**结构化状态**，而不仅仅是消息历史。每个 Agent 贡献不同类型的数据：草稿内容、评分、提取的实体、一致性报告。StateGraph 提供：

- **类型化共享状态**（`NovelState`），所有节点可读写，对比 MessageGraph 只能追加消息
- **条件路由**，根据评分、严重问题数量、重试次数决定下一步，线性消息传递无法做到
- **检查点**（`MemorySaver`），支持暂停/恢复和未来回放

### 为什么需要反馈闭环（而非线性 O→W→E→C→完成）？

没有反馈的话，失败的章节会用**同样的 prompt** 重写——Writer 永远不知道哪里出了问题。Orchestrator Review 节点分析 Editor/Continuity 报告，生成具体可执行的重写指导：

```
失败 → Orchestrator 分析："角色语气不一致" →
  Writer 收到："第1章设定主角是愤世嫉俗的性格，这一章突然变得乐观积极。
  重写时保持愤世嫉俗的语气，让转变有铺垫。"
```

这就是 Agent 和脚本的区别：系统在重试之前**推理失败原因**。

### 为什么用 `interrupt()` 实现 Human-in-the-loop？

`human_review_node` 使用 LangGraph 的 `interrupt()` **暂停**图执行。调用方（CLI 或 Chainlit）捕获 `GraphInterrupt`，向人类展示草稿和报告，然后用 `Command(resume=feedback)` 恢复执行。

这个模式展示了：
- **非阻塞图执行**——图不是轮询等待，而是挂起
- **结构化反馈**——人类输入回流到 Agent 流水线（Orchestrator Review → Writer）
- **双界面**——同一个图、同一个 interrupt，不同 UI（CLI 交互式 vs. Chainlit 按钮式）

### 为什么选 ChromaDB 做长期记忆？

- **原生向量存储**：章节以向量形式存储，支持语义检索。Writer 的 `search_context` 工具通过相似度搜索查询"角色 X 在前文章节发生了什么"
- **零依赖**：嵌入式模式，无需独立服务，随 Python 进程运行
- **取舍**：不如 pgvector/Weaviate 适合 1000+ 章的超长篇，但目标场景完全够用。`hosted` 扩展组提供了 pgvector 迁移路径

### 为什么双模型路由？

创作需要强模型（GPT-4、Claude）。结构化分析（审查、一致性检查、实体提取）用便宜模型（DeepSeek、GPT-4o-mini）即可。`ModelRouter` 按任务分配：

| TaskClass | 模型 | Temperature | 用途 |
|-----------|------|-------------|------|
| `CREATIVE` | `QUALITY_MODEL` | 0.9 | Writer |
| `STRUCTURAL` | `BUDGET_MODEL` | 0.3 | Orchestrator |
| `REVIEW` | `BUDGET_MODEL` | 0.1 | Editor, Continuity |
| `EXTRACTION` | `BUDGET_MODEL` | 0.1 | Worldbuilding |

关键设计：`resolve()` **每次调用时**读取环境变量，而非初始化时——支持运行时切换模型无需重启。

### 为什么 3 层输出校验？

LLM 的 JSON 输出不可靠。校验流水线（`schema/parser.py` + `schema/validator.py`）处理：
1. **直接解析** — 合法的 JSON
2. **Markdown 提取** — ``` 代码块中的 JSON（instruct 模型常见）
3. **正则兜底** — 从非结构化文本中提取键值对
4. **强制转换** — 修复常见问题（字符串分数 → int、缺少列表包装）
5. **默认值** — 缺失字段填入安全默认值

### 为什么 Writer 要调工具？

Writer 不只是生成文本——它使用 `search_context` 工具在写作前查询 ChromaDB 获取相关前文片段。这对长篇小说至关重要：Writer 无法在上下文中容纳 50+ 章内容，但可以检索当前章节需要的关键信息。

## 快速开始

### 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- OpenAI 兼容 API（OpenAI、DeepSeek 等）

### 安装

```bash
git clone https://github.com/your-org/novel-agent.git
cd novel-agent
uv sync
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API key 和模型偏好
```

### CLI 使用

```bash
# 初始化项目（含篇幅设置）
novel-agent init -n "MyNovel" -t "我的小说" -g "都市" -l long -w 3000

# 完整流水线生成（O→W→E→C→WB→Human Review）
novel-agent write -c 1 -o "主角穿越到异世界，发现拥有系统"

# 单章覆盖字数
novel-agent write -c 1 -o "战斗场景" -w 5000

# 快速生成（仅 Writer，跳过审查）
novel-agent quick -c 1 -o "主角穿越到异世界"

# 查看 trace
novel-agent trace ls
novel-agent trace show traces/trace-xxx.json
```

### Human-in-the-loop

运行 `novel-agent write` 时，流水线在 Human Review 处暂停：

```
============================================================
  HUMAN REVIEW — Chapter 1
============================================================
  Editor: 85/100  |  Continuity: 92/100
  Retries: 0

  ── Draft Preview ──
  [草稿预览...]

  Approve or reject? (approve/reject) [a]:
```

输入 `a`/`approve` 批准，或 `r`/`reject` 并附修改意见触发带指导的重写。

### Web UI

```bash
chainlit run novel_agent/api/app.py
```

打开 http://localhost:8000 ，通过按钮点击（而非 CLI 输入）进行审批。

### Docker

```bash
docker compose up chainlit    # Web UI
docker compose run novel-agent write -c 1 -o "大纲"  # CLI
```

## 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | API 密钥 | - |
| `OPENAI_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `BUDGET_MODEL` | 结构化/审查工作模型 | `deepseek-chat` |
| `QUALITY_MODEL` | 创意写作模型 | 同 BUDGET |

## 项目结构

```
novel_agent/
├── agents/              # 5 个专用 Agent + 基类
│   ├── base.py          # BaseAgent（工具调用循环）
│   ├── writer.py        # 章节创作（含 search_context 工具）
│   ├── editor.py        # 5 维审查 + DetectAiFlavorTool
│   ├── continuity.py    # 跨章节一致性审计
│   ├── orchestrator.py  # 叙事策略 + 反馈分析
│   └── worldbuilding.py # 实体提取 + 冲突检测
├── graph/               # LangGraph StateGraph
│   ├── state.py         # NovelState TypedDict
│   └── chapter.py       # 7 节点流水线（反馈闭环 + HITL）
├── memory/              # 双层记忆系统
│   ├── compressor.py    # ContextCompressor（40K token 阈值）
│   └── embeddings.py    # ChromaDB 向量存储
├── storage/             # SQLite 持久化
│   ├── models.py        # Schema + 迁移
│   └── manager.py       # ProjectManager（章节、实体管理）
├── schema/              # 输出校验边界
│   ├── models.py        # 所有 Agent 输出的 Pydantic 模型
│   ├── parser.py        # JSON 解析器（3 层兜底）
│   └── validator.py     # OutputValidator（3 层强制转换）
├── routing/             # 模型路由（双模型、运行时读环境变量）
├── trace/               # JSON trace 采集 + Rich CLI 查看器
├── tools/               # MCP 兼容工具协议
├── style/               # AI 味检测引擎（30+ 规则模式）
├── api/                 # Chainlit Web UI（支持 HITL 按钮）
└── cli/                 # Click CLI（支持交互式人工审批）
```

## License

MIT
