# Cleanup Audit — Pre-production Architecture Reset

Baseline: `aef2420` → Round 1 (Phase B–F, 已全部落地) → Round 2 (Contract Reset, 本文档)

## Round 1 结果（已验证 ✅）

| Symbol | Action | Status |
|---|---|---|
| `ContextCompressor` / `memory/compressor.py` | DELETE | ✅ REMOVED |
| `_story_arc` tracking | DELETE | ✅ REMOVED |
| Writer V1 prompt / `prompt_profile` | DELETE | ✅ REMOVED |
| `primary_storyline` | DELETE | ✅ REMOVED (Phase E) |
| Editor legacy flat scores | DELETE | ✅ REMOVED (Phase E) |
| `_migrate()` / `backfill_world_relations` | DELETE | ✅ REMOVED (Phase D) — CREATE TABLE 即最终态 |
| 旧 Style 路径（`style/ai_flavor.py`、`tools/style.py`、`tools/detect_ai_flavor.py`、`context/compiler.py`、`routing/`） | CONFIRMED | ✅ REMOVED (Phase F) |
| State 派生 context 字段（7 个平铺字段） | CONSOLIDATE | ✅ REMOVED — 只保留 `context_packet` |
| API `retry_count` / `workflow_version` 参数 | DELETE | ✅ REMOVED |

## Round 2 结果（Contract Reset，已验证 ✅）

| Item | Action | Status |
|---|---|---|
| `ContextPacket.to_state()` 平铺 + nested 双重表达 | DELETE 平铺 | ✅ REMOVED — `to_state()` 只返回 `{"context_packet": ...}` 单键 |
| `packet_hash` / `sources` / `token_budget` 字段 | DELETE | ✅ REMOVED — 全仓 0 消费方（仅测试自断言），属 observability/derived，不进 checkpoint |
| `ContextCompiler.compile()` 全量 `get_all_world_entities` 查询 | DELETE | ✅ REMOVED — entities 仅用于已删除的 sources 统计，省一次全量 DB 读 |
| `rewrite_instructions`（Writer 参数 + chapter.py 字符串装配） | RENAME | ✅ → `improvement_plan: dict` 直传 Writer；`_format_improvement_plan` 下沉到 `WriterAgent`，Graph 层不再解释 plan |
| 进化指导措辞「必须输出完整正文/严禁压缩」 | REWORD | ✅ 改为「输出完整正文 + 只修改必要部分 + 字数不得缩水」——为局部 revision 铺路 |
| Orchestrator fallback `ending_type="cliffhanger"` | FIX | ✅ → `natural_continuation` — 解析失败不再回流章章钩子模式（回归测试锁定） |
| `narrative_mode` None=旧项目 legacy 语义 | DELETE | ✅ 注释/docstring 改为 None=未配置 |
| `StyleProfile` 空壳（无字段的"未来预留"对象） | DELETE | ✅ REMOVED — 全仓 0 消费方 |
| `existing_world_entities` State 字段 | DELETE | ✅ REMOVED — worldbuilding_node 改为自查 DB（与 existing_fs 同一 mgr），消除 DB 复制缓存 |
| orchestrator_node 双 `ProjectManager` 实例 | FIX | ✅ 复用单一 mgr（chapters + foreshadowings） |
| `get_all_chapters()` 全量读取 + Python 过滤 | FIX | ✅ → `get_recent_chapters(project_id, before, limit)` SQL `ORDER BY ... DESC LIMIT` + `count_chapters()`（回归测试锁定） |
| README ContextCompressor / 迁移 / Scene-first 默认值描述 | FIX | ✅ 已修正为 ContextCompiler 投影 / CREATE TABLE 即最终态 / Chapter-first 默认 |

## Round 2 审计核实但**不删**的项（含原因）

| Item | 结论 | 原因 |
|---|---|---|
| Orchestrator 19 个 chapter_strategy 输出字段 | KEEP ALL | reachability audit 实测：`tension_profile`/`scene_composition`/`character_emotional_state`/`character_arcs`/`foreshadowing_management`(high-risk)/`storyline_intersection`/`ending_tone`/`stage_boundary`/`climax_sequence` 等全部有 Writer 消费方（`_format_global/conditional/auxiliary_section` 三层 formatter）。按「consumer=none 才删」标准无一合格 |
| `writing_runs.retry_count`（DB 列） | KEEP | `run_service.py` 在 resume 时递增（run lifecycle operational metadata）；`outbox_events.retry_count` 参与 worker 重试控制（`retry_count < max` 门控）。非旧 linear retry 残留 |
| `scene_drafts`（State 字段） | KEEP | SSE 层 `result.get("scene_drafts")` 用于 scenes manifest 持久化（`save_scenes`）；quality gate 场景完整性检查消费。非纯临时 accumulator |
| `style_report` / `quality_gate_report` / `quality_guard_report`（State 字段） | KEEP | editor_node 产出 → evolution 消费，跨节点真实依赖 |
| `StyleIssue.extra="allow"` | KEEP | 行为设计（evidence 字段可扩展），非瘦身项 |
| `compile_for_run` / `context_metrics` / `estimate_tokens` | KEEP | run lifecycle API + 可观测性 util，均有测试覆盖 |
| `ContextCompiler` 全量 foreshadowings/story_events 读取 | DEFER | 属 task-aware retrieval 长篇规模优化（S2 下一阶段），涉及 retrieval 策略设计，本轮只做已明确的死查询删除 |

## 后续建议（不属于本轮 Contract Reset）

1. **Context Minimality 第二阶段**：`get_foreshadowings` / `get_story_events` 从全量读取改为 relevance query（长篇 100+ 章的规模性优化）
2. **Orchestrator prompt 收缩评估**：字段保留（都有消费方），但「每章输出全部 19 字段」的 prompt 要求可按 narrative_mode 进一步裁剪输出成本
3. **Paragraphizer / 局部 Revision**：进化措辞已铺路（「只修改必要部分」），实现需等 A/B 评测数据
