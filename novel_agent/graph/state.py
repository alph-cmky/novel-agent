"""Shared state that flows through the chapter-writing StateGraph.

v2: Recursive self-evolution architecture. No legacy retry/feedback fields.
"""

from typing import TypedDict


class NovelState(TypedDict, total=False):
    """贯穿一次章节创作的共享状态"""

    # ── 项目上下文 ──
    project_id: str
    writing_run_id: str
    chapter_number: int
    chapter_outline: str
    story_length: str
    target_chapter_words: int
    # 叙事模式：linear / unit_arc / hybrid / multi_perspective / ensemble（None = 未配置）
    narrative_mode: str | None
    # 叙事视角：first_person / third_person_limited / third_person_omniscient / ...
    narrative_perspective: str

    # ── Agent 输出（当前轮的，每轮覆盖）──
    draft_content: str
    editor_report: dict
    continuity_report: dict
    worldbuilding_report: dict
    orchestrator_strategy: dict
    # Deterministic style analysis (0 LLM) — consumed by Editor and Evolution
    style_report: dict
    skip_orchestrator: bool
    skip_reviews: bool
    skip_worldbuilding: bool
    review_interval: int
    skip_evolution_enrichment: bool

    # ── Agent 注入的上下文（单一载体）──
    context_packet: dict
    scene_first: bool
    scene_plan: list[dict]
    scene_drafts: list[str]

    # ── 人类审阅 ──
    human_approved: bool
    human_feedback: dict
    evolution_human_rejects: int  # 人类拒绝次数（最多3次）
    human_review_exhausted: bool

    # ── 进化控制 ──
    evolution_max_rounds: int  # 最大轮次，默认 5
    evolution_convergence_threshold: float  # 收敛阈值，默认 3.0
    evolution_round: int  # 当前轮次 (0-based)
    evolution_version: int  # 当前版本号

    # ── 进化状态 ──
    # 每轮 {v, editor, ct, composite, dimensions, delta, focus}
    evolution_history: list[dict]
    evolution_candidates: list[dict]
    evolution_improvement_plan: dict  # 当前轮改进计划，Writer 消费
    # "" | "converged" | "regressed" | "plateau" | "max_rounds" | "ceiling"
    evolution_termination: str
    evolution_best_candidate_version: int
    quality_guard_report: dict
    quality_gate_report: dict
    deterministic_gate_first: bool

    # ── 存储路径 ──
    persist_dir: str

    # ── 可观测性 ──
    trace_id: str
    # LLM 成本观测（按章节累计；正常章节 enrichment ≈ 0）
    evolution_rule_plan_calls: int
    evolution_llm_enrichment_calls: int
    writer_model_calls: int
    writer_tool_calls: int
    writer_search_calls: int
    # 真实 token 消耗（provider usage_metadata；cached/reasoning 依赖 provider 暴露）
    orchestrator_input_tokens: int
    orchestrator_output_tokens: int
    orchestrator_cached_tokens: int
    orchestrator_reasoning_tokens: int
    writer_input_tokens: int
    writer_output_tokens: int
    writer_cached_tokens: int
    writer_reasoning_tokens: int
    editor_input_tokens: int
    editor_output_tokens: int
    editor_cached_tokens: int
    editor_reasoning_tokens: int
    continuity_input_tokens: int
    continuity_output_tokens: int
    continuity_cached_tokens: int
    continuity_reasoning_tokens: int
    worldbuilding_input_tokens: int
    worldbuilding_output_tokens: int
    worldbuilding_cached_tokens: int
    worldbuilding_reasoning_tokens: int
    evolution_input_tokens: int
    evolution_output_tokens: int
    evolution_cached_tokens: int
    evolution_reasoning_tokens: int
    # per-role model_calls（writer 已有 writer_model_calls；C-2 cost attribution）
    orchestrator_model_calls: int
    editor_model_calls: int
    continuity_model_calls: int
    worldbuilding_model_calls: int
    evolution_model_calls: int
    # per-role latency（C-2 cost attribution；秒级，节点 wall time）
    orchestrator_latency_seconds: float
    writer_latency_seconds: float
    editor_latency_seconds: float
    continuity_latency_seconds: float
    worldbuilding_latency_seconds: float
    evolution_latency_seconds: float
