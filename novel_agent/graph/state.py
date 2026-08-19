"""Shared state that flows through the chapter-writing StateGraph.

v2: Evolution architecture. The old retry_count + linear feedback loop has been
removed; its fields are retained only for backward-compatible checkpoint loading.
"""

from typing import TypedDict


class NovelState(TypedDict, total=False):
    """贯穿一次章节创作的共享状态"""

    # ── 项目上下文 ──
    project_id: str
    chapter_number: int
    chapter_outline: str
    story_length: str
    target_chapter_words: int
    # 叙事模式：linear / unit_arc / hybrid / multi_perspective / ensemble（None = 旧项目）
    narrative_mode: str | None
    # 叙事视角：first_person / third_person_limited / third_person_omniscient / ...
    narrative_perspective: str

    # ── Agent 输出（当前轮的，每轮覆盖）──
    draft_content: str
    editor_report: dict
    continuity_report: dict
    worldbuilding_report: dict
    orchestrator_strategy: dict
    skip_orchestrator: bool
    skip_reviews: bool
    skip_worldbuilding: bool
    review_interval: int
    skip_evolution_enrichment: bool

    # ── Orchestrator 注入的上下文 ──
    character_context: str
    world_context: str
    recent_summary: str
    unresolved_foreshadowings: list[str]

    # ── Worldbuilding 上下文 ──
    existing_world_entities: list[dict]

    # ── 人类审阅 ──
    human_approved: bool
    human_feedback: dict
    evolution_human_rejects: int          # 人类拒绝次数（最多3次）

    # ── 进化控制 ──
    evolution_max_rounds: int             # 最大轮次，默认 5
    evolution_convergence_threshold: float  # 收敛阈值，默认 3.0
    evolution_round: int                  # 当前轮次 (0-based)
    evolution_version: int                # 当前版本号

    # ── 进化状态 ──
    # 每轮 {v, editor, ct, composite, dimensions, delta, focus}
    evolution_history: list[dict]
    evolution_improvement_plan: dict      # 当前轮改进计划，Writer 消费
    # "" | "converged" | "regressed" | "plateau" | "max_rounds" | "ceiling"
    evolution_termination: str
    evolution_best_version: int
    evolution_best_draft: str             # 最优版本完整文本（只在 state，最后落库）
    evolution_best_editor_report: dict
    evolution_best_continuity_report: dict
    evolution_best_quality_guard: dict
    evolution_best_worldbuilding_report: dict
    evolution_best_outline_coverage: float
    evolution_best_required_facts_missing: int
    quality_guard_report: dict

    # ── 存储路径 ──
    persist_dir: str

    # ── 可观测性 ──
    trace_id: str
