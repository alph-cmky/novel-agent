"""Shared state that flows through the chapter-writing StateGraph."""

from typing import TypedDict


class NovelState(TypedDict, total=False):
    """贯穿一次章节创作的共享状态"""

    # 项目上下文
    project_id: str
    chapter_number: int
    chapter_outline: str
    story_length: str
    target_chapter_words: int

    # Agent输出
    draft_content: str
    editor_report: dict
    continuity_report: dict
    worldbuilding_report: dict

    # 流程控制
    retry_count: int
    human_approved: bool
    human_feedback: dict  # {action: "approve"|"reject", comments: str, edited_text: str}
    rewrite_instructions: str  # Orchestrator 给 Writer 的重写指导

    # Orchestrator注入的上下文
    orchestrator_strategy: dict
    character_context: str
    world_context: str
    recent_summary: str
    unresolved_foreshadowings: list[str]

    # Worldbuilding上下文
    existing_world_entities: list[dict]

    # 存储路径
    persist_dir: str

    # 可观测性
    trace_id: str
