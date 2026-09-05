"""Shared state for the C5 agent loop architecture.

No evolution subgraph, no scene-first, no LangGraph checkpoint fields.
The agent loop runner returns a dict with these keys.
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
    narrative_mode: str | None
    narrative_perspective: str

    # ── Agent 输出 ──
    draft_content: str
    editor_report: dict
    continuity_report: dict
    worldbuilding_report: dict
    orchestrator_strategy: dict
    style_report: dict
    quality_gate_report: dict

    # ── 上下文 ──
    context_packet: dict
    skip_orchestrator: bool

    # ── 人类审阅 ──
    human_approved: bool
    human_feedback: dict

    # ── Agent loop 控制 ──
    loop_retries: int

    # ── 存储路径 ──
    persist_dir: str

    # ── 可观测性 ──
    trace_id: str
    writer_model_calls: int
    writer_tool_calls: int
    writer_search_calls: int
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
    orchestrator_model_calls: int
    editor_model_calls: int
    continuity_model_calls: int
    worldbuilding_model_calls: int
    orchestrator_latency_seconds: float
    writer_latency_seconds: float
    editor_latency_seconds: float
    continuity_latency_seconds: float
    worldbuilding_latency_seconds: float
