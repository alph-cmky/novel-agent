"""Agent loop runner — replaces LangGraph with a single Writer tool-calling loop.

Architecture (experiment/full-agent-loop):
    Orchestrator → Writer agent loop [
        generate → analyze_style → humanize_passage → editor_review
        → check_continuity → worldbuilding_extract → revise → finalize
    ] → deterministic QualityGate → output

The Writer drives the entire generate-check-revise cycle via tool calls.
No fixed graph nodes, no evolution subgraph. Bounded by max_rounds.

This is an experiment branch — the production graph (chapter.py) is untouched.
"""

from __future__ import annotations

import time
from typing import Any

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.agents.writer import WriterAgent, strip_writer_preamble
from novel_agent.config import DEFAULT_MAX_TOKENS
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.model_router import TaskClass
from novel_agent.services.context import ContextCompiler
from novel_agent.services.quality import QualityService
from novel_agent.tools.continuity import CheckContinuityTool
from novel_agent.tools.editor_review import EditorReviewTool
from novel_agent.tools.humanize import HumanizeTool
from novel_agent.tools.style_check import StyleCheckTool
from novel_agent.tools.worldbuilding_extract import WorldbuildingExtractTool


def _config_for(task: TaskClass) -> AgentConfig:
    """Create AgentConfig from the model router's decision."""
    from novel_agent.model_router import router

    route = router.resolve(task)
    kwargs: dict = {"model": route.model, "temperature": route.temperature}
    kwargs["max_tokens"] = 8192 if task is TaskClass.EXTRACTION else 4096
    if route.api_key:
        kwargs["api_key"] = route.api_key
    if route.base_url:
        kwargs["base_url"] = route.base_url
    kwargs["is_reasoning"] = route.is_reasoning
    return AgentConfig(**kwargs)


def _get_chapter_store(persist_dir: str) -> ChapterStore | None:
    try:
        return ChapterStore(persist_dir.rstrip("/") + "/chroma_data")
    except Exception:
        return None


async def run_agent_loop(
    state: dict[str, Any],
    *,
    max_rounds: int = 12,
) -> dict[str, Any]:
    """Run the full agent loop: Orchestrator → Writer loop → QualityGate.

    Args:
        state: Initial state dict (same shape as LangGraph initial_state).
        max_rounds: Max tool-calling rounds for the Writer loop.

    Returns:
        Final state dict with draft_content, editor_report, etc.
        Compatible with what the eval adapter expects from outcome.values.
    """
    persist_dir = state.get("persist_dir", "./novel-data")
    project_id = state.get("project_id", "")
    chapter_number = state.get("chapter_number", 1)
    target_words = state.get("target_chapter_words", 3000)
    narrative_mode = state.get("narrative_mode")
    narrative_perspective = state.get("narrative_perspective", "")

    # ── 1. Orchestrator (unchanged from graph) ──
    orchestrator = OrchestratorAgent(config=_config_for(TaskClass.STRUCTURAL))

    previous_chapters: list[dict] = []
    total_chapters = 0
    unresolved: list[str] = []
    mgr = None
    if project_id:
        try:
            from novel_agent.storage.manager import ProjectManager

            mgr = ProjectManager(persist_dir)
            previous_chapters = mgr.get_recent_chapters(project_id, before=chapter_number, limit=5)
            previous_chapters.reverse()
            total_chapters = mgr.count_chapters(project_id, before=chapter_number)
            relevant_fs = mgr.get_relevant_foreshadowings(project_id, chapter_number)
            unresolved = [
                f"[第{f.get('planted_chapter', '?')}章] {f.get('description', '')}"
                for f in relevant_fs
            ]
        except Exception as exc:
            print(f"  [AgentLoop] Orchestrator 加载前文失败: {exc}")

    full_packet = dict(state.get("context_packet") or {})
    if unresolved:
        full_packet["unresolved_foreshadowings"] = unresolved

    _t0 = time.monotonic()
    strategy = await orchestrator.analyze(
        chapter_number=chapter_number,
        chapter_outline=state.get("chapter_outline", ""),
        previous_chapters=previous_chapters,
        story_length=state.get("story_length", "long"),
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
        arc_summary="",
        context_packet=ContextCompiler.for_orchestrator(full_packet),
        total_chapters=total_chapters,
        scene_first=False,
    )
    _orch_latency = time.monotonic() - _t0

    context_needed = strategy.get("context_needed", {})
    if context_needed and mgr and project_id:
        compiler = ContextCompiler(mgr)
        full_packet = compiler.apply_context_needed(
            full_packet, context_needed, project_id, chapter_number
        )

    print(
        f"  [AgentLoop] Orchestrator: stage={strategy.get('narrative_stage', '?')}, "
        f"tokens: {orchestrator.input_tokens}/{orchestrator.output_tokens}"
    )

    # ── 2. Writer agent loop ──
    max_tokens = max(DEFAULT_MAX_TOKENS, int(target_words * 3))
    writer_config = _config_for(TaskClass.CREATIVE)
    writer_config.max_tokens = max_tokens

    store = _get_chapter_store(persist_dir)
    shared: dict[str, Any] = {}

    writer_packet = ContextCompiler.for_writer(full_packet) if full_packet else None

    loop_tools: list = [
        StyleCheckTool(),
        HumanizeTool(config=_config_for(TaskClass.REVIEW)),
        EditorReviewTool(
            config=_config_for(TaskClass.REVIEW),
            narrative_mode=narrative_mode,
            context_packet=ContextCompiler.for_editor(full_packet) if full_packet else None,
            shared=shared,
        ),
        WorldbuildingExtractTool(
            config=_config_for(TaskClass.EXTRACTION),
            existing_entities=[],
            existing_foreshadowings=unresolved
            and [{"description": u, "status": "open"} for u in unresolved]
            or [],
            narrative_mode=narrative_mode,
            shared=shared,
        ),
    ]
    if store and project_id:
        loop_tools.append(CheckContinuityTool(store, project_id))

    writer = WriterAgent(
        config=writer_config,
        chapter_store=store,
        project_id=project_id,
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
        agent_loop_tools=loop_tools,
    )

    _w_t0 = time.monotonic()
    content, _ = await writer.write_with_loop(
        chapter_number=chapter_number,
        outline=state.get("chapter_outline", ""),
        context_packet=writer_packet,
        target_chapter_words=target_words,
        orchestrator_strategy=strategy,
        max_rounds=max_rounds,
    )
    _w_latency = time.monotonic() - _w_t0

    content = strip_writer_preamble(content)
    print(
        f"  [AgentLoop] Writer: {len(content)} chars, "
        f"tokens: {writer.input_tokens}/{writer.output_tokens}, "
        f"tools: {writer.tool_call_counts}"
    )

    # ── 3. Deterministic QualityGate ──
    quality_gate_report = QualityService.check_draft_hard_gates(
        content,
        target_words=target_words,
        chapter_outline=state.get("chapter_outline", ""),
    )
    print(f"  [AgentLoop] QualityGate: {'PASS' if quality_gate_report['passed'] else 'FAIL'}")

    # ── 4. Assemble final state ──
    editor_report = shared.get("editor_report") or {"unavailable": True}
    worldbuilding_report = shared.get("worldbuilding_report") or {}

    return {
        "draft_content": content,
        "orchestrator_strategy": strategy,
        "context_packet": full_packet,
        "quality_gate_report": quality_gate_report,
        "editor_report": editor_report,
        "worldbuilding_report": worldbuilding_report,
        "continuity_report": {},
        "human_approved": None,
        "chapter_number": chapter_number,
        "project_id": project_id,
        "writing_run_id": state.get("writing_run_id", ""),
        "evolution_termination": "agent_loop",
        "evolution_history": [],
        "evolution_candidates": [],
        "evolution_best_candidate_version": None,
        "scene_plan": [],
        "scene_drafts": [],
        # Token accounting (per-role, compatible with eval adapter)
        "orchestrator_input_tokens": orchestrator.input_tokens,
        "orchestrator_output_tokens": orchestrator.output_tokens,
        "orchestrator_cached_tokens": orchestrator.cached_tokens,
        "orchestrator_reasoning_tokens": orchestrator.reasoning_tokens,
        "orchestrator_model_calls": orchestrator.model_calls,
        "orchestrator_latency_seconds": _orch_latency,
        "writer_input_tokens": writer.input_tokens,
        "writer_output_tokens": writer.output_tokens,
        "writer_cached_tokens": writer.cached_tokens,
        "writer_reasoning_tokens": writer.reasoning_tokens,
        "writer_model_calls": writer.model_calls,
        "writer_latency_seconds": _w_latency,
        "writer_tool_calls": sum(writer.tool_call_counts.values()),
        "writer_search_calls": writer.tool_call_counts.get("search_context", 0),
        # Other roles unused in agent loop
        "editor_input_tokens": 0,
        "editor_output_tokens": 0,
        "editor_cached_tokens": 0,
        "editor_reasoning_tokens": 0,
        "editor_model_calls": 0,
        "editor_latency_seconds": 0.0,
        "continuity_input_tokens": 0,
        "continuity_output_tokens": 0,
        "continuity_cached_tokens": 0,
        "continuity_reasoning_tokens": 0,
        "continuity_model_calls": 0,
        "continuity_latency_seconds": 0.0,
        "worldbuilding_input_tokens": 0,
        "worldbuilding_output_tokens": 0,
        "worldbuilding_cached_tokens": 0,
        "worldbuilding_reasoning_tokens": 0,
        "worldbuilding_model_calls": 0,
        "worldbuilding_latency_seconds": 0.0,
        "evolution_input_tokens": 0,
        "evolution_output_tokens": 0,
        "evolution_cached_tokens": 0,
        "evolution_reasoning_tokens": 0,
        "evolution_model_calls": 0,
        "evolution_latency_seconds": 0.0,
    }
