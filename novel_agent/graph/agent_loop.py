"""Agent loop runner — C5 architecture with S1 conditional review.

    Orchestrator → Writer agent loop [generate → analyze_style → humanize_passage]
    → deterministic Hard Gate
         PASS → SKIP Editor + Continuity (telemetry recorded)
         FAIL → Editor → (verdict=rewrite → re-enter Writer ≤ max_retries) → Continuity
    → Worldbuilding → output for human approve/reject

The Writer loop handles generation + de-AI flavor (analyze_style + humanize_passage).
Editor/Continuity run only when the Hard Gate fails. Worldbuilding always runs.
If Editor verdict is "rewrite", the loop re-enters with Editor feedback (max 2 retries).
"""

from __future__ import annotations

import time
from typing import Any

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.continuity import ContinuityAgent
from novel_agent.agents.editor import EditorAgent
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.agents.worldbuilding import WorldbuildingAgent
from novel_agent.agents.writer import WriterAgent, strip_writer_preamble
from novel_agent.config import DEFAULT_MAX_TOKENS
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.model_router import TaskClass
from novel_agent.services.context import ContextCompiler
from novel_agent.services.quality import QualityService
from novel_agent.style.analyzer import StyleAnalyzer
from novel_agent.tools.humanize import HumanizeTool
from novel_agent.tools.search import SearchContextTool
from novel_agent.tools.style_check import StyleCheckTool


class EmptyDraftError(RuntimeError):
    """Writer finished without producing usable chapter text."""


def _config_for(task: TaskClass) -> AgentConfig:
    """Create AgentConfig from the model router's decision."""
    from novel_agent.model_router import ModelRouter

    router = ModelRouter()
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


def _skipped_editor_report(reason: str = "hard_gate_passed") -> dict:
    return {
        "skipped": True,
        "reason": reason,
        "verdict": "accept",
        "overall_score": None,
        "issues": [],
    }


def _skipped_continuity_report(reason: str = "hard_gate_passed") -> dict:
    return {
        "skipped": True,
        "reason": reason,
        "overall_score": None,
        "issues": [],
    }


async def run_agent_loop(
    state: dict[str, Any],
    *,
    max_rounds: int = 8,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run the C5 agent loop with S1 conditional Editor/Continuity.

    Args:
        state: Initial state dict (same shape as old LangGraph initial_state).
            May include ``revision_feedback`` from human reject / resume.
        max_rounds: Max tool-calling rounds for the Writer loop.
        max_retries: Max Editor-triggered re-entry into the loop.

    Returns:
        Final state dict compatible with eval adapter expectations.

    Raises:
        EmptyDraftError: If Writer returns empty text after the loop (and any
            Editor-triggered rewrite retries), so empty is never a success chapter.
    """
    persist_dir = state.get("persist_dir", "./novel-data")
    project_id = state.get("project_id", "")
    chapter_number = state.get("chapter_number", 1)
    target_words = state.get("target_chapter_words", 3000)
    narrative_mode = state.get("narrative_mode")
    narrative_perspective = state.get("narrative_perspective", "")
    outline = state.get("chapter_outline", "")
    revision_feedback = (state.get("revision_feedback") or "").strip() or None

    # ── 1. Orchestrator ──
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
        chapter_outline=outline,
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

    # ── 2. Writer agent loop (generate + de-AI) ──
    max_tokens = max(DEFAULT_MAX_TOKENS, int(target_words * 3))
    writer_config = _config_for(TaskClass.CREATIVE)
    writer_config.max_tokens = max_tokens

    store = _get_chapter_store(persist_dir)
    writer_packet = ContextCompiler.for_writer(full_packet) if full_packet else None

    # C5: loop only has generation + de-AI tools.
    # Editor/Continuity are conditional (S1); Worldbuilding is a fixed node after.
    loop_tools: list = [
        StyleCheckTool(),
        HumanizeTool(config=_config_for(TaskClass.REVIEW)),
    ]
    if store and project_id:
        loop_tools.append(SearchContextTool(store, project_id))

    writer = WriterAgent(
        config=writer_config,
        chapter_store=store,
        project_id=project_id,
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
        agent_loop_tools=loop_tools,
    )

    content, _ = await writer.write_with_loop(
        chapter_number=chapter_number,
        outline=outline,
        context_packet=writer_packet,
        target_chapter_words=target_words,
        orchestrator_strategy=strategy,
        max_rounds=max_rounds,
        revision_feedback=revision_feedback,
    )
    content = strip_writer_preamble(content)
    print(
        f"  [AgentLoop] Writer: {len(content)} chars, "
        f"tokens: {writer.input_tokens}/{writer.output_tokens}, "
        f"tools: {writer.tool_call_counts}"
        + (f", revision_feedback={'yes' if revision_feedback else 'no'}")
    )

    if not content.strip():
        raise EmptyDraftError(
            f"Writer returned empty draft for chapter {chapter_number} "
            f"(after tool loop; refusing to treat empty as a successful chapter)"
        )

    # ── 3. Deterministic Hard Gate (real gate under S1) ──
    quality_gate_report = QualityService.check_draft_hard_gates(
        content, target_words=target_words, chapter_outline=outline
    )
    gate_passed = bool(quality_gate_report.get("passed"))
    print(f"  [AgentLoop] QualityGate: {'PASS' if gate_passed else 'FAIL'}")

    editor_skipped = False
    continuity_skipped = False
    editor_report: dict
    continuity_report: dict
    retries = 0
    _e_latency = 0.0
    _c_latency = 0.0

    # Token counters for skipped path (agents not constructed)
    editor_input_tokens = 0
    editor_output_tokens = 0
    editor_cached_tokens = 0
    editor_reasoning_tokens = 0
    editor_model_calls = 0
    continuity_input_tokens = 0
    continuity_output_tokens = 0
    continuity_cached_tokens = 0
    continuity_reasoning_tokens = 0
    continuity_model_calls = 0

    if gate_passed:
        # S1: Hard Gate PASS → skip Editor + Continuity
        editor_skipped = True
        continuity_skipped = True
        editor_report = _skipped_editor_report()
        continuity_report = _skipped_continuity_report()
        print("  [AgentLoop] S1: Hard Gate PASS → skip Editor + Continuity")
    else:
        # ── 4. Editor (FAIL path only) ──
        editor = EditorAgent(config=_config_for(TaskClass.REVIEW))
        editor_packet = ContextCompiler.for_editor(full_packet) if full_packet else None

        async def _run_editor(draft: str) -> dict:
            style_report = StyleAnalyzer().analyze(draft).model_dump()
            report, _ = await editor.review(
                chapter_number=chapter_number,
                draft_content=draft,
                narrative_mode=narrative_mode,
                style_report=style_report,
                context_packet=editor_packet,
            )
            return report

        _e_t0 = time.monotonic()
        editor_report = await _run_editor(content)
        _e_latency = time.monotonic() - _e_t0
        print(
            f"  [AgentLoop] Editor: score={editor_report.get('overall_score', '?')}, "
            f"verdict={editor_report.get('verdict', '?')}"
        )

        # ── 5. Editor-triggered retry (re-enter loop with feedback) ──
        while editor_report.get("verdict") == "rewrite" and retries < max_retries:
            retries += 1
            print(f"  [AgentLoop] Editor retry {retries}/{max_retries}")

            feedback = editor_report.get("issues") or []
            feedback_text = "\n".join(
                f"- [{i.get('dimension', '?')}] {i.get('description', '')}" for i in feedback[:5]
            )

            content, _ = await writer.write_with_loop(
                chapter_number=chapter_number,
                outline=outline,
                context_packet=writer_packet,
                target_chapter_words=target_words,
                orchestrator_strategy=strategy,
                max_rounds=max_rounds,
                revision_feedback=feedback_text,
            )
            content = strip_writer_preamble(content)
            print(f"  [AgentLoop] Writer retry {retries}: {len(content)} chars")

            if not content.strip():
                raise EmptyDraftError(
                    f"Writer returned empty draft on Editor rewrite retry "
                    f"{retries}/{max_retries} for chapter {chapter_number}"
                )

            # Re-check hard gate after rewrite; if now PASS, stop review loop
            quality_gate_report = QualityService.check_draft_hard_gates(
                content, target_words=target_words, chapter_outline=outline
            )
            gate_passed = bool(quality_gate_report.get("passed"))
            print(
                f"  [AgentLoop] QualityGate after rewrite {retries}: "
                f"{'PASS' if gate_passed else 'FAIL'}"
            )
            if gate_passed:
                editor_report = {
                    **editor_report,
                    "verdict": "accept",
                    "note": "hard_gate_passed_after_rewrite",
                }
                break

            _e_t0 = time.monotonic()
            editor_report = await _run_editor(content)
            _e_latency += time.monotonic() - _e_t0
            print(
                f"  [AgentLoop] Editor retry {retries}: "
                f"score={editor_report.get('overall_score', '?')}, "
                f"verdict={editor_report.get('verdict', '?')}"
            )

        editor_input_tokens = editor.input_tokens
        editor_output_tokens = editor.output_tokens
        editor_cached_tokens = editor.cached_tokens
        editor_reasoning_tokens = editor.reasoning_tokens
        editor_model_calls = editor.model_calls

        # ── 6. Continuity (FAIL path only; skip if rewrite recovered to PASS) ──
        if gate_passed:
            continuity_skipped = True
            continuity_report = _skipped_continuity_report("hard_gate_passed_after_rewrite")
            print("  [AgentLoop] S1: Hard Gate PASS after rewrite → skip Continuity")
        else:
            continuity = ContinuityAgent(
                config=_config_for(TaskClass.REVIEW),
                chapter_store=store,
                project_id=project_id,
            )
            continuity_packet = ContextCompiler.for_continuity(full_packet) if full_packet else None
            _c_t0 = time.monotonic()
            continuity_report, _ = await continuity.audit(
                chapter_number=chapter_number,
                draft_content=content,
                narrative_mode=narrative_mode,
                context_packet=continuity_packet,
            )
            _c_latency = time.monotonic() - _c_t0
            continuity_input_tokens = continuity.input_tokens
            continuity_output_tokens = continuity.output_tokens
            continuity_cached_tokens = continuity.cached_tokens
            continuity_reasoning_tokens = continuity.reasoning_tokens
            continuity_model_calls = continuity.model_calls
            print(f"  [AgentLoop] Continuity: score={continuity_report.get('overall_score', '?')}")

    if not content.strip():
        raise EmptyDraftError(
            f"Empty draft remaining after review path for chapter {chapter_number}"
        )

    # ── 7. Worldbuilding (always runs) ──
    existing_entities: list = []
    existing_foreshadowings = (
        [{"description": u, "status": "open"} for u in unresolved] if unresolved else []
    )
    wb_existing_load_error: str | None = None
    if mgr and project_id:
        try:
            # Correct manager API (get_world_entities does not exist)
            existing_entities = mgr.get_all_world_entities(project_id) or []
        except Exception as exc:
            wb_existing_load_error = f"{type(exc).__name__}: {exc}"
            print(f"  [AgentLoop] Worldbuilding load existing entities failed: {exc}")

    worldbuilding = WorldbuildingAgent(
        config=_config_for(TaskClass.EXTRACTION),
        existing_entities=existing_entities,
        existing_foreshadowings=existing_foreshadowings,
    )
    _wb_t0 = time.monotonic()
    worldbuilding_report, _ = await worldbuilding.extract(
        chapter_number=chapter_number,
        draft_content=content,
        narrative_mode=narrative_mode,
    )
    _wb_latency = time.monotonic() - _wb_t0

    new_entities = worldbuilding_report.get("new_entities") or []
    updated_entities = worldbuilding_report.get("updated_entities") or []
    entity_count = len(new_entities) + len(updated_entities)
    print(
        f"  [AgentLoop] Worldbuilding: new_entities={len(new_entities)}, "
        f"updated_entities={len(updated_entities)}, "
        f"existing_loaded={len(existing_entities)}"
    )

    wb_warnings: list[str] = []
    if wb_existing_load_error:
        wb_warnings.append(f"existing_entities_load_failed: {wb_existing_load_error}")
    if entity_count == 0:
        warning = (
            f"Worldbuilding extracted 0 entities for chapter {chapter_number} "
            f"(existing_loaded={len(existing_entities)}); not silent"
        )
        wb_warnings.append(warning)
        print(f"  [AgentLoop] WARNING: {warning}")
        worldbuilding_report = {
            **worldbuilding_report,
            "zero_entities_warning": warning,
        }

    # ── 8. Assemble final state ──
    return {
        "draft_content": content,
        "orchestrator_strategy": strategy,
        "context_packet": full_packet,
        "quality_gate_report": quality_gate_report,
        "quality_gate_passed": gate_passed,
        "editor_report": editor_report,
        "continuity_report": continuity_report,
        "editor_skipped": editor_skipped,
        "continuity_skipped": continuity_skipped,
        "worldbuilding_report": worldbuilding_report,
        "worldbuilding_warnings": wb_warnings,
        "human_approved": None,
        "chapter_number": chapter_number,
        "project_id": project_id,
        "writing_run_id": state.get("writing_run_id", ""),
        "loop_retries": retries,
        "revision_feedback_consumed": bool(revision_feedback),
        "scene_plan": [],
        "scene_drafts": [],
        # Token accounting (per-role)
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
        "writer_latency_seconds": time.monotonic() - _t0 - _orch_latency,
        "writer_tool_calls": sum(writer.tool_call_counts.values()),
        "writer_search_calls": writer.tool_call_counts.get("search_context", 0),
        "editor_input_tokens": editor_input_tokens,
        "editor_output_tokens": editor_output_tokens,
        "editor_cached_tokens": editor_cached_tokens,
        "editor_reasoning_tokens": editor_reasoning_tokens,
        "editor_model_calls": editor_model_calls,
        "editor_latency_seconds": _e_latency,
        "continuity_input_tokens": continuity_input_tokens,
        "continuity_output_tokens": continuity_output_tokens,
        "continuity_cached_tokens": continuity_cached_tokens,
        "continuity_reasoning_tokens": continuity_reasoning_tokens,
        "continuity_model_calls": continuity_model_calls,
        "continuity_latency_seconds": _c_latency,
        "worldbuilding_input_tokens": worldbuilding.input_tokens,
        "worldbuilding_output_tokens": worldbuilding.output_tokens,
        "worldbuilding_cached_tokens": worldbuilding.cached_tokens,
        "worldbuilding_reasoning_tokens": worldbuilding.reasoning_tokens,
        "worldbuilding_model_calls": worldbuilding.model_calls,
        "worldbuilding_latency_seconds": _wb_latency,
    }
