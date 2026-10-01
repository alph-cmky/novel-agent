"""Chapter runner — discourse contract, then one review board.

    Orchestrator (beats + discourse contract)
    → Writer writes the chapter, no tools
    → deterministic Hard Gate (fail rewrites before the board)
    → review board: timeline, discourse contract, Editor, Continuity
    → one revision brief, Writer rewrites ≤ max_retries
    → surface humanize only after structure, and only for clustered patterns
    → Worldbuilding → output

Writer is the only prose author. Reviewers do not rewrite.
"""

from __future__ import annotations

import asyncio
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
from novel_agent.services.context import (
    _FORESHADOW_LIMIT,
    _PROPERTY_CHARS,
    ContextCompiler,
    _prefer_foreshadowings,
    closing_repeats,
)
from novel_agent.services.continuity import ContinuityService
from novel_agent.services.discourse import (
    check_discourse_contract,
    direct_revision,
    ensure_discourse,
    needs_rewrite,
)
from novel_agent.services.quality import QualityService
from novel_agent.style.analyzer import StyleAnalyzer
from novel_agent.tools.humanize import HumanizeTool


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


def _wb_entities(entities: list[dict]) -> list[dict]:
    from novel_agent.storage.manager import _property_text

    clipped = []
    for entity in entities:
        name = entity.get("name")
        if not name:
            continue
        clipped.append(
            {
                "entity_type": entity.get("entity_type"),
                "name": name,
                "properties": _property_text(entity.get("properties") or "", _PROPERTY_CHARS),
            }
        )
    return clipped


def _timeline_feedback(report: dict) -> str:
    lines = []
    for item in report.get("findings") or []:
        if item.get("type") != "dead_character_reappeared":
            continue
        lines.append(
            f"- 角色「{item.get('subject', '')}」已在第{item.get('death_chapter', '?')}章死亡，"
            "本章不得让其活着出场。改写时删掉这个人的在场，不要只改措辞。"
        )
    return "\n".join(lines)


def _draft_timeline(packet: dict, draft: str, chapter_number: int) -> dict:
    events = packet.get("timeline_events") or []
    return ContinuityService.check_draft_against_timeline(
        events,
        draft,
        current_chapter=chapter_number,
    )


def _style_dump(draft: str) -> dict:
    report = StyleAnalyzer().analyze(draft)
    if hasattr(report, "model_dump"):
        data = report.model_dump()
        return data if isinstance(data, dict) else {}
    return {}


def _clustered_surface(style: dict) -> bool:
    """Surface rewrite only when the same pattern repeats. One hit is not enough.

    Paired dashes are a rhythm mark in this prose, not a cluster. The humanizer
    is told to keep them, so they must not open a rewrite.
    """
    issues = style.get("issues") or []
    banned_kinds = 0
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        label = f"{issue.get('pattern') or ''}{issue.get('phrase') or ''}"
        if "破折号" in label or label.strip() == "——":
            continue
        count = int(issue.get("count") or 0)
        kind = issue.get("type")
        if kind in {"banned_phrase", "sentence_pattern", "cliche"} and count >= 2:
            return True
        if kind == "banned_phrase" and count >= 1:
            banned_kinds += 1
    return banned_kinds >= 3


async def run_agent_loop(
    state: dict[str, Any],
    *,
    max_rounds: int = 8,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run the C5 agent loop with S1 conditional Editor/Continuity.

    Args:
        state: Initial state dict for one chapter run.
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
    mgr = None
    full_packet = dict(state.get("context_packet") or {})
    unresolved = list(full_packet.get("unresolved_foreshadowings") or [])
    if project_id:
        try:
            from novel_agent.storage.manager import ProjectManager

            mgr = ProjectManager(persist_dir)
            previous_chapters = mgr.get_chapter_numbers(project_id, before=chapter_number)
            total_chapters = mgr.count_chapters(project_id, before=chapter_number)
            if not unresolved:
                relevant_fs = _prefer_foreshadowings(
                    mgr.get_relevant_foreshadowings(project_id, chapter_number, limit=None),
                    _FORESHADOW_LIMIT,
                    full_packet.get("timeline_events") or [],
                )
                unresolved = [
                    f"[第{f.get('planted_chapter', '?')}章] {f.get('description', '')}"
                    for f in relevant_fs
                ]
                if unresolved:
                    full_packet["unresolved_foreshadowings"] = unresolved
        except Exception as exc:
            print(f"  [AgentLoop] Orchestrator 加载前文失败: {exc}")

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
    )
    _orch_latency = time.monotonic() - _t0
    strategy = ensure_discourse(strategy)

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

    # ── 2. Writer writes the chapter. Tools stay outside this call. ──
    max_tokens = max(DEFAULT_MAX_TOKENS, int(target_words * 3))
    writer_config = _config_for(TaskClass.CREATIVE)
    writer_config.max_tokens = max_tokens

    store = _get_chapter_store(persist_dir)
    writer_packet = ContextCompiler.for_writer(full_packet) if full_packet else None
    contract = strategy.get("discourse_contract") or {}

    writer = WriterAgent(
        config=writer_config,
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
    )

    async def _write(feedback: str | None) -> str:
        drafted, _ = await writer.write_with_loop(
            chapter_number=chapter_number,
            outline=outline,
            context_packet=writer_packet,
            target_chapter_words=target_words,
            orchestrator_strategy=strategy,
            max_rounds=1,
            revision_feedback=feedback,
        )
        drafted = strip_writer_preamble(drafted)
        if not drafted.strip():
            raise EmptyDraftError(
                f"Writer returned empty draft for chapter {chapter_number}"
            )
        return drafted

    content = await _write(revision_feedback)
    print(
        f"  [AgentLoop] Writer: {len(content)} chars, "
        f"tokens: {writer.input_tokens}/{writer.output_tokens}"
    )

    def _gate(text: str) -> tuple[dict, bool]:
        report = QualityService.check_draft_hard_gates(
            text, target_words=target_words, chapter_outline=outline
        )
        return report, bool(report.get("passed"))

    quality_gate_report, gate_passed = _gate(content)
    retries = 0
    revision_brief = ""
    while not gate_passed and retries < max_retries:
        retries += 1
        revision_brief = direct_revision(gate_report=quality_gate_report)
        print(f"  [AgentLoop] Hard gate retry {retries}/{max_retries}")
        content = await _write(revision_brief)
        quality_gate_report, gate_passed = _gate(content)
        print(
            f"  [AgentLoop] QualityGate after retry {retries}: "
            f"{'PASS' if gate_passed else 'FAIL'}"
        )

    previous_draft = ""
    if mgr is not None and project_id and chapter_number > 1:
        earlier = mgr.get_chapter(project_id, chapter_number - 1) or {}
        if isinstance(earlier, dict):
            previous_draft = str(earlier.get("draft_content") or "")
    if previous_draft and closing_repeats(previous_draft, content):
        print("  [AgentLoop] Closing repeats the previous chapter → rewrite once")
        echo_brief = (
            "章末还在重复上一章末尾的动作。"
            "用本章新发生的事重写最后一段，前面的情节保持不变。"
        )
        rewritten = await _write(echo_brief)
        rewritten_report, rewritten_passed = _gate(rewritten)
        if rewritten_passed:
            content = rewritten
            quality_gate_report = rewritten_report
            gate_passed = True
            revision_brief = echo_brief
            print(f"  [AgentLoop] Closing rewrite kept: {len(content)} chars")
        else:
            print("  [AgentLoop] Closing rewrite missed the hard gate; kept the earlier draft")

    timeline_report = _draft_timeline(full_packet, content, chapter_number)
    contract_report = check_discourse_contract(content, contract)
    timeline_passed = bool(timeline_report.get("passed"))
    contract_passed = bool(contract_report.get("passed"))
    editor_skipped = False
    continuity_skipped = False
    editor_report: dict = {}
    continuity_report: dict = {}
    _e_latency = 0.0
    _c_latency = 0.0
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

    def _deterministic_clean() -> bool:
        return gate_passed and timeline_passed and contract_passed

    if _deterministic_clean():
        editor_skipped = True
        continuity_skipped = True
        editor_report = _skipped_editor_report("deterministic_checks_passed")
        continuity_report = _skipped_continuity_report("deterministic_checks_passed")
        print(
            "  [AgentLoop] Gate, timeline, and contract PASS "
            "→ skip Editor + Continuity"
        )
    else:
        editor = EditorAgent(config=_config_for(TaskClass.REVIEW))
        continuity = ContinuityAgent(
            config=_config_for(TaskClass.REVIEW),
            chapter_store=store,
            project_id=project_id,
        )

        async def _board(draft: str) -> tuple[dict, dict, dict, dict]:
            nonlocal _e_latency, _c_latency
            timeline = _draft_timeline(full_packet, draft, chapter_number)
            contract_report = check_discourse_contract(draft, contract)
            editor_source = dict(full_packet)
            if not timeline.get("passed"):
                editor_source["timeline_findings"] = [
                    *(editor_source.get("timeline_findings") or []),
                    *(timeline.get("findings") or []),
                ]
            editor_packet = (
                ContextCompiler.for_editor(editor_source) if editor_source else None
            )
            continuity_packet = (
                ContextCompiler.for_continuity(full_packet) if full_packet else None
            )
            style_report = _style_dump(draft)

            async def _edit() -> dict:
                nonlocal _e_latency
                started = time.monotonic()
                report, _ = await editor.review(
                    chapter_number=chapter_number,
                    draft_content=draft,
                    narrative_mode=narrative_mode,
                    style_report=style_report,
                    context_packet=editor_packet,
                )
                _e_latency += time.monotonic() - started
                return report

            async def _audit() -> dict:
                nonlocal _c_latency
                started = time.monotonic()
                report, _ = await continuity.audit(
                    chapter_number=chapter_number,
                    draft_content=draft,
                    narrative_mode=narrative_mode,
                    context_packet=continuity_packet,
                )
                _c_latency += time.monotonic() - started
                return report

            edited, audited = await asyncio.gather(_edit(), _audit())
            return timeline, contract_report, edited, audited

        (
            timeline_report,
            contract_report,
            editor_report,
            continuity_report,
        ) = await _board(content)
        timeline_passed = bool(timeline_report.get("passed"))
        contract_passed = bool(contract_report.get("passed"))
        print(
            f"  [AgentLoop] Board: gate={'PASS' if gate_passed else 'FAIL'}, "
            f"timeline={'PASS' if timeline_passed else 'FAIL'}, "
            f"contract={'PASS' if contract_passed else 'FAIL'}, "
            f"editor={editor_report.get('verdict', '?')}"
        )

        while retries < max_retries and needs_rewrite(
            gate_passed=gate_passed,
            timeline_report=timeline_report,
            contract_report=contract_report,
            editor_report=editor_report,
            continuity_report=continuity_report,
        ):
            retries += 1
            revision_brief = direct_revision(
                gate_report=quality_gate_report,
                timeline_report=timeline_report,
                contract_report=contract_report,
                editor_report=editor_report,
                continuity_report=continuity_report,
            )
            print(f"  [AgentLoop] Revision {retries}/{max_retries}")
            content = await _write(revision_brief)
            quality_gate_report, gate_passed = _gate(content)
            (
                timeline_report,
                contract_report,
                editor_report,
                continuity_report,
            ) = await _board(content)
            timeline_passed = bool(timeline_report.get("passed"))
            contract_passed = bool(contract_report.get("passed"))

        editor_input_tokens = editor.input_tokens
        editor_output_tokens = editor.output_tokens
        editor_cached_tokens = editor.cached_tokens
        editor_reasoning_tokens = editor.reasoning_tokens
        editor_model_calls = editor.model_calls
        continuity_input_tokens = continuity.input_tokens
        continuity_output_tokens = continuity.output_tokens
        continuity_cached_tokens = continuity.cached_tokens
        continuity_reasoning_tokens = continuity.reasoning_tokens
        continuity_model_calls = continuity.model_calls

    surface_applied = False
    style = _style_dump(content)
    if _clustered_surface(style):
        phrases = [
            str(issue.get("phrase"))
            for issue in (style.get("issues") or [])
            if isinstance(issue, dict) and issue.get("phrase")
        ]
        tool = HumanizeTool(config=_config_for(TaskClass.REVIEW))
        result = await tool.execute(text=content, focus_patterns="、".join(phrases[:8]))
        rewritten = ""
        if getattr(result, "success", False):
            rewritten = str((getattr(result, "data", None) or {}).get("humanized_text") or "")
        rewritten = strip_writer_preamble(rewritten).strip()
        if rewritten and len(rewritten) >= int(len(content) * 0.75):
            content = rewritten
            surface_applied = True
            print(f"  [AgentLoop] Surface humanize kept: {len(content)} chars")
        else:
            print("  [AgentLoop] Surface humanize discarded")

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
            existing_entities = _wb_entities(
                mgr.get_relevant_world_entities(project_id, content) or []
            )
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
        "timeline_report": timeline_report,
        "timeline_passed": timeline_passed,
        "contract_report": contract_report,
        "contract_passed": contract_passed,
        "revision_brief": revision_brief,
        "surface_applied": surface_applied,
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
