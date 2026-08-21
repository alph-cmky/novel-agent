"""Chapter-writing StateGraph — v2 with recursive self-evolution.

Evolution enabled (default):
    Orchestrator → Evolution Subgraph [
        Writer → Editor → Continuity → EvolutionOrchestrator → [continue|select_best]
    ] → Worldbuilding → Human Review → [approved → END | rejected → evolution_writer]

Evolution disabled (legacy):
    Orchestrator → Writer → Editor → Continuity → [
        pass → Worldbuilding → Human Review,
        fail → Orchestrator Review → Writer (feedback loop)
    ]
"""

import asyncio
import json as _json
import re
import sqlite3
from pathlib import Path
from typing import Literal

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from novel_agent.agents.base import AgentConfig
from novel_agent.agents.continuity import ContinuityAgent
from novel_agent.agents.editor import EditorAgent
from novel_agent.agents.evolution_orchestrator import EvolutionOrchestratorAgent
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.agents.worldbuilding import WorldbuildingAgent
from novel_agent.agents.writer import WriterAgent
from novel_agent.config import DEFAULT_MAX_TOKENS
from novel_agent.graph.evolution import (
    EvolutionConfig,
    build_improvement_plan_rule,
    build_quality_guard_report,
    check_quality_guards,
    composite_score,
    compute_delta,
    continuity_overall,
    decide_termination,
    editor_overall,
    extract_scores,
    is_better_candidate,
)
from novel_agent.graph.state import NovelState
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.routing import ModelRouter, TaskClass

router = ModelRouter()


def _text_units(text: str) -> int:
    """Count CJK characters or whitespace-delimited words for length guards."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    if cjk >= len(text) * 0.2:
        return cjk
    return len(re.findall(r"\b[\w']+\b", text))

# ── Helpers ─────────────────────────────────────────────


def _build_arc_summary(previous_chapters: list[dict]) -> str:
    """Build a summary of recent chapter performance from stored reports."""
    if not previous_chapters:
        return ""
    recent = previous_chapters[-5:]
    entries = []
    for c in recent:
        cn = c.get("chapter_number", "?")
        try:
            er = _json.loads(c.get("editor_report", "{}"))
        except (_json.JSONDecodeError, TypeError):
            er = {}
        try:
            cr = _json.loads(c.get("continuity_report", "{}"))
        except (_json.JSONDecodeError, TypeError):
            cr = {}
        e_score = er.get("overall_score", "?")
        if cr.get("unavailable"):
            entries.append(f"第{cn}章: Editor {e_score}/100, Continuity N/A（审计不可用）")
        else:
            c_score = cr.get("overall_score", "?")
            entries.append(f"第{cn}章: Editor {e_score}/100, Continuity {c_score}/100")
    return "## 最近章节表现\n" + "\n".join(entries) if entries else ""


# ChromaDB ChapterStore 按 persist_dir 缓存，避免每个进化轮都 new PersistentClient
_chapter_store_cache: dict[str, ChapterStore] = {}


def _get_chapter_store(persist_dir: str) -> ChapterStore:
    store_dir = Path(persist_dir) / "chroma_data"
    key = str(store_dir.resolve())
    if key not in _chapter_store_cache:
        _chapter_store_cache[key] = ChapterStore(store_dir)
    return _chapter_store_cache[key]


# ── Node config ─────────────────────────────────────────


def _config_for(task: TaskClass) -> AgentConfig:
    """Create AgentConfig from the model router's decision."""
    route = router.resolve(task)
    kwargs: dict = {"model": route.model, "temperature": route.temperature}
    # worldbuilding 要输出大量实体+伏笔，4096 容易截断导致 JSON 不完整
    kwargs["max_tokens"] = 8192 if task is TaskClass.EXTRACTION else 4096
    if route.api_key:
        kwargs["api_key"] = route.api_key
    if route.base_url:
        kwargs["base_url"] = route.base_url
    kwargs["is_reasoning"] = route.is_reasoning
    return AgentConfig(**kwargs)


def _format_improvement_plan(plan: dict | None, version: int) -> str:
    """Format an improvement_plan dict into a prompt section for Writer."""
    if not plan:
        return ""

    parts = [f"## 进化改进指导 (第 {version + 1} 次迭代)"]
    parts.append(f"这是第 {version + 1} 次改进。前面几轮的改进已经提升了部分维度的质量。")

    focus = plan.get("focus_dimensions", [])
    if focus:
        dim_labels = {
            "rhythm": "节奏", "ai_flavor": "AI味", "dialogue": "对话",
            "logic": "逻辑", "writing": "文笔",
        }
        focus_cn = ", ".join(dim_labels.get(d, d) for d in focus)
        parts.append(f"\n### 本轮重点维度\n{focus_cn}")

    primary = plan.get("primary_instruction", "")
    if primary:
        parts.append(f"\n### 核心指令\n{primary}")

    secondary = plan.get("secondary_instructions", [])
    if secondary:
        parts.append("\n### 辅助指令")
        for s in secondary:
            parts.append(f"- {s}")

    constraints = plan.get("constraints", {})
    preserve = constraints.get("preserve", [])
    if preserve:
        dim_labels = {
            "rhythm": "节奏", "ai_flavor": "AI味", "dialogue": "对话",
            "logic": "逻辑", "writing": "文笔",
        }
        preserve_cn = ", ".join(dim_labels.get(d, d) for d in preserve)
        parts.append(f"\n### 请保持\n{preserve_cn} 方面的已有进步，不要牺牲它们")

    avoid = constraints.get("avoid", [])
    if avoid:
        parts.append("\n### 明确禁止")
        for a in avoid:
            parts.append(f"- {a}")

    # 篇幅与结构完整性保护（防止迭代时字数缩水或仅输出片段）
    parts.append(
        "\n### ⚠️ 篇幅与结构硬性要求（必须严格遵守）\n"
        "1. **必须输出完整的全章节正文**：绝对禁止只输出修改片段、大纲提要或省略号！\n"
        "2. **字数必须达标**：严格保持全章篇幅完整展开，情节要充分铺陈细化，严禁过度压缩概括。\n"
        "3. **保持故事连续性**：在优化局部细节的同时，必须保留全部核心剧情、人物对话与冲突高潮。"
    )

    return "\n".join(parts)


# ── Nodes ──────────────────────────────────────────────

async def orchestrator_node(state: NovelState) -> dict:
    """Orchestrator analyzes narrative position and assembles context."""
    if state.get("skip_orchestrator"):
        return {
            "orchestrator_strategy": {
                "narrative_stage": "development",
                "chapter_strategy": {"pacing": "normal"},
                "context_needed": {},
            }
        }
    orchestrator = OrchestratorAgent(config=_config_for(TaskClass.STRUCTURAL))
    story_length = state.get("story_length", "long")

    persist_dir = state.get("persist_dir", "./novel-data")
    project_id = state.get("project_id", "")
    previous_chapters: list[dict] = []
    if project_id:
        try:
            from novel_agent.storage.manager import ProjectManager
            mgr = ProjectManager(persist_dir)
            all_chapters = mgr.get_all_chapters(project_id)
            previous_chapters = [
                c for c in all_chapters
                if c["chapter_number"] < state.get("chapter_number", 1)
                and c.get("status") != "failed"
            ]
        except Exception as exc:
            print(f"  [Orchestrator] 加载前文失败，跳过: {exc}")

    target_words = state.get("target_chapter_words", 3000)
    narrative_mode = state.get("narrative_mode")
    narrative_perspective = state.get("narrative_perspective", "")

    arc_summary = _build_arc_summary(previous_chapters)

    context_for_compression = {
        "chapters": "\n".join(c.get("draft_content", "") for c in previous_chapters),
        "character_context": state.get("character_context", ""),
        "world_context": state.get("world_context", ""),
    }
    if orchestrator._compressor.should_compress(context_for_compression):
        compressed = await orchestrator._compressor.compress(previous_chapters)
        recent_summary = compressed.get("recent_summary", "")
        critical = compressed.get("critical_snippets", "")
        if critical:
            recent_summary = f"{recent_summary}\n\n## 关键长期事实\n{critical}"
    else:
        recent_summary = state.get("recent_summary", "")

    # Load unresolved foreshadowings before planning so the same long-range
    # constraints reach both the orchestrator and the writer.
    unresolved: list[str] = []
    if project_id:
        try:
            mgr2 = ProjectManager(persist_dir)
            all_fs = mgr2.get_foreshadowings(project_id)
            open_fs = [f for f in all_fs if f.get("status") in ("open", "planted")]
            unresolved = [
                f"[第{f.get('planted_chapter', '?')}章] {f.get('description', '')}"
                for f in open_fs
            ]
            if unresolved:
                print(f"  [Orchestrator] {len(unresolved)} unresolved foreshadowings")
        except Exception as exc:
            print(f"  [Orchestrator] 加载伏笔失败，跳过: {exc}")

    strategy = await orchestrator.analyze(
        chapter_number=state.get("chapter_number", 1),
        chapter_outline=state.get("chapter_outline", ""),
        previous_chapters=previous_chapters,
        character_context=state.get("character_context", ""),
        world_context=state.get("world_context", ""),
        story_length=story_length,
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
        arc_summary=arc_summary,
        unresolved_foreshadowings=unresolved,
    )

    stage = strategy.get("narrative_stage", "?")
    pacing = strategy.get("chapter_strategy", {}).get("pacing", "?")
    prev_count = len(previous_chapters)
    print(f"  [Orchestrator] Stage: {stage}, Pacing: {pacing}, Prev: {prev_count}ch")

    context_needed = strategy.get("context_needed", {})
    extra_chars = ", ".join(context_needed.get("characters", []))
    extra_world = ", ".join(context_needed.get("world_elements", []))

    char_ctx = state.get("character_context", "")
    if extra_chars:
        hint = f"[本章涉及角色: {extra_chars}]"
        char_ctx = f"{char_ctx}\n{hint}" if char_ctx else hint

    world_ctx = state.get("world_context", "")
    if extra_world:
        hint = f"[本章涉及设定: {extra_world}]"
        world_ctx = f"{world_ctx}\n{hint}" if world_ctx else hint

    persp_specific = context_needed.get("perspective_specific", "")
    if persp_specific:
        char_ctx = (
            f"{char_ctx}\n[视角特定信息: {persp_specific}]"
            if char_ctx else f"[视角特定信息: {persp_specific}]"
        )

    cross_timeline = context_needed.get("cross_timeline_references", [])
    if cross_timeline:
        timeline_hint = f"[跨时间线参考: {', '.join(cross_timeline)}]"
        world_ctx = (
            f"{world_ctx}\n{timeline_hint}"
            if world_ctx else timeline_hint
        )

    recent_ref = context_needed.get("recent_reference", "")
    if recent_ref:
        ref_hint = f"[主编提示：本章需要回顾 — {recent_ref}]"
        recent_summary = (
            f"{recent_summary}\n{ref_hint}" if recent_summary else ref_hint
        )

    return {
        "orchestrator_strategy": strategy,
        "character_context": char_ctx,
        "world_context": world_ctx,
        "recent_summary": recent_summary,
        "unresolved_foreshadowings": unresolved,
    }


async def writer_node(state: NovelState, config: RunnableConfig | None = None) -> dict:
    """Writer Agent generates (or rewrites) chapter content.

    Supports both evolution improvement_plan (v2) and legacy rewrite_instructions (v1).
    """
    target_words = state.get("target_chapter_words", 3000)
    max_tokens = max(DEFAULT_MAX_TOKENS, int(target_words * 3))

    agent_config = _config_for(TaskClass.CREATIVE)
    agent_config.max_tokens = max_tokens
    _persist = state.get("persist_dir", "./novel-data")
    store = _get_chapter_store(_persist)
    project_id = state.get("project_id", "")

    narrative_mode = state.get("narrative_mode")
    narrative_perspective = state.get("narrative_perspective", "")

    writer = WriterAgent(
        config=agent_config,
        chapter_store=store,
        project_id=project_id,
        target_chapter_words=target_words,
        narrative_mode=narrative_mode,
        narrative_perspective=narrative_perspective,
    )

    stream_queue: asyncio.Queue | None = None
    if config and config.get("configurable"):
        stream_queue = config["configurable"].get("stream_queue")

    # Determine rewrite instructions — evolution plan takes priority
    rewrite_text = ""
    constraints = {}
    evolution_version = state.get("evolution_version", 0)

    improvement_plan = state.get("evolution_improvement_plan") or {}
    if improvement_plan.get("primary_instruction"):
        # Evolution mode: format improvement_plan into instructions
        rewrite_text = _format_improvement_plan(improvement_plan, evolution_version)
        constraints = improvement_plan.get("constraints", {})
    else:
        # Legacy mode: handle rewrite_instructions (str | dict)
        raw_instructions = state.get("rewrite_instructions", "")
        if isinstance(raw_instructions, dict):
            rewrite_text = raw_instructions.get("instructions", "")
            constraints = raw_instructions.get("constraints", {})
        else:
            rewrite_text = raw_instructions

    # Merge constraints.strategy_override into orchestrator_strategy
    strategy = state.get("orchestrator_strategy", {})
    override = constraints.get("strategy_override")
    # LLM 偶发把 strategy_override 输出成 list/str，dict.update 会抛
    # ValueError（进化第二轮触发）；非 dict 直接忽略。
    if isinstance(override, dict) and override:
        cs = strategy.setdefault("chapter_strategy", {})
        cs.update(override)

    write_args = dict(
        chapter_number=state.get("chapter_number", 1),
        outline=state.get("chapter_outline", ""),
        character_context=state.get("character_context", ""),
        world_context=state.get("world_context", ""),
        recent_summary=state.get("recent_summary", ""),
        unresolved_foreshadowings=state.get("unresolved_foreshadowings", []),
        target_chapter_words=target_words,
        rewrite_instructions=rewrite_text,
        orchestrator_strategy=strategy,
    )

    if stream_queue is not None:
        collected: list[str] = []
        async for chunk in writer.write_stream(**write_args):
            collected.append(chunk)
            await stream_queue.put(("chunk", chunk))

        content = "".join(collected)

        if not content.strip():
            print("  [Writer] Stream returned empty, falling back to non-streaming...")
            content, trace = await writer.write(**write_args)
            if content.strip():
                await stream_queue.put(("chunk", content))
            tok_info = f"{trace.input_tokens}/{trace.output_tokens}" if trace else "?/?"
        else:
            trace = writer.latest_trace
            tok_info = f"{trace.input_tokens}/{trace.output_tokens}" if trace else "?/?"
    else:
        content, trace = await writer.write(**write_args)
        tok_info = f"{trace.input_tokens}/{trace.output_tokens}"

    # 篇幅硬底线保障：新版本至少保留目标的 85%，否则补偿生成一次。
    content_units = _text_units(content.strip())
    min_required_units = max(600, int(target_words * 0.85))
    if content_units < min_required_units and target_words >= 1000:
        print(
            f"  [Writer] 章节篇幅过短 ({content_units} units < {min_required_units})，"
            "触发保底补偿生成..."
        )
        compensated_args = dict(write_args)
        compensated_args["rewrite_instructions"] = (
            f"【紧急注意】：上一版输出篇幅严重不足（仅有 {content_units} units）。"
            "请务必充分展开场景细节、人物神态、环境氛围与多轮交锋对话，"
            f"写满至少 {target_words} 字，严禁大纲式概括！\n" + (write_args.get("rewrite_instructions") or "")
        )
        content_comp, trace_comp = await writer.write(**compensated_args)
        if _text_units(content_comp.strip()) > content_units:
            content = content_comp
            trace = trace_comp
            tok_info = f"{trace.input_tokens}/{trace.output_tokens}"

    label = f"(v{evolution_version})"
    wmsg = f"  [Writer] {len(content)} chars/{_text_units(content)} units {label} "
    wmsg += f"(target: {target_words}w, tokens: {tok_info})"
    print(wmsg)
    return {
        "draft_content": content,
        "evolution_round": state.get("evolution_round", 0),
    }


async def editor_node(state: NovelState) -> dict:
    """Editor Agent reviews the chapter."""
    editor = EditorAgent(config=_config_for(TaskClass.REVIEW))
    report, _ = await editor.review(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
        narrative_mode=state.get("narrative_mode"),
    )
    if report.get("unavailable"):
        print("  [Editor] unavailable（空输出，审查维度跳过）")
        return {"editor_report": report}
    score = report.get("overall_score", 0)
    print(f"  [Editor] {score}/100 — {report.get('verdict', '?')}")
    return {"editor_report": report}


async def continuity_node(state: NovelState) -> dict:
    """Continuity Agent audits cross-chapter consistency."""
    config = _config_for(TaskClass.REVIEW)
    _persist = state.get("persist_dir", "./novel-data")
    store = _get_chapter_store(_persist)
    project_id = state.get("project_id", "")

    auditor = ContinuityAgent(
        config=config, chapter_store=store, project_id=project_id
    )
    report, _ = await auditor.audit(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
        narrative_mode=state.get("narrative_mode"),
    )
    score = report.get("overall_score", 0)
    if report.get("unavailable"):
        print("  [Continuity] unavailable（空输出，一致性维度跳过）")
        return {"continuity_report": report}
    criticals = [
        i for i in (report.get("inconsistencies") or [])
        if isinstance(i, dict) and i.get("severity") == "critical"
    ]
    print(f"  [Continuity] {score}/100, Critical: {len(criticals)}")
    return {"continuity_report": report}


async def evolution_orchestrator_node(state: NovelState) -> dict:
    """EvolutionOrchestrator — version comparison, Delta, termination, improvement plan.

    Rule layer (deterministic) + optional LLM enrichment (natural language guidance).
    """
    current_round = state.get("evolution_round", 0)
    version = state.get("evolution_version", 0)
    current_scores = extract_scores(state)
    evo_config = EvolutionConfig(max_rounds=state.get("evolution_max_rounds", 5))

    # ── Branch A: First round (no history) ──
    if not state.get("evolution_history"):
        initial_guard = build_quality_guard_report(state, state)
        entry = {
            "v": version,
            "editor": current_scores["editor_overall"],
            "continuity": current_scores["continuity_overall"],
            "composite": composite_score(current_scores),
            "dimensions": current_scores["dimensions"],
            "delta": None,
            "focus": None,
            "quality_guard": initial_guard,
        }

        # Rule-based improvement plan for v0
        rule_plan = build_improvement_plan_rule(current_scores, delta=None, config=evo_config)

        # LLM enrichment (optional — fails gracefully to rule_plan)
        plan = rule_plan
        if state.get("evolution_max_rounds", 5) > 0 and not state.get(
            "skip_evolution_enrichment"
        ):
            try:
                agent = EvolutionOrchestratorAgent(config=_config_for(TaskClass.META_EVALUATION))
                enriched = await agent.enrich_plan(
                    current_version=version,
                    current_scores=current_scores,
                    delta=None,
                    rule_plan=rule_plan,
                    history=[],
                    draft_preview=state.get("draft_content", ""),
                )
                if enriched and enriched.get("primary_instruction"):
                    plan = enriched
            except Exception:
                pass

        editor_score = current_scores["editor_overall"]
        continuity_score = current_scores["continuity_overall"]
        print(f"  [EvoOrchestrator] v{version} E:{editor_score} C:{continuity_score} "
              f"→ 首轮，记录历史，生成改进计划 focus={plan.get('focus_dimensions', [])}")

        return {
            "evolution_round": 1,
            "evolution_version": 1,
            "evolution_history": [entry],
            "evolution_best_version": version,
            "evolution_best_draft": state.get("draft_content", ""),
            "evolution_best_editor_report": state.get("editor_report", {}),
            "evolution_best_continuity_report": state.get("continuity_report", {}),
            "evolution_best_quality_guard": initial_guard,
            "quality_guard_report": initial_guard,
            "evolution_improvement_plan": plan,
            "evolution_termination": "",
        }

    # ── Branch B: Has history → Delta comparison ──
    previous = state["evolution_history"][-1]
    previous_scores = {
        "editor_overall": previous["editor"],
        "continuity_overall": previous["continuity"],
        "dimensions": previous.get("dimensions", {}),
    }
    delta = compute_delta(current_scores, previous_scores)

    best_ed_rpt = state.get("evolution_best_editor_report", {}) or {}
    best_ct_rpt = state.get("evolution_best_continuity_report", {}) or {}
    # 与 current_scores 同口径走 extract_scores：维度缺失补 0、editor/continuity
    # 假 0 一律中和，否则 dims_avg 分母不一致 / best 假 0 拖低对比基准。
    best_scores = extract_scores({
        "editor_report": best_ed_rpt,
        "continuity_report": best_ct_rpt,
    })
    best_state = {
        "draft_content": state.get("evolution_best_draft", ""),
        "editor_report": best_ed_rpt,
        "continuity_report": best_ct_rpt,
        "worldbuilding_report": state.get("evolution_best_worldbuilding_report", {}),
        "outline_coverage": state.get("evolution_best_outline_coverage"),
        "required_facts_missing": state.get("evolution_best_required_facts_missing", 0),
    }
    guard_report = check_quality_guards(state, best_state, evo_config)

    # 1. Rule layer: termination decision
    termination, reason = decide_termination(
        delta,
        current_scores,
        best_scores,
        state["evolution_history"],
        evo_config,
        current_round,
        guard_report,
    )

    # 2. Rule layer: improvement plan
    rule_plan = build_improvement_plan_rule(current_scores, delta, evo_config)

    # 3. LLM enrichment (only if continuing)
    plan = rule_plan
    if not termination and not state.get("skip_evolution_enrichment"):
        try:
            agent = EvolutionOrchestratorAgent(config=_config_for(TaskClass.META_EVALUATION))
            enriched = await agent.enrich_plan(
                current_version=version,
                current_scores=current_scores,
                delta=delta,
                rule_plan=rule_plan,
                history=state["evolution_history"],
                draft_preview=state.get("draft_content", ""),
            )
            if enriched and enriched.get("primary_instruction"):
                plan = enriched
        except Exception:
            pass

    # 4. Build history entry
    new_entry = {
        "v": version,
        "editor": current_scores["editor_overall"],
        "continuity": current_scores["continuity_overall"],
        "composite": composite_score(current_scores),
        "dimensions": current_scores["dimensions"],
        "delta": delta,
        "focus": plan.get("focus_dimensions", []) if plan else [],
        "quality_guard": guard_report,
    }

    result: dict = {
        "evolution_history": state["evolution_history"] + [new_entry],
        "evolution_termination": termination,
        "quality_guard_report": guard_report,
    }

    # 5. Check if new best version
    is_new_best, selection_report = is_better_candidate(state, best_state, evo_config)
    if is_new_best:
        result.update({
            "evolution_best_version": version,
            "evolution_best_draft": state.get("draft_content", ""),
            "evolution_best_editor_report": state.get("editor_report", {}),
            "evolution_best_continuity_report": state.get("continuity_report", {}),
            "evolution_best_quality_guard": selection_report,
            "evolution_best_worldbuilding_report": state.get("worldbuilding_report", {}),
            "evolution_best_outline_coverage": state.get("outline_coverage"),
            "evolution_best_required_facts_missing": state.get("required_facts_missing", 0),
        })

    # 6. If continuing, increment counters + issue plan
    if not termination:
        result.update({
            "evolution_round": current_round + 1,
            "evolution_version": version + 1,
            "evolution_improvement_plan": plan,
        })

    editor_score = current_scores["editor_overall"]
    continuity_score = current_scores["continuity_overall"]
    best_v = result.get("evolution_best_version", state.get("evolution_best_version", 0))
    status = f"终止:{termination}" if termination else "继续"
    print(f"  [EvoOrchestrator] v{version} E:{editor_score} C:{continuity_score} "
          f"Δ={delta['trend']} best=v{best_v} {status}")

    return result


def select_best_version_node(state: NovelState) -> dict:
    """Select the best version and prepare for DB write.

    The actual DB write happens in the SSE layer (create_sse_stream / resume_graph),
    not here — this node just sets the final draft_content to the best version.
    """
    best_version = state.get("evolution_best_version", 0)
    current_version = state.get("evolution_version", 0)
    termination = state.get("evolution_termination", "")

    # If best is current (no-op) or best is a previous version (rollback)
    if best_version != current_version:
        print(
            f"  [SelectBest] Rolling back: "
            f"v{current_version} → v{best_version} (best)"
        )
        best_draft = state.get("evolution_best_draft", "")
        best_ed = state.get("evolution_best_editor_report", {})
        best_ct = state.get("evolution_best_continuity_report", {})
        return {
            "draft_content": best_draft or state.get("draft_content", ""),
            "editor_report": best_ed or state.get("editor_report", {}),
            "continuity_report": best_ct or state.get("continuity_report", {}),
        }

    print(f"  [SelectBest] v{best_version} is best, termination={termination}")
    return {}


async def worldbuilding_node(state: NovelState) -> dict:
    """Worldbuilding Agent extracts entities, conflicts, and foreshadowings."""
    if state.get("skip_worldbuilding"):
        return {"worldbuilding_report": {}}
    existing = state.get("existing_world_entities", [])

    persist_dir = state.get("persist_dir", "./novel-data")
    project_id = state.get("project_id", "")
    existing_fs: list[dict] = []
    if project_id:
        try:
            from novel_agent.storage.manager import ProjectManager
            mgr = ProjectManager(persist_dir)
            existing_fs = mgr.get_foreshadowings(project_id)
        except Exception as exc:
            print(f"  [Worldbuilding] 加载伏笔失败，跳过: {exc}")

    wb = WorldbuildingAgent(
        config=_config_for(TaskClass.EXTRACTION),
        existing_entities=existing,
        existing_foreshadowings=existing_fs,
    )
    report, _ = await wb.extract(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
        narrative_mode=state.get("narrative_mode"),
    )
    entities = len(report.get("new_entities", []))
    conflicts = len(report.get("conflicts", []))
    new_fs = len(report.get("foreshadowings", []))
    resolved_fs = len(report.get("resolved_foreshadowings", []))
    print(
        f"  [Worldbuilding] {entities} entities, {conflicts} conflicts, "
        f"{new_fs} new foreshadowings, {resolved_fs} resolved"
    )
    return {"worldbuilding_report": report}


def human_review_node(state: NovelState) -> dict:
    """Human-in-the-loop review node.

    Uses LangGraph interrupt() to pause the graph and wait for human input.
    In evolution mode, rejection triggers a fresh evolution cycle (max 2 rounds).
    """
    editor_report = state.get("editor_report", {}) or {}
    continuity_report = state.get("continuity_report", {}) or {}
    wb_report = state.get("worldbuilding_report", {})

    editor_score = editor_overall(editor_report, continuity_report)
    continuity_score = continuity_overall(editor_report, continuity_report)
    editor_unavailable = bool(editor_report.get("unavailable"))
    continuity_unavailable = bool(continuity_report.get("unavailable"))

    evolution_rounds = len(state.get("evolution_history", []))
    evolution_termination = state.get("evolution_termination", "")

    feedback = interrupt({
        "type": "human_review",
        "chapter_number": state.get("chapter_number", 1),
        "draft_preview": state.get("draft_content", "")[:1000],
        "draft_full": state.get("draft_content", ""),
        "editor_score": editor_score,
        "continuity_score": continuity_score,
        "editor_unavailable": editor_unavailable,
        "continuity_unavailable": continuity_unavailable,
        "editor_issues": editor_report.get("issues", [])[:10],
        "continuity_issues": continuity_report.get("inconsistencies", [])[:10],
        "wb_new_entities": len(wb_report.get("new_entities", [])),
        "wb_conflicts": len(wb_report.get("conflicts", [])),
        "retry_count": state.get("retry_count", 0),
        "evolution_rounds": evolution_rounds,
        "evolution_termination": evolution_termination,
    })

    approved = feedback.get("action") == "approve"
    comments = feedback.get("comments", "")

    print(f"\n  {'─' * 40}")
    print(f"  [Human Review] {'Approved' if approved else 'Rejected'}")
    if comments:
        print(f"  Comments: {comments[:120]}")
    print(f"  {'─' * 40}\n")

    if approved:
        return {"human_approved": True, "human_feedback": feedback}

    # Rejected — human feedback triggers a fresh evolution cycle
    rejects = state.get("evolution_human_rejects", 0) + 1

    if rejects >= 3:
        # Do not convert repeated rejection into consent. End as a draft so
        # the user can explicitly retry, edit, or abandon the chapter.
        return {
            "human_approved": False,
            "human_feedback": feedback,
            "evolution_human_rejects": rejects,
            "human_review_exhausted": True,
        }

    # Build improvement plan from human feedback
    plan = {
        "focus_dimensions": [],
        "primary_instruction": f"人类审阅者拒绝了这个版本。意见：{comments}",
        "secondary_instructions": [],
        "constraints": {
            "preserve": ["章节大纲", "核心情节走向"],
            "avoid": [],
            "strategy_override": {},
        },
    }

    return {
        "human_approved": False,
        "human_feedback": feedback,
        "evolution_human_rejects": rejects,
        "evolution_improvement_plan": plan,
        "evolution_round": 0,
        "evolution_version": 0,
        "evolution_history": [],
        "evolution_max_rounds": 2,
        "evolution_termination": "",
    }


# ── Routers ────────────────────────────────────────────

def route_after_evolution(
    state: NovelState,
) -> Literal["evolution_writer", "evolution_select_best"]:
    """Evolution router: continue iterating or select best and exit."""
    if state.get("evolution_termination"):
        return "evolution_select_best"

    max_rounds = state.get("evolution_max_rounds", 5)
    # The first evolution_orchestrator pass records v0 and does not rewrite.
    # max_rounds therefore counts actual Writer rewrites, not bookkeeping passes.
    if max(state.get("evolution_round", 0) - 1, 0) >= max_rounds:
        return "evolution_select_best"

    return "evolution_writer"


def route_after_writer(state: NovelState) -> Literal["evolution_editor", "worldbuilding"]:
    """Fast evaluation profile skips reviews when no rewrite can consume them."""
    interval = max(state.get("review_interval", 1), 1)
    chapter_number = state.get("chapter_number", 1)
    if state.get("skip_reviews") or chapter_number % interval != 0:
        return "worldbuilding"
    return "evolution_editor"


def route_after_human_evolution(state: NovelState) -> Literal["__end__", "evolution_writer"]:
    """Human approved → done. Rejected → new evolution cycle."""
    if state.get("human_approved", False):
        return "__end__"

    # Rejection limit ends the run as an unapproved draft, never as approval.
    if state.get("evolution_human_rejects", 0) >= 3:
        return "__end__"

    return "evolution_writer"


# ── Build Graph ────────────────────────────────────────

_checkpointer_cache: dict[str, SqliteSaver] = {}


def _get_checkpointer(persist_dir: str) -> SqliteSaver | MemorySaver:
    """Return a SqliteSaver for the project directory, or MemorySaver as fallback."""
    if not persist_dir:
        return MemorySaver()
    db_path = Path(persist_dir) / "checkpoints.db"
    db_key = str(db_path.resolve())
    if db_key not in _checkpointer_cache:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _checkpointer_cache[db_key] = SqliteSaver(conn)
    return _checkpointer_cache[db_key]


def _build_workflow() -> StateGraph:
    """Build the recursive self-evolution StateGraph for chapter generation."""
    workflow = StateGraph(NovelState)

    # 1. 宏观规划与上下文装配
    workflow.add_node("orchestrator", orchestrator_node)

    # 2. 递归自演化子图节点
    workflow.add_node("evolution_writer", writer_node)
    workflow.add_node("evolution_editor", editor_node)
    workflow.add_node("evolution_continuity", continuity_node)
    # Extract facts before selecting a version so hard constraints can use
    # worldbuilding conflicts; run it again after selection for final state.
    workflow.add_node("evolution_worldbuilding", worldbuilding_node)
    workflow.add_node("evolution_orchestrator", evolution_orchestrator_node)
    workflow.add_node("evolution_select_best", select_best_version_node)

    # 3. 设定固化与人工审核
    workflow.add_node("worldbuilding", worldbuilding_node)
    workflow.add_node("human_review", human_review_node)

    # 状态拓扑编排
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "evolution_writer")
    workflow.add_conditional_edges(
        "evolution_writer",
        route_after_writer,
        {"evolution_editor": "evolution_editor", "worldbuilding": "worldbuilding"},
    )
    workflow.add_edge("evolution_editor", "evolution_continuity")
    workflow.add_edge("evolution_continuity", "evolution_worldbuilding")
    workflow.add_edge("evolution_worldbuilding", "evolution_orchestrator")
    workflow.add_conditional_edges(
        "evolution_orchestrator", route_after_evolution,
        {
            "evolution_writer": "evolution_writer",
            "evolution_select_best": "evolution_select_best",
        },
    )
    workflow.add_edge("evolution_select_best", "worldbuilding")
    workflow.add_edge("worldbuilding", "human_review")
    workflow.add_conditional_edges(
        "human_review", route_after_human_evolution,
        {"__end__": END, "evolution_writer": "evolution_writer"},
    )

    return workflow


def build_chapter_graph(persist_dir: str = "") -> StateGraph:
    """Build the chapter pipeline with sync checkpointer (for CLI / Chainlit)."""
    workflow = _build_workflow()
    checkpointer = _get_checkpointer(persist_dir)
    return workflow.compile(checkpointer=checkpointer)


# ── Async Build (for SSE / astream_events) ────────────────

_async_checkpointer_cache: dict[str, AsyncSqliteSaver] = {}


async def _get_checkpointer_async(persist_dir: str) -> AsyncSqliteSaver | MemorySaver:
    """AsyncSqliteSaver for use with graph.astream_events()."""
    if not persist_dir:
        return MemorySaver()
    db_path = Path(persist_dir) / "checkpoints.db"
    db_key = str(db_path.resolve())
    if db_key not in _async_checkpointer_cache:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        _async_checkpointer_cache[db_key] = AsyncSqliteSaver(conn)
    return _async_checkpointer_cache[db_key]


async def aclose_checkpointers() -> None:
    """Close all cached AsyncSqliteSaver connections and clear the cache.

    aiosqlite spawns a non-daemon worker thread per connection (started on
    ``await aiosqlite.connect(...)``); without an explicit ``await conn.close()``
    that thread keeps the interpreter alive at process exit, so a short-lived
    process that used :func:`build_chapter_graph_async` would hang on exit. Call
    this on server shutdown (FastAPI lifespan) or at the end of a short-lived
    run (CLI / eval harness) before the event loop closes.
    """
    savers = list(_async_checkpointer_cache.values())
    _async_checkpointer_cache.clear()
    for saver in savers:
        conn = getattr(saver, "conn", None)
        if conn is None:
            continue
        try:
            await conn.close()
        except Exception:
            pass


async def build_chapter_graph_async(
    persist_dir: str = "",
) -> StateGraph:
    """Async version for SSE endpoints (uses AsyncSqliteSaver)."""
    workflow = _build_workflow()
    checkpointer = await _get_checkpointer_async(persist_dir)
    return workflow.compile(checkpointer=checkpointer)
