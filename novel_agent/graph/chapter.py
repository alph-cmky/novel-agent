"""Chapter-writing StateGraph with feedback loop and Human-in-the-loop.

Orchestrator → Writer → Editor → Continuity → [
    pass → Worldbuilding → Human Review (interrupt) → [
        approved → Done,
        rejected → Orchestrator Review → Writer (with rewrite guidance)
    ],
    fail + retries → Orchestrator Review → Writer (auto feedback loop),
    fail + no retries → Worldbuilding → Human Review (human gets final say)
]
"""

import asyncio
import json as _json
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
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.agents.worldbuilding import WorldbuildingAgent
from novel_agent.agents.writer import WriterAgent
from novel_agent.config import get_length_config
from novel_agent.graph.state import NovelState
from novel_agent.memory.embeddings import ChapterStore
from novel_agent.routing import ModelRouter, TaskClass

router = ModelRouter()

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
        c_score = cr.get("overall_score", "?")
        entries.append(f"第{cn}章: Editor {e_score}/100, Continuity {c_score}/100")
    return "## 最近章节表现\n" + "\n".join(entries) if entries else ""


# ── Routing thresholds ──────────────────────────────────

MAX_RETRIES = 3
CONTINUITY_PASS_SCORE = 80       # pass to worldbuilding
EDITOR_APPROVE_SCORE = 60        # minimum editor score for auto-approval


def _config_for(task: TaskClass) -> AgentConfig:
    """Create AgentConfig from the model router's decision."""
    route = router.resolve(task)
    return AgentConfig(model=route.model, temperature=route.temperature)

# ── Nodes ──────────────────────────────────────────────

async def orchestrator_node(state: NovelState) -> dict:
    """Orchestrator analyzes narrative position and assembles context."""
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
            ]
        except Exception:
            pass

    target_words = state.get("target_chapter_words", 3000)
    narrative_mode = state.get("narrative_mode")
    narrative_perspective = state.get("narrative_perspective", "")

    # Build arc summary from previous chapters' stored reports
    arc_summary = _build_arc_summary(previous_chapters)

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

    # Inject new context_needed fields (Phase 0)
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

    # Inject recent_reference into recent_summary
    recent_summary = state.get("recent_summary", "")
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
    }


async def writer_node(state: NovelState, config: RunnableConfig | None = None) -> dict:
    """Writer Agent generates (or rewrites) chapter content.

    When config contains a stream_queue (asyncio.Queue), uses write_stream()
    and pushes chunks to the queue for SSE delivery.
    """
    target_words = state.get("target_chapter_words", 3000)
    story_length = state.get("story_length", "long")

    length_cfg = get_length_config(story_length)
    # Reasoning models need extra token budget for thinking phase
    max_tokens = max(length_cfg.max_tokens, int(target_words * 3))

    agent_config = _config_for(TaskClass.CREATIVE)
    agent_config.max_tokens = max_tokens
    _persist = state.get("persist_dir", "./novel-data")
    store = ChapterStore(_persist + "/chroma_data")
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

    # Handle structured rewrite_instructions (str | dict)
    raw_instructions = state.get("rewrite_instructions", "")
    constraints = {}
    if isinstance(raw_instructions, dict):
        rewrite_text = raw_instructions.get("instructions", "")
        constraints = raw_instructions.get("constraints", {})
    else:
        rewrite_text = raw_instructions

    # Merge constraints.strategy_override into orchestrator_strategy
    strategy = state.get("orchestrator_strategy", {})
    if constraints.get("strategy_override"):
        cs = strategy.setdefault("chapter_strategy", {})
        cs.update(constraints["strategy_override"])

    write_args = dict(
        chapter_number=state.get("chapter_number", 1),
        outline=state.get("chapter_outline", ""),
        character_context=state.get("character_context", ""),
        world_context=state.get("world_context", ""),
        recent_summary=state.get("recent_summary", ""),
        target_chapter_words=target_words,
        rewrite_instructions=rewrite_text,
        orchestrator_strategy=state.get("orchestrator_strategy", {}),
    )

    if stream_queue is not None:
        collected: list[str] = []
        async for chunk in writer.write_stream(**write_args):
            collected.append(chunk)
            await stream_queue.put(("chunk", chunk))

        content = "".join(collected)

        # Fallback: reasoning models may use all tokens for thinking,
        # leaving content empty. Retry with non-streaming write() which
        # uses higher effective token budget via tool-calling flow.
        if not content.strip():
            print("  [Writer] Stream returned empty, falling back to non-streaming...")
            content, trace = await writer.write(**write_args)
            if content.strip():
                # Push the full content as one chunk so the frontend gets it
                await stream_queue.put(("chunk", content))
            tok_info = f"{trace.input_tokens}/{trace.output_tokens}" if trace else "?/?"
        else:
            trace = writer._latest_trace
            tok_info = f"{trace.input_tokens}/{trace.output_tokens}" if trace else "?/?"
    else:
        content, trace = await writer.write(**write_args)
        tok_info = f"{trace.input_tokens}/{trace.output_tokens}"

    retry = state.get("retry_count", 0) + 1
    label = f"(retry {retry})" if retry > 1 else ""
    print(f"  [Writer] {len(content)} chars {label} (target: {target_words}w, tokens: {tok_info})")
    return {
        "draft_content": content,
        "retry_count": retry,
        "rewrite_instructions": "",  # consumed
    }


async def editor_node(state: NovelState) -> dict:
    """Editor Agent reviews the chapter."""
    editor = EditorAgent(config=_config_for(TaskClass.REVIEW))
    report, _ = await editor.review(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
        narrative_mode=state.get("narrative_mode"),
    )
    score = report.get("overall_score", 0)
    print(f"  [Editor] {score}/100 — {report.get('verdict', '?')}")
    return {"editor_report": report}


async def continuity_node(state: NovelState) -> dict:
    """Continuity Agent audits cross-chapter consistency."""
    config = _config_for(TaskClass.REVIEW)
    _persist = state.get("persist_dir", "./novel-data")
    store = ChapterStore(_persist + "/chroma_data")
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
    criticals = [
        i for i in report.get("inconsistencies", [])
        if i.get("severity") == "critical"
    ]
    print(f"  [Continuity] {score}/100, Critical: {len(criticals)}")
    return {"continuity_report": report}


async def orchestrator_review_node(state: NovelState) -> dict:
    """Orchestrator analyzes failure reports and generates rewrite instructions.

    Returns structured dict: {"instructions": str, "constraints": dict}.
    The writer_node handles both str and dict rewrite_instructions.
    """
    orchestrator = OrchestratorAgent(config=_config_for(TaskClass.STRUCTURAL))

    human_feedback = state.get("human_feedback") or None
    source = "human" if human_feedback else "auto"

    print(f"  [Orchestrator Review] Analyzing {source} feedback, generating rewrite guide...")

    result = await orchestrator.review_feedback(
        chapter_number=state.get("chapter_number", 1),
        chapter_outline=state.get("chapter_outline", ""),
        draft_content=state.get("draft_content", ""),
        editor_report=state.get("editor_report", {}),
        continuity_report=state.get("continuity_report", {}),
        human_feedback=human_feedback,
    )

    instr_len = len(result.get("instructions", ""))
    constraints = result.get("constraints", {})
    print(f"  [Orchestrator Review] Guide: {instr_len} chars, "
          f"constraints: {list(constraints.keys())}")

    return {
        "rewrite_instructions": result,
    }


async def worldbuilding_node(state: NovelState) -> dict:
    """Worldbuilding Agent extracts entities from approved chapter."""
    existing = state.get("existing_world_entities", [])
    wb = WorldbuildingAgent(
        config=_config_for(TaskClass.EXTRACTION),
        existing_entities=existing,
    )
    report, _ = await wb.extract(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
        narrative_mode=state.get("narrative_mode"),
    )
    entities = len(report.get("new_entities", []))
    conflicts = len(report.get("conflicts", []))
    print(f"  [Worldbuilding] {entities} new entities, {conflicts} conflicts")
    return {"worldbuilding_report": report}


def human_review_node(state: NovelState) -> dict:
    """Human-in-the-loop review node.

    Uses LangGraph interrupt() to pause the graph and wait for human input.
    The caller (CLI or Chainlit) catches GraphInterrupt, presents the draft
    to the user, and resumes with the human's decision.

    Interrupt payload includes draft preview, scores, and issue summaries
    so the human can make an informed decision.
    """
    editor_report = state.get("editor_report", {})
    continuity_report = state.get("continuity_report", {})
    wb_report = state.get("worldbuilding_report", {})

    editor_score = editor_report.get("overall_score", 0)
    continuity_score = continuity_report.get("overall_score", 0)

    # Build the interrupt payload — presented to the human reviewer
    feedback = interrupt({
        "type": "human_review",
        "chapter_number": state.get("chapter_number", 1),
        "draft_preview": state.get("draft_content", "")[:1000],
        "draft_full": state.get("draft_content", ""),
        "editor_score": editor_score,
        "continuity_score": continuity_score,
        "editor_issues": editor_report.get("issues", [])[:10],
        "continuity_issues": continuity_report.get("inconsistencies", [])[:10],
        "wb_new_entities": len(wb_report.get("new_entities", [])),
        "wb_conflicts": len(wb_report.get("conflicts", [])),
        "retry_count": state.get("retry_count", 0),
    })

    approved = feedback.get("action") == "approve"
    comments = feedback.get("comments", "")

    print(f"\n  {'─' * 40}")
    print(f"  [Human Review] {'Approved' if approved else 'Rejected'}")
    if comments:
        print(f"  Comments: {comments[:120]}")
    print(f"  {'─' * 40}\n")

    return {
        "human_approved": approved,
        "human_feedback": feedback,
    }


# ── Routers ────────────────────────────────────────────

def route_after_continuity(
    state: NovelState,
) -> Literal["worldbuilding", "orchestrator_review"]:
    """Decide: pass to worldbuilding, or trigger feedback loop.

    - Both scores good + no criticals → worldbuilding → human review
    - Issues + retries left → orchestrator_review → writer (auto feedback loop)
    - Issues + no retries → worldbuilding → human review (human gets final say)
    """
    c_score = state.get("continuity_report", {}).get("overall_score", 0)
    e_score = state.get("editor_report", {}).get("overall_score", 0)
    criticals = [
        i for i in state.get("continuity_report", {}).get("inconsistencies", [])
        if i.get("severity") == "critical"
    ]
    retry = state.get("retry_count", 0)

    if c_score >= CONTINUITY_PASS_SCORE and not criticals and e_score >= EDITOR_APPROVE_SCORE:
        return "worldbuilding"

    if retry < MAX_RETRIES:
        return "orchestrator_review"

    return "worldbuilding"


def route_after_human(state: NovelState) -> Literal["__end__", "orchestrator_review"]:
    """Human approved → done. Rejected + retries left → feedback loop.
    Rejected + no retries left → force end (prevents infinite reject loop)."""
    if state.get("human_approved", False):
        return "__end__"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "__end__"
    return "orchestrator_review"


# ── Build Graph ────────────────────────────────────────

_checkpointer_cache: dict[str, SqliteSaver] = {}


def _get_checkpointer(persist_dir: str) -> SqliteSaver | MemorySaver:
    """Return a SqliteSaver for the project directory, or MemorySaver as fallback.

    Caches checkpointer instances per persist_dir so we reuse the same
    SQLite connection across requests for the same project.
    """
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
    """Build the StateGraph with all nodes and edges (shared by sync + async builders)."""
    workflow = StateGraph(NovelState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("continuity", continuity_node)
    workflow.add_node("orchestrator_review", orchestrator_review_node)
    workflow.add_node("worldbuilding", worldbuilding_node)
    workflow.add_node("human_review", human_review_node)

    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_edge("editor", "continuity")

    workflow.add_conditional_edges(
        "continuity",
        route_after_continuity,
        {
            "worldbuilding": "worldbuilding",
            "orchestrator_review": "orchestrator_review",
        },
    )

    workflow.add_edge("orchestrator_review", "writer")
    workflow.add_edge("worldbuilding", "human_review")

    workflow.add_conditional_edges(
        "human_review",
        route_after_human,
        {"__end__": END, "orchestrator_review": "orchestrator_review"},
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


async def build_chapter_graph_async(persist_dir: str = "") -> StateGraph:
    """Async version for SSE endpoints (uses AsyncSqliteSaver)."""
    workflow = _build_workflow()
    checkpointer = await _get_checkpointer_async(persist_dir)
    return workflow.compile(checkpointer=checkpointer)
