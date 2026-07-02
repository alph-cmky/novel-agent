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

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
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

# ── Routing thresholds ──────────────────────────────────

MAX_RETRIES = 3
CONTINUITY_PASS_SCORE = 80       # pass to worldbuilding
EDITOR_APPROVE_SCORE = 60        # minimum for human approval
CONTINUITY_APPROVE_SCORE = 60    # minimum for human approval


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

    strategy = await orchestrator.analyze(
        chapter_number=state.get("chapter_number", 1),
        chapter_outline=state.get("chapter_outline", ""),
        previous_chapters=previous_chapters,
        character_context=state.get("character_context", ""),
        world_context=state.get("world_context", ""),
        story_length=story_length,
        target_chapter_words=target_words,
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

    return {
        "orchestrator_strategy": strategy,
        "character_context": char_ctx,
        "world_context": world_ctx,
        "recent_summary": state.get("recent_summary", ""),
    }


async def writer_node(state: NovelState) -> dict:
    """Writer Agent generates (or rewrites) chapter content."""
    target_words = state.get("target_chapter_words", 3000)
    story_length = state.get("story_length", "long")

    length_cfg = get_length_config(story_length)
    max_tokens = max(length_cfg.max_tokens, int(target_words * 2))

    config = _config_for(TaskClass.CREATIVE)
    config.max_tokens = max_tokens
    store = ChapterStore(state.get("persist_dir", "./novel-data/chroma_data"))
    project_id = state.get("project_id", "")

    writer = WriterAgent(
        config=config,
        chapter_store=store,
        project_id=project_id,
        target_chapter_words=target_words,
    )
    content, trace = await writer.write(
        chapter_number=state.get("chapter_number", 1),
        outline=state.get("chapter_outline", ""),
        character_context=state.get("character_context", ""),
        world_context=state.get("world_context", ""),
        recent_summary=state.get("recent_summary", ""),
        target_chapter_words=target_words,
        rewrite_instructions=state.get("rewrite_instructions", ""),
    )

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
    )
    score = report.get("overall_score", 0)
    print(f"  [Editor] {score}/100 — {report.get('verdict', '?')}")
    return {"editor_report": report}


async def continuity_node(state: NovelState) -> dict:
    """Continuity Agent audits cross-chapter consistency."""
    config = _config_for(TaskClass.REVIEW)
    store = ChapterStore(state.get("persist_dir", "./novel-data/chroma_data"))
    project_id = state.get("project_id", "")

    auditor = ContinuityAgent(
        config=config, chapter_store=store, project_id=project_id
    )
    report, _ = await auditor.audit(
        chapter_number=state.get("chapter_number", 1),
        draft_content=state.get("draft_content", ""),
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

    This is the feedback loop: instead of blindly retrying, the Orchestrator
    reads Editor/Continuity/Human feedback and produces specific, actionable
    guidance for the Writer to follow during rewrite.
    """
    orchestrator = OrchestratorAgent(config=_config_for(TaskClass.STRUCTURAL))

    human_feedback = state.get("human_feedback") or None
    source = "human" if human_feedback else "auto"

    print(f"  [Orchestrator Review] Analyzing {source} feedback, generating rewrite guide...")

    instructions = await orchestrator.review_feedback(
        chapter_number=state.get("chapter_number", 1),
        chapter_outline=state.get("chapter_outline", ""),
        draft_content=state.get("draft_content", ""),
        editor_report=state.get("editor_report", {}),
        continuity_report=state.get("continuity_report", {}),
        human_feedback=human_feedback,
    )

    print(f"  [Orchestrator Review] Guide: {len(instructions)} chars")

    return {
        "rewrite_instructions": instructions,
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

    - Good score + no criticals → worldbuilding → human review
    - Issues + retries left → orchestrator_review → writer (auto feedback loop)
    - Issues + no retries → worldbuilding → human review (human gets final say)
    """
    score = state.get("continuity_report", {}).get("overall_score", 0)
    criticals = [
        i for i in state.get("continuity_report", {}).get("inconsistencies", [])
        if i.get("severity") == "critical"
    ]
    retry = state.get("retry_count", 0)

    if score >= CONTINUITY_PASS_SCORE and not criticals:
        return "worldbuilding"

    if retry < MAX_RETRIES:
        return "orchestrator_review"

    return "worldbuilding"


def route_after_human(state: NovelState) -> Literal["__end__", "orchestrator_review"]:
    """Human approved → done. Human rejected → feedback loop for rewrite."""
    if state.get("human_approved", False):
        return "__end__"
    return "orchestrator_review"


# ── Build Graph ────────────────────────────────────────

def build_chapter_graph() -> StateGraph:
    """Build the multi-Agent chapter pipeline with feedback loop and HITL.

    Flow:
    Orchestrator → Writer → Editor → Continuity → [
        pass → Worldbuilding → Human Review (interrupt) → [
            approved → Done,
            rejected → Orchestrator Review → Writer
        ],
        fail + retries → Orchestrator Review → Writer (auto feedback loop),
        fail + no retries → Worldbuilding → Human Review (human decides)
    ]
    """
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

    return workflow.compile(checkpointer=MemorySaver())
