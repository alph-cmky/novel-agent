"""SSE streaming and session management for the writing pipeline."""

import asyncio
import json
import uuid
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from novel_agent.observability.langfuse import create_trace, score_trace
from novel_agent.storage.manager import ProjectManager


class SessionStore:
    """In-memory store for active writing sessions.

    Each session holds a reference to the compiled graph, its config,
    and an asyncio.Queue for streaming chunks from writer_node to SSE.
    """

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, graph, queue: asyncio.Queue) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = {
            "graph": graph,
            "config": None,
            "queue": queue,
            "project_id": None,
            "chapter_number": None,
        }
        return session_id

    def get(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def get_queue(self, session_id: str) -> asyncio.Queue | None:
        s = self._sessions.get(session_id)
        return s["queue"] if s else None

    def set_config(self, session_id: str, config: dict) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["config"] = config

    def set_context(self, session_id: str, project_id: str, chapter_number: int) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["project_id"] = project_id
            self._sessions[session_id]["chapter_number"] = chapter_number

    def find_session(self, project_id: str, chapter_number: int) -> str | None:
        for sid, s in self._sessions.items():
            if s.get("project_id") == project_id and s.get("chapter_number") == chapter_number:
                return sid
        return None

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def _drain_queue(queue: asyncio.Queue):
    """Drain chunk events from the queue, yielding SSE strings."""
    try:
        while True:
            event_type, payload = await asyncio.wait_for(queue.get(), timeout=0.05)
            yield _sse_event(event_type, payload)
    except TimeoutError:
        return


_NODE_LABELS: dict[str, str] = {
    "orchestrator": "策略规划",
    "writer": "内容创作",
    "editor": "编辑审查",
    "continuity": "一致性审计",
    "orchestrator_review": "反馈分析",
    "worldbuilding": "世界观提取",
    "human_review": "人工审批",
}


async def _background_drain(
    queue: asyncio.Queue,
    output: asyncio.Queue,
    running: asyncio.Event,
):
    """Continuously drain writer chunks from queue into output queue."""
    while running.is_set():
        try:
            event_type, payload = await asyncio.wait_for(queue.get(), timeout=0.05)
            await output.put(_sse_event(event_type, payload))
        except TimeoutError:
            continue


async def _flush_output(output: asyncio.Queue):
    """Yield all pending items from the output queue."""
    while not output.empty():
        yield output.get_nowait()


async def _make_progress_event(
    name: str, status: str, event: dict | None = None
) -> str:
    """Build a progress SSE event from a node name and status."""
    label = _NODE_LABELS.get(name, name)
    score = None
    detail = None

    if status == "done" and event:
        output = event.get("data", {}).get("output", {})
        if name == "editor":
            report = output.get("editor_report", {})
            score = report.get("overall_score")
            detail = report.get("verdict")
        elif name == "continuity":
            report = output.get("continuity_report", {})
            score = report.get("overall_score")
        elif name == "worldbuilding":
            report = output.get("worldbuilding_report", {})
            n_ent = len(report.get('new_entities', []))
            n_conf = len(report.get('conflicts', []))
            detail = f"实体:{n_ent} 冲突:{n_conf}"

    return _sse_event("progress", {
        "node": name, "label": label, "status": status,
        "score": score, "detail": detail,
    })


async def create_sse_stream(
    store: SessionStore,
    session_id: str,
    graph,
    initial_state: dict,
    config: dict,
    mgr: ProjectManager,
    project_id: str,
    chapter_number: int,
):
    """Run the writing graph and stream events via SSE.

    Uses a background drain task to stream writer chunks in real-time,
    and graph.astream_events() for node-level progress events.
    """
    # Create LangFuse trace — sets contextvar handler so all agent
    # LLM calls in the graph are automatically grouped under this trace.
    create_trace(
        name=f"chapter_{chapter_number}",
        project_id=project_id,
        chapter_number=chapter_number,
    )

    store.set_context(session_id, project_id, chapter_number)
    store.set_config(session_id, config)
    queue = store.get_queue(session_id)
    output: asyncio.Queue = asyncio.Queue()
    running = asyncio.Event()
    running.set()

    drain_task = asyncio.create_task(_background_drain(queue, output, running))

    try:
        yield _sse_event("start", {"message": "开始写作..."})

        async for event in graph.astream_events(initial_state, config, version="v2"):
            # Flush writer chunks
            async for s in _flush_output(output):
                yield s

            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chain_start" and name in _NODE_LABELS:
                yield await _make_progress_event(name, "running")
            elif kind == "on_chain_end" and name in _NODE_LABELS:
                yield await _make_progress_event(name, "done", event)

        # Final flush
        running.clear()
        await drain_task
        async for s in _flush_output(output):
            yield s
        async for s in _drain_queue(queue):
            yield s

        # Check if graph was interrupted (human_review called interrupt())
        # Note: GraphInterrupt may not raise in all LangGraph versions with
        # astream_events v2, so we check get_state() after the loop.
        final_state = await graph.aget_state(config)
        if final_state and final_state.next:
            # Graph is paused at human_review — build payload from state
            vals = final_state.values or {}
            wb = vals.get("worldbuilding_report", {})
            ed = vals.get("editor_report", {})
            ct = vals.get("continuity_report", {})
            yield _sse_event("progress", {
                "node": "human_review", "label": "人工审批", "status": "running",
                "score": None, "detail": None,
            })
            yield _sse_event("review_required", {
                "type": "human_review",
                "chapter_number": chapter_number,
                "draft_preview": vals.get("draft_content", "")[:1000],
                "draft_full": vals.get("draft_content", ""),
                "editor_score": ed.get("overall_score", 0),
                "continuity_score": ct.get("overall_score", 0),
                "editor_issues": ed.get("issues", [])[:10],
                "continuity_issues": ct.get("inconsistencies", [])[:10],
                "wb_new_entities": len(wb.get("new_entities", [])),
                "wb_conflicts": len(wb.get("conflicts", [])),
                "retry_count": vals.get("retry_count", 0),
            })
        else:
            # Graph completed normally
            if final_state and final_state.values:
                _save_chapter_result(mgr, project_id, chapter_number, final_state.values)
                _push_quality_scores(final_state.values)
            yield _sse_event("done", {"chapter_content": "", "status": "completed"})

    except GraphInterrupt as gi:
        running.clear()
        await drain_task
        async for s in _flush_output(output):
            yield s
        async for s in _drain_queue(queue):
            yield s

        interrupt_data = gi.args[0] if gi.args else {}
        yield _sse_event("progress", {
            "node": "human_review", "label": "人工审批", "status": "running",
            "score": None, "detail": None,
        })
        yield _sse_event("review_required", interrupt_data)

    except Exception as e:
        running.clear()
        await drain_task
        yield _sse_event("error", {"message": str(e), "node": "unknown"})
    finally:
        running.clear()
        if not drain_task.done():
            drain_task.cancel()


async def resume_graph(
    store: SessionStore,
    session_id: str,
    feedback: dict,
    mgr: ProjectManager | None = None,
    project_id: str = "",
    chapter_number: int = 0,
):
    """Resume a graph that was paused at Human Review."""
    session = store.get(session_id)
    if not session:
        yield _sse_event("error", {"message": "Session not found"})
        return

    graph = session["graph"]
    config = session.get("config", {})
    queue = session.get("queue")
    output: asyncio.Queue = asyncio.Queue()
    running = asyncio.Event()
    running.set()

    drain_task = asyncio.create_task(
        _background_drain(queue, output, running)
    ) if queue else None

    try:
        yield _sse_event("start", {"message": "继续写作..."})
        yield await _make_progress_event("human_review", "done")

        async for event in graph.astream_events(
            Command(resume=feedback), config, version="v2"
        ):
            async for s in _flush_output(output):
                yield s

            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chain_start" and name in _NODE_LABELS:
                yield await _make_progress_event(name, "running")
            elif kind == "on_chain_end" and name in _NODE_LABELS:
                yield await _make_progress_event(name, "done", event)

        running.clear()
        if drain_task:
            await drain_task
        async for s in _flush_output(output):
            yield s

        # Check if graph was interrupted again (e.g. reject → rewrite → human_review)
        # Note: GraphInterrupt may not raise in all LangGraph versions with
        # astream_events v2, so we check get_state() after the loop.
        final_state = await graph.aget_state(config)
        if final_state and final_state.next:
            vals = final_state.values or {}
            wb = vals.get("worldbuilding_report", {})
            ed = vals.get("editor_report", {})
            ct = vals.get("continuity_report", {})
            yield _sse_event("progress", {
                "node": "human_review", "label": "人工审批", "status": "running",
                "score": None, "detail": None,
            })
            yield _sse_event("review_required", {
                "type": "human_review",
                "chapter_number": chapter_number,
                "draft_preview": vals.get("draft_content", "")[:1000],
                "draft_full": vals.get("draft_content", ""),
                "editor_score": ed.get("overall_score", 0),
                "continuity_score": ct.get("overall_score", 0),
                "editor_issues": ed.get("issues", [])[:10],
                "continuity_issues": ct.get("inconsistencies", [])[:10],
                "wb_new_entities": len(wb.get("new_entities", [])),
                "wb_conflicts": len(wb.get("conflicts", [])),
                "retry_count": vals.get("retry_count", 0),
            })
        else:
            # Graph completed normally
            if final_state and final_state.values:
                _save_chapter_result(mgr, project_id, chapter_number, final_state.values)
                _push_quality_scores(final_state.values)
            yield _sse_event("done", {"chapter_content": "", "status": "completed"})

    except GraphInterrupt as gi:
        running.clear()
        if drain_task:
            await drain_task
        async for s in _flush_output(output):
            yield s

        interrupt_data = gi.args[0] if gi.args else {}
        yield _sse_event("progress", {
            "node": "human_review", "label": "人工审批", "status": "running",
            "score": None, "detail": None,
        })
        yield _sse_event("review_required", interrupt_data)
    except Exception as e:
        running.clear()
        if drain_task:
            await drain_task
        yield _sse_event("error", {"message": str(e), "node": "unknown"})
    finally:
        running.clear()
        if drain_task and not drain_task.done():
            drain_task.cancel()


def _push_quality_scores(state_values: dict) -> None:
    """Extract editor and continuity scores from graph state and push to LangFuse."""
    scores = {}
    ed = state_values.get("editor_report", {})
    ct = state_values.get("continuity_report", {})
    if isinstance(ed, dict) and "overall_score" in ed:
        scores["editor_score"] = float(ed["overall_score"])
    if isinstance(ct, dict) and "overall_score" in ct:
        scores["continuity_score"] = float(ct["overall_score"])
    if scores:
        score_trace(scores)


def _save_chapter_result(
    mgr: ProjectManager, project_id: str, chapter_number: int, result: dict
) -> None:
    """Persist the completed chapter to storage."""

    draft = result.get("draft_content", "")
    wb_report = result.get("worldbuilding_report", {})
    editor_report = result.get("editor_report", {})
    continuity_report = result.get("continuity_report", {})
    outline = result.get("chapter_outline", "")

    status = "approved" if result.get("human_approved") else "draft"

    mgr.save_chapter(
        project_id=project_id,
        chapter_number=chapter_number,
        outline=outline,
        draft_content=draft,
        status=status,
        editor_report=json.dumps(editor_report, ensure_ascii=False),
        continuity_report=json.dumps(continuity_report, ensure_ascii=False),
    )

    # Save worldbuilding report to chapter record
    mgr.update_chapter_worldbuilding(project_id, chapter_number, wb_report)

    # Save world entities
    if wb_report:
        mgr.save_world_entities(project_id, wb_report)
