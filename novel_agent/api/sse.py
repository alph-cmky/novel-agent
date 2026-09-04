"""SSE streaming and session management for the writing pipeline."""

import asyncio
import json
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from novel_agent.api.run_service import ChapterRunService
from novel_agent.graph.runner import (
    GRAPH_INTERRUPT_EVENT,
    chapter_run_context,
    finalize_chapter,
    iterate_chapter_events,
)
from novel_agent.schema.enums import ChapterStatus, RunStatus
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
            "run_id": None,
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

    def set_context(
        self,
        session_id: str,
        project_id: str,
        chapter_number: int,
        run_id: str | None = None,
    ) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["project_id"] = project_id
            self._sessions[session_id]["chapter_number"] = chapter_number
            self._sessions[session_id]["run_id"] = run_id

    def find_session(self, project_id: str, chapter_number: int) -> str | None:
        for sid, s in self._sessions.items():
            if s.get("project_id") == project_id and s.get("chapter_number") == chapter_number:
                return sid
        return None

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _review_payload(values: dict, chapter_number: int) -> dict:
    wb = values.get("worldbuilding_report", {}) or {}
    ed = values.get("editor_report", {}) or {}
    ct = values.get("continuity_report", {}) or {}
    return {
        "type": "human_review",
        "chapter_number": chapter_number,
        "draft_preview": values.get("draft_content", "")[:1000],
        "draft_full": values.get("draft_content", ""),
        "editor_score": ed.get("overall_score", 0),
        "continuity_score": ct.get("overall_score", 0),
        "editor_issues": ed.get("issues", [])[:10],
        "continuity_issues": ct.get("inconsistencies", [])[:10],
        "wb_new_entities": len(wb.get("new_entities", [])),
        "wb_conflicts": len(wb.get("conflicts", [])),
        "evolution_rounds": len(values.get("evolution_history", [])),
        "evolution_termination": values.get("evolution_termination", ""),
    }


async def replay_review(values: dict, chapter_number: int):
    """Re-send a persisted human-review checkpoint after a process restart."""
    yield _sse_event("start", {"message": "恢复人工审批..."})
    yield _sse_event(
        "progress",
        {
            "node": "human_review",
            "label": "人工审批",
            "status": "running",
            "score": None,
            "detail": None,
        },
    )
    yield _sse_event("review_required", _review_payload(values, chapter_number))


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
    "worldbuilding": "世界观提取",
    "human_review": "人工审批",
    "evolution_writer": "内容创作",
    "evolution_editor": "编辑审查",
    "evolution_continuity": "一致性审计",
    "evolution_orchestrator": "进化评估",
    "evolution_select_best": "选择最佳版本",
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


async def _make_progress_event(name: str, status: str, event: dict | None = None) -> str:
    """Build a progress SSE event from a node name and status."""
    label = _NODE_LABELS.get(name, name)
    score = None
    detail = None
    meta = None

    if status == "done" and event:
        output = event.get("data", {}).get("output", {})
        if name in ("editor", "evolution_editor"):
            report = output.get("editor_report", {})
            score = report.get("overall_score")
            detail = report.get("verdict")
        elif name in ("continuity", "evolution_continuity"):
            report = output.get("continuity_report", {})
            score = report.get("overall_score")
        elif name == "worldbuilding":
            report = output.get("worldbuilding_report", {})
            n_ent = len(report.get("new_entities", []))
            n_conf = len(report.get("conflicts", []))
            detail = f"实体:{n_ent} 冲突:{n_conf}"
        elif name == "evolution_orchestrator":
            # Include evolution-specific metadata
            history = output.get("evolution_history", [])
            if history:
                last = history[-1]
                meta = {
                    "version": last.get("v"),
                    "editor": last.get("editor"),
                    "continuity": last.get("continuity"),
                    "composite": last.get("composite"),
                    "delta": last.get("delta"),
                    "termination": output.get("evolution_termination", ""),
                }
                score = last.get("composite")
                term = output.get("evolution_termination", "")
                detail = f"v{last.get('v')} E:{last.get('editor')} C:{last.get('continuity')}"
                if term:
                    detail += f" 终止:{term}"

    return _sse_event(
        "progress",
        {
            "node": name,
            "label": label,
            "status": status,
            "score": score,
            "detail": detail,
            "meta": meta,
        },
    )


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
    and run_chapter event iteration for node-level progress events.
    """
    store.set_context(session_id, project_id, chapter_number)
    store.set_config(session_id, config)
    queue = store.get_queue(session_id)
    output: asyncio.Queue = asyncio.Queue()
    running = asyncio.Event()
    running.set()

    drain_task = asyncio.create_task(_background_drain(queue, output, running))
    current_node: str | None = None

    try:
        run_id = initial_state.get("writing_run_id")
        if run_id:
            mgr.update_writing_run(
                run_id,
                status=RunStatus.RUNNING.value,
                started_at=datetime.now(UTC).isoformat(),
            )
            store.set_context(session_id, project_id, chapter_number, run_id)
        yield _sse_event("start", {"message": "开始写作..."})

        interrupt_payload: dict | None = None
        async with chapter_run_context(graph, config=config, state=initial_state) as ctx:
            async for event in iterate_chapter_events(graph, ctx.payload, ctx.config):
                async for s in _flush_output(output):
                    yield s
                if event.get("event") == GRAPH_INTERRUPT_EVENT:
                    data = event.get("data")
                    interrupt_payload = data if isinstance(data, dict) else {}
                    continue
                kind = event["event"]
                name = event.get("name", "")
                if kind == "on_chain_start" and name in _NODE_LABELS:
                    current_node = name
                    if run_id:
                        mgr.update_writing_run(
                            run_id,
                            status=RunStatus.RUNNING.value,
                            current_node=name,
                        )
                    yield await _make_progress_event(name, "running")
                elif kind == "on_chain_end" and name in _NODE_LABELS:
                    yield await _make_progress_event(name, "done", event)

            running.clear()
            await drain_task
            async for s in _flush_output(output):
                yield s
            async for s in _drain_queue(queue):
                yield s

            outcome = await finalize_chapter(
                graph,
                ctx.config,
                interrupt_payload=interrupt_payload,
                fallback_trace_id=ctx.trace_id,
            )
            ctx.handle.record_outcome(outcome.values, interrupted=outcome.interrupted)
        if outcome.interrupted:
            vals = outcome.values
            if run_id:
                mgr.update_writing_run(
                    run_id,
                    status=RunStatus.WAITING_REVIEW.value,
                    current_node="human_review",
                )
            yield _sse_event(
                "progress",
                {
                    "node": "human_review",
                    "label": "人工审批",
                    "status": "running",
                    "score": None,
                    "detail": None,
                },
            )
            yield _sse_event(
                "review_required",
                outcome.interrupt_payload or _review_payload(vals, chapter_number),
            )
        else:
            if outcome.values:
                _save_chapter_result(mgr, project_id, chapter_number, outcome.values)
            status = "approved" if outcome.values.get("human_approved") else "draft"
            yield _sse_event("done", {"chapter_content": "", "status": status})
            store.remove(session_id)

    except Exception as e:
        running.clear()
        await drain_task
        traceback.print_exc()
        mgr.mark_chapter_failed(project_id, chapter_number)
        run_id = initial_state.get("writing_run_id")
        if run_id:
            mgr.update_writing_run(
                run_id,
                status=RunStatus.FAILED.value,
                error_code=type(e).__name__,
                error_message=str(e),
                finished_at=datetime.now(UTC).isoformat(),
            )
        yield _sse_event("error", {"message": str(e), "node": current_node})
        store.remove(session_id)
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
    run_id = session.get("run_id")
    queue = session.get("queue")
    output: asyncio.Queue = asyncio.Queue()
    running = asyncio.Event()
    running.set()

    drain_task = asyncio.create_task(_background_drain(queue, output, running)) if queue else None
    current_node: str | None = None

    try:
        if run_id and mgr:
            mgr.update_writing_run(
                run_id,
                status=RunStatus.RUNNING.value,
                current_node="human_review",
            )
        yield _sse_event("start", {"message": "继续写作..."})
        yield await _make_progress_event("human_review", "done")

        interrupt_payload: dict | None = None
        async with chapter_run_context(graph, config=config, resume=feedback) as ctx:
            async for event in iterate_chapter_events(graph, ctx.payload, ctx.config):
                async for s in _flush_output(output):
                    yield s
                if event.get("event") == GRAPH_INTERRUPT_EVENT:
                    data = event.get("data")
                    interrupt_payload = data if isinstance(data, dict) else {}
                    continue
                kind = event["event"]
                name = event.get("name", "")
                if kind == "on_chain_start" and name in _NODE_LABELS:
                    current_node = name
                    if run_id and mgr:
                        mgr.update_writing_run(
                            run_id,
                            status=RunStatus.RUNNING.value,
                            current_node=name,
                        )
                    yield await _make_progress_event(name, "running")
                elif kind == "on_chain_end" and name in _NODE_LABELS:
                    yield await _make_progress_event(name, "done", event)

            running.clear()
            if drain_task:
                await drain_task
            async for s in _flush_output(output):
                yield s
            if queue:
                async for s in _drain_queue(queue):
                    yield s

            outcome = await finalize_chapter(
                graph,
                ctx.config,
                interrupt_payload=interrupt_payload,
                fallback_trace_id=ctx.trace_id,
            )
            ctx.handle.record_outcome(outcome.values, interrupted=outcome.interrupted)
        if outcome.interrupted:
            vals = outcome.values
            if run_id and mgr:
                mgr.update_writing_run(
                    run_id,
                    status=RunStatus.WAITING_REVIEW.value,
                    current_node="human_review",
                )
            yield _sse_event(
                "progress",
                {
                    "node": "human_review",
                    "label": "人工审批",
                    "status": "running",
                    "score": None,
                    "detail": None,
                },
            )
            yield _sse_event(
                "review_required",
                outcome.interrupt_payload or _review_payload(vals, chapter_number),
            )
        else:
            if outcome.values:
                _save_chapter_result(mgr, project_id, chapter_number, outcome.values)
            status = "approved" if outcome.values.get("human_approved") else "draft"
            yield _sse_event("done", {"chapter_content": "", "status": status})
            store.remove(session_id)

    except Exception as e:
        running.clear()
        if drain_task:
            await drain_task
        if queue:
            async for s in _drain_queue(queue):
                yield s
        traceback.print_exc()
        if mgr:
            mgr.mark_chapter_failed(project_id, chapter_number)
            if run_id:
                mgr.update_writing_run(
                    run_id,
                    status=RunStatus.FAILED.value,
                    error_code=type(e).__name__,
                    error_message=str(e),
                    finished_at=datetime.now(UTC).isoformat(),
                )
        yield _sse_event("error", {"message": str(e), "node": current_node})
        store.remove(session_id)
    finally:
        running.clear()
        if drain_task and not drain_task.done():
            drain_task.cancel()


def _save_foreshadowings(
    mgr: ProjectManager,
    project_id: str,
    chapter_number: int,
    wb_report: dict,
) -> None:
    """Persist new and resolved foreshadowings from worldbuilding report."""
    # Upsert open foreshadowings so progress does not create duplicates.
    for fs in wb_report.get("foreshadowings", []) or []:
        if not isinstance(fs, dict) or not fs.get("description"):
            continue
        try:
            description = str(fs["description"])
            planted = int(fs.get("planted_chapter", chapter_number))
            changed = mgr.update_foreshadowing_status(
                project_id,
                description,
                planted,
                status="open",
                expected_resolve_chapter=(
                    int(fs["expected_resolve_chapter"])
                    if fs.get("expected_resolve_chapter")
                    else None
                ),
                risk_level=str(fs.get("risk_level", "medium")),
                action_needed=str(fs.get("action_needed", "maintain")),
                reader_knows=bool(fs.get("reader_knows", False)),
                characters_aware=fs.get("characters_aware", []),
                characters_unaware=fs.get("characters_unaware", []),
            )
            if not changed:
                mgr.add_foreshadowing(
                    project_id=project_id,
                    description=description,
                    planted_chapter=planted,
                    expected_resolve_chapter=(
                        int(fs["expected_resolve_chapter"])
                        if fs.get("expected_resolve_chapter")
                        else None
                    ),
                    risk_level=str(fs.get("risk_level", "medium")),
                    action_needed=str(fs.get("action_needed", "maintain")),
                    reader_knows=bool(fs.get("reader_knows", False)),
                    characters_aware=fs.get("characters_aware", []),
                    characters_unaware=fs.get("characters_unaware", []),
                )
        except Exception:
            pass

    # Resolve existing foreshadowings
    for fs in wb_report.get("resolved_foreshadowings", []) or []:
        if not isinstance(fs, dict) or not fs.get("description"):
            continue
        try:
            mgr.update_foreshadowing_status(
                project_id=project_id,
                description=str(fs["description"]),
                planted_chapter=(
                    int(fs["planted_chapter"]) if fs.get("planted_chapter") is not None else None
                ),
                status="resolved",
                resolved_chapter=chapter_number,
            )
        except Exception:
            pass


def _save_chapter_result(
    mgr: ProjectManager, project_id: str, chapter_number: int, result: dict
) -> None:
    """Persist the completed chapter to storage, including evolution data."""

    draft = result.get("draft_content", "")
    wb_report = result.get("worldbuilding_report", {})
    editor_report = result.get("editor_report", {})
    continuity_report = result.get("continuity_report", {})
    outline = result.get("chapter_outline", "")

    approved = bool(result.get("human_approved"))
    run_id = result.get("writing_run_id")
    version_record = None
    if run_id:
        version_record = mgr.create_chapter_version(
            project_id,
            chapter_number,
            draft,
            run_id=run_id,
            origin="evolution" if result.get("evolution_history") else "initial_generation",
            scene_plan=result.get("scene_plan", []),
            scene_drafts=result.get("scene_drafts", []),
        )
        mgr.update_writing_run(
            run_id,
            current_version_id=version_record["id"],
            status=RunStatus.WAITING_REVIEW.value,
        )
    # Commit the durable record as a draft first. Approval is published only
    # after world state and vector indexing have succeeded.
    status = ChapterStatus.DRAFT.value

    # Build evolution summary from state
    evolution_history = result.get("evolution_history", [])
    version = 0
    evolution_summary = "{}"
    if evolution_history:
        version = len(evolution_history)
        evolution_summary = json.dumps(
            {
                "total_rounds": len(evolution_history),
                "best_version": result.get("evolution_best_candidate_version", 0),
                "termination": result.get("evolution_termination", ""),
                "score_history": evolution_history,
            },
            ensure_ascii=False,
        )

    mgr.save_chapter(
        project_id=project_id,
        chapter_number=chapter_number,
        outline=outline,
        draft_content=draft,
        status=status,
        editor_report=json.dumps(editor_report, ensure_ascii=False),
        continuity_report=json.dumps(continuity_report, ensure_ascii=False),
        version=version,
        evolution_summary=evolution_summary,
        index=False,
    )

    # Save worldbuilding report to chapter record
    mgr.update_chapter_worldbuilding(project_id, chapter_number, wb_report)

    if wb_report:
        if run_id:
            # V2 runs produce a proposal; only Canon Commit may mutate the
            # formal entity/relation/foreshadowing tables.
            mgr.create_canon_proposal(
                project_id,
                chapter_number,
                "worldbuilding",
                wb_report,
                run_id=run_id,
                version_id=version_record["id"] if version_record else None,
            )
        else:
            # Runs without a version record use direct-write persistence.
            mgr.save_world_entities(project_id, wb_report, chapter_number)
            mgr.save_world_relations(project_id, chapter_number, wb_report)
            _save_foreshadowings(mgr, project_id, chapter_number, wb_report)

    if run_id and approved:
        for proposal in mgr.list_canon_proposals(project_id, run_id=run_id, status="proposed"):
            mgr.review_canon_proposal(proposal["id"], "accepted", "章节已批准")
        ChapterRunService(mgr).commit(run_id)
        return

    if approved:
        # Index before publishing the approved status. A vector-store failure
        # therefore leaves a retryable draft instead of a false approval.
        mgr.save_chapter(
            project_id=project_id,
            chapter_number=chapter_number,
            outline=outline,
            draft_content=draft,
            status=ChapterStatus.DRAFT.value,
            editor_report=json.dumps(editor_report, ensure_ascii=False),
            continuity_report=json.dumps(continuity_report, ensure_ascii=False),
            version=version,
            evolution_summary=evolution_summary,
            index=True,
        )
        mgr.save_chapter(
            project_id=project_id,
            chapter_number=chapter_number,
            outline=outline,
            draft_content=draft,
            status=ChapterStatus.APPROVED.value,
            editor_report=json.dumps(editor_report, ensure_ascii=False),
            continuity_report=json.dumps(continuity_report, ensure_ascii=False),
            version=version,
            evolution_summary=evolution_summary,
            index=False,
        )
        if version_record:
            mgr.commit_chapter_version(version_record["id"])
