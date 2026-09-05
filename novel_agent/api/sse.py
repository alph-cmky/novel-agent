"""SSE streaming and session management for the C5 agent loop."""

import asyncio
import json
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from novel_agent.api.run_service import ChapterRunService
from novel_agent.graph.runner import run_chapter_agent_loop
from novel_agent.schema.enums import ChapterStatus, RunStatus
from novel_agent.storage.manager import ProjectManager


class SessionStore:
    """In-memory store for active writing sessions.

    Each session holds the initial state and an asyncio.Queue for
    streaming progress events from the agent loop to SSE.
    """

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, queue: asyncio.Queue) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = {
            "queue": queue,
            "project_id": None,
            "chapter_number": None,
            "run_id": None,
            "values": None,
        }
        return session_id

    def get(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def get_queue(self, session_id: str) -> asyncio.Queue | None:
        s = self._sessions.get(session_id)
        return s["queue"] if s else None

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

    def set_values(self, session_id: str, values: dict) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["values"] = values

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
    qg = values.get("quality_gate_report", {}) or {}
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
        "quality_gate_passed": qg.get("passed", False),
        "loop_retries": values.get("loop_retries", 0),
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


async def create_sse_stream(
    store: SessionStore,
    session_id: str,
    initial_state: dict,
    mgr: ProjectManager,
    project_id: str,
    chapter_number: int,
):
    """Run the C5 agent loop and stream progress events via SSE.

    The loop runs to completion (Orchestrator → Writer loop → Editor →
    Continuity → Worldbuilding), then emits a review_required event.
    Human review is a post-generation approve/reject step.
    """
    store.set_context(session_id, project_id, chapter_number)

    run_id = initial_state.get("writing_run_id")
    if run_id:
        mgr.update_writing_run(
            run_id,
            status=RunStatus.RUNNING.value,
            started_at=datetime.now(UTC).isoformat(),
        )
        store.set_context(session_id, project_id, chapter_number, run_id)

    yield _sse_event("start", {"message": "开始写作..."})

    try:
        outcome = await run_chapter_agent_loop(
            initial_state,
            source="api",
        )
        values = outcome.values

        # Emit progress events for each stage
        yield _sse_event(
            "progress",
            {
                "node": "orchestrator",
                "label": "策略规划",
                "status": "done",
                "score": None,
                "detail": values.get("orchestrator_strategy", {}).get("narrative_stage", ""),
            },
        )
        yield _sse_event(
            "progress",
            {
                "node": "writer",
                "label": "内容创作",
                "status": "done",
                "score": None,
                "detail": f"{len(values.get('draft_content', ''))} 字",
            },
        )
        ed = values.get("editor_report", {}) or {}
        yield _sse_event(
            "progress",
            {
                "node": "editor",
                "label": "编辑审查",
                "status": "done",
                "score": ed.get("overall_score"),
                "detail": ed.get("verdict", ""),
            },
        )
        ct = values.get("continuity_report", {}) or {}
        yield _sse_event(
            "progress",
            {
                "node": "continuity",
                "label": "一致性审计",
                "status": "done",
                "score": ct.get("overall_score"),
                "detail": None,
            },
        )
        wb = values.get("worldbuilding_report", {}) or {}
        yield _sse_event(
            "progress",
            {
                "node": "worldbuilding",
                "label": "世界观提取",
                "status": "done",
                "score": None,
                "detail": (
                    f"实体:{len(wb.get('new_entities', []))} 冲突:{len(wb.get('conflicts', []))}"
                ),
            },
        )

        store.set_values(session_id, values)

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
        yield _sse_event("review_required", _review_payload(values, chapter_number))

    except Exception as e:
        traceback.print_exc()
        mgr.mark_chapter_failed(project_id, chapter_number)
        if run_id:
            mgr.update_writing_run(
                run_id,
                status=RunStatus.FAILED.value,
                error_code=type(e).__name__,
                error_message=str(e),
                finished_at=datetime.now(UTC).isoformat(),
            )
        yield _sse_event("error", {"message": str(e)})
        store.remove(session_id)


async def resume_graph(
    store: SessionStore,
    session_id: str,
    feedback: dict,
    mgr: ProjectManager | None = None,
    project_id: str = "",
    chapter_number: int = 0,
):
    """Handle human review feedback: approve or reject-and-revise.

    feedback["action"] == "approve": commit the chapter.
    feedback["action"] == "reject": re-run the agent loop with feedback.
    """
    session = store.get(session_id)
    if not session:
        yield _sse_event("error", {"message": "Session not found"})
        return

    run_id = session.get("run_id")
    values = session.get("values") or {}

    try:
        action = feedback.get("action", "approve")

        if action == "approve":
            values["human_approved"] = True
            if mgr and values:
                _save_chapter_result(mgr, project_id, chapter_number, values)
            yield _sse_event("done", {"chapter_content": "", "status": "approved"})
            store.remove(session_id)
            return

        # Reject: re-run with feedback
        comments = feedback.get("comments", "")
        yield _sse_event("start", {"message": "根据反馈修订..."})

        revision_state = dict(values)
        revision_state["revision_feedback"] = comments
        revision_state.pop("draft_content", None)

        outcome = await run_chapter_agent_loop(revision_state, source="api")
        new_values = outcome.values

        store.set_values(session_id, new_values)

        yield _sse_event(
            "progress",
            {
                "node": "writer",
                "label": "内容修订",
                "status": "done",
                "score": None,
                "detail": f"{len(new_values.get('draft_content', ''))} 字",
            },
        )

        if run_id and mgr:
            mgr.update_writing_run(
                run_id,
                status=RunStatus.WAITING_REVIEW.value,
                current_node="human_review",
            )
        yield _sse_event("review_required", _review_payload(new_values, chapter_number))

    except Exception as e:
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
        yield _sse_event("error", {"message": str(e)})
        store.remove(session_id)


def _save_foreshadowings(
    mgr: ProjectManager,
    project_id: str,
    chapter_number: int,
    wb_report: dict,
) -> None:
    """Persist new and resolved foreshadowings from worldbuilding report."""
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
    """Persist the completed chapter to storage."""

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
            origin="agent_loop",
            scene_plan=result.get("scene_plan", []),
            scene_drafts=result.get("scene_drafts", []),
        )
        mgr.update_writing_run(
            run_id,
            current_version_id=version_record["id"],
            status=RunStatus.WAITING_REVIEW.value,
        )

    status = ChapterStatus.DRAFT.value
    mgr.save_chapter(
        project_id=project_id,
        chapter_number=chapter_number,
        outline=outline,
        draft_content=draft,
        status=status,
        editor_report=json.dumps(editor_report, ensure_ascii=False),
        continuity_report=json.dumps(continuity_report, ensure_ascii=False),
        version=0,
        evolution_summary="{}",
        index=False,
    )

    mgr.update_chapter_worldbuilding(project_id, chapter_number, wb_report)

    if wb_report:
        if run_id:
            mgr.create_canon_proposal(
                project_id,
                chapter_number,
                "worldbuilding",
                wb_report,
                run_id=run_id,
                version_id=version_record["id"] if version_record else None,
            )
        else:
            mgr.save_world_entities(project_id, wb_report, chapter_number)
            mgr.save_world_relations(project_id, chapter_number, wb_report)
            _save_foreshadowings(mgr, project_id, chapter_number, wb_report)

    if run_id and approved:
        for proposal in mgr.list_canon_proposals(project_id, run_id=run_id, status="proposed"):
            mgr.review_canon_proposal(proposal["id"], "accepted", "章节已批准")
        ChapterRunService(mgr).commit(run_id)
        return

    if approved:
        mgr.save_chapter(
            project_id=project_id,
            chapter_number=chapter_number,
            outline=outline,
            draft_content=draft,
            status=ChapterStatus.DRAFT.value,
            editor_report=json.dumps(editor_report, ensure_ascii=False),
            continuity_report=json.dumps(continuity_report, ensure_ascii=False),
            version=0,
            evolution_summary="{}",
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
            version=0,
            evolution_summary="{}",
            index=False,
        )
        if version_record:
            mgr.commit_chapter_version(version_record["id"])
