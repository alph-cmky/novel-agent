"""REST API routes for novel-agent Web UI."""

import asyncio
import difflib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from novel_agent.agents.continuity import ContinuityAgent
from novel_agent.agents.editor import EditorAgent
from novel_agent.api.graph_data import build_graph_data
from novel_agent.api.outline import generate_outline
from novel_agent.api.run_service import ChapterRunService
from novel_agent.api.sse import (
    SessionStore,
    create_sse_stream,
    replay_review,
    resume_graph,
)
from novel_agent.graph.chapter import (
    _config_for,
    _get_chapter_store,
    build_chapter_graph_async,
)
from novel_agent.model_router import TaskClass
from novel_agent.schema.enums import ChapterStatus, OutlineStatus
from novel_agent.services.context import ContextCompiler
from novel_agent.storage.manager import ProjectManager

router = APIRouter()
session_store = SessionStore()


def _get_persist_dir() -> Path:
    return Path(os.getenv("NOVEL_DATA_DIR", "./novel-data"))


def _get_manager() -> ProjectManager:
    return ProjectManager(_get_persist_dir())


def _get_run_service() -> ChapterRunService:
    return ChapterRunService(_get_manager())


async def _restore_session(project_id: str, chapter_number: int) -> str | None:
    """Recreate the in-memory SSE handle from a durable pending checkpoint."""
    persist_dir = str(_get_persist_dir())
    graph = await build_chapter_graph_async(persist_dir=persist_dir)
    thread_id = f"{project_id}:ch{chapter_number}"
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    if not state or not state.next:
        return None
    session_id = session_store.create(graph, asyncio.Queue())
    runs = _get_manager().list_writing_runs(project_id, chapter_number)
    active_run = next(
        (
            run
            for run in runs
            if run["status"] in {"queued", "running", "waiting_review", "waiting_user", "retrying"}
        ),
        None,
    )
    session_store.set_config(session_id, config)
    session_store.set_context(
        session_id,
        project_id,
        chapter_number,
        active_run["id"] if active_run else None,
    )
    return session_id


# ── Request / Response models ──────────────────────────


class CreateProjectRequest(BaseModel):
    name: str
    title: str = ""
    genre: str = ""
    story_length: str = "long"
    target_chapter_words: int = 3000
    outline_text: str = ""
    world_setting: str = ""
    outline: str = ""  # story synopsis for AI outline generation
    narrative_mode: str | None = None
    narrative_perspective: str = ""


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    title: str | None = None
    genre: str | None = None
    story_length: str | None = None
    target_chapter_words: int | None = None
    world_setting: str | None = None
    narrative_mode: str | None = None
    narrative_perspective: str | None = None


class SaveOutlineRequest(BaseModel):
    chapters: list[dict]


class RejectRequest(BaseModel):
    comments: str = ""


class SaveDraftRequest(BaseModel):
    content: str


class CreateRunRequest(BaseModel):
    run_type: str = "draft"
    workflow_version: str = "v2"


class CreateVersionRequest(BaseModel):
    content: str
    origin: str = "initial_generation"
    status: str = "candidate"


class ProposalReviewRequest(BaseModel):
    reviewer_note: str = ""


class SceneRewriteRequest(BaseModel):
    content: str


# ── Projects ───────────────────────────────────────────


@router.get("/projects")
async def list_projects():
    mgr = _get_manager()
    return mgr.list_projects_with_progress()


@router.post("/projects")
async def create_project(req: CreateProjectRequest):
    mgr = _get_manager()
    project_id = mgr.init_project(
        name=req.name,
        title=req.title or req.name,
        genre=req.genre,
        story_length=req.story_length,
        target_chapter_words=req.target_chapter_words,
        world_setting=req.world_setting,
        outline=req.outline,
        narrative_mode=req.narrative_mode,
        narrative_perspective=req.narrative_perspective,
    )
    if req.outline_text:
        # Parse outline_text into chapters (simple format: one line per chapter)
        lines = [line.strip() for line in req.outline_text.split("\n") if line.strip()]
        chapters = [
            {
                "chapter_number": i + 1,
                "title": line[:50],
                "summary": line,
                "status": OutlineStatus.PENDING.value,
                "sort_order": i + 1,
            }
            for i, line in enumerate(lines)
        ]
        if chapters:
            mgr.save_outline(project_id, chapters)
    return mgr.get_project_with_progress(project_id)


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    mgr = _get_manager()
    project = mgr.get_project_with_progress(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if updates:
        mgr.update_project(project_id, **updates)
    return mgr.get_project_with_progress(project_id)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    mgr.delete_project(project_id)
    return {"ok": True}


# ── Outline ────────────────────────────────────────────


@router.get("/projects/{project_id}/outline")
async def get_outline(project_id: str):
    mgr = _get_manager()
    return mgr.get_outline(project_id)


@router.put("/projects/{project_id}/outline")
async def save_outline(project_id: str, req: SaveOutlineRequest):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    mgr.save_outline(project_id, req.chapters)
    return {"ok": True}


@router.post("/projects/{project_id}/outline/generate")
async def generate_outline_endpoint(project_id: str):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        chapters = await generate_outline(mgr, project_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"大纲生成失败: {str(e)}",
        ) from e
    return chapters


# ── Graph ──────────────────────────────────────────────


@router.get("/projects/{project_id}/graph")
async def get_graph(project_id: str, until_chapter: int = Query(0)):
    mgr = _get_manager()
    return build_graph_data(mgr, project_id, until_chapter)


# ── Chapters ───────────────────────────────────────────


@router.get("/projects/{project_id}/chapters")
async def list_chapters(project_id: str):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return mgr.get_all_chapters(project_id)


@router.get("/projects/{project_id}/chapters/{chapter_number}")
async def get_chapter(project_id: str, chapter_number: int):
    mgr = _get_manager()
    chapter = mgr.get_chapter(project_id, chapter_number)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    # Include context data for the writing page
    ctx = ContextCompiler(mgr).compile(project_id, chapter_number).to_state()
    outline_items = mgr.get_outline(project_id)
    for item in outline_items:
        if item["chapter_number"] == chapter_number:
            chapter["chapter_outline"] = item.get("summary", "") or item.get("title", "")
            break
    chapter["character_context"] = ctx.get("character_context", "")
    chapter["world_context"] = ctx.get("world_context", "")
    chapter["recent_summary"] = ctx.get("recent_summary", "")
    return chapter


@router.post("/projects/{project_id}/chapters/{chapter_number}/runs")
async def create_writing_run(
    project_id: str,
    chapter_number: int,
    req: CreateRunRequest,
):
    """Create a durable V2 run without starting the legacy graph yet."""
    mgr = _get_manager()
    if not mgr.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        return _get_run_service().create_run(
            project_id,
            chapter_number,
            run_type=req.run_type,
            workflow_version=req.workflow_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
async def get_writing_run(run_id: str):
    run = _get_manager().get_writing_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/runs/{run_id}/versions")
async def create_chapter_version(run_id: str, req: CreateVersionRequest):
    mgr = _get_manager()
    run = mgr.get_writing_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return _get_run_service().attach_candidate(
            run_id,
            req.content,
            origin=req.origin,
            status=req.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/commit")
async def commit_writing_run(run_id: str):
    mgr = _get_manager()
    run = mgr.get_writing_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    version_id = run.get("current_version_id")
    if not version_id:
        raise HTTPException(status_code=409, detail="Run has no candidate version")
    try:
        return _get_run_service().commit(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/runs/{run_id}/proposals")
async def list_run_proposals(run_id: str):
    mgr = _get_manager()
    run = mgr.get_writing_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return mgr.list_canon_proposals(project_id=run["project_id"], run_id=run_id)


@router.get("/projects/{project_id}/chapters/{chapter_number}/scenes")
async def list_chapter_scenes(project_id: str, chapter_number: int):
    mgr = _get_manager()
    chapter = mgr.get_chapter(project_id, chapter_number)
    if not chapter or not chapter.get("current_version_id"):
        return []
    return mgr.get_scene_manifest(chapter["current_version_id"])


@router.post("/runs/{run_id}/scenes/{scene_index}/rewrite")
async def rewrite_scene(
    run_id: str,
    scene_index: int,
    req: SceneRewriteRequest,
):
    mgr = _get_manager()
    run = mgr.get_writing_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return _get_run_service().rewrite_scene(run_id, scene_index, req.content)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/versions/{version_id}/diff")
async def diff_chapter_version(version_id: str):
    mgr = _get_manager()
    version = mgr.get_chapter_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    parent_id = version.get("parent_version_id")
    parent = mgr.get_chapter_version(parent_id) if parent_id else None
    diff = difflib.unified_diff(
        (parent["content"] if parent else "").splitlines(),
        version["content"].splitlines(),
        fromfile=f"version:{parent_id or 'empty'}",
        tofile=f"version:{version_id}",
        lineterm="",
    )
    return {"version_id": version_id, "parent_version_id": parent_id, "diff": list(diff)}


@router.post("/versions/{version_id}/scenes/{scene_index}/review")
async def review_scene_version(version_id: str, scene_index: int):
    mgr = _get_manager()
    version = mgr.get_chapter_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    scene = next(
        (
            item
            for item in mgr.get_scene_manifest(version_id)
            if item.get("scene_index") == scene_index
        ),
        None,
    )
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    project = mgr.get_project(version["project_id"]) or {}
    draft = scene.get("content", "")
    editor = EditorAgent(config=_config_for(TaskClass.REVIEW))
    editor_report, _ = await editor.review(
        chapter_number=version["chapter_number"],
        draft_content=draft,
        narrative_mode=project.get("narrative_mode"),
    )
    continuity = ContinuityAgent(
        config=_config_for(TaskClass.REVIEW),
        chapter_store=_get_chapter_store(str(_get_persist_dir())),
        project_id=version["project_id"],
    )
    continuity_report, _ = await continuity.audit(
        chapter_number=version["chapter_number"],
        draft_content=draft,
        narrative_mode=project.get("narrative_mode"),
    )
    valid = not editor_report.get("unavailable") and not continuity_report.get("unavailable")
    return {
        "version_id": version_id,
        "scene_index": scene_index,
        "valid": valid,
        "editor_report": editor_report,
        "continuity_report": continuity_report,
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_writing_run(run_id: str):
    try:
        return _get_run_service().cancel(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/runs/{run_id}/retry")
async def retry_writing_run(run_id: str):
    try:
        return _get_run_service().retry(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/projects/{project_id}/chapters/{chapter_number}/versions")
async def list_chapter_versions(project_id: str, chapter_number: int):
    mgr = _get_manager()
    if not mgr.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return _get_run_service().list_versions(project_id, chapter_number)


@router.post("/versions/{version_id}/restore")
async def restore_chapter_version(version_id: str):
    try:
        return _get_run_service().restore_version(version_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/projects/{project_id}/canon")
async def get_project_canon(project_id: str):
    mgr = _get_manager()
    if not mgr.get_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "entities": mgr.get_all_world_entities(project_id),
        "relations": mgr.get_all_world_relations(project_id),
        "foreshadowings": mgr.get_foreshadowings(project_id),
    }


@router.get("/projects/{project_id}/outbox")
async def list_outbox_events(project_id: str, status: str | None = None):
    return _get_manager().list_outbox_events(project_id, status=status)


@router.get("/projects/{project_id}/events")
async def list_story_events(
    project_id: str,
    chapter_number: int | None = Query(None),
):
    return _get_manager().get_story_events(project_id, chapter_number)


@router.post("/outbox/{event_id}/process")
async def process_outbox_event(event_id: str):
    try:
        return _get_manager().process_outbox_event(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/outbox/{event_id}/retry")
async def retry_outbox_event(event_id: str):
    try:
        return _get_manager().retry_outbox_event(event_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/proposals/{proposal_id}/accept")
async def accept_canon_proposal(
    proposal_id: str,
    req: ProposalReviewRequest,
):
    try:
        return _get_manager().review_canon_proposal(
            proposal_id,
            "accepted",
            req.reviewer_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/proposals/{proposal_id}/reject")
async def reject_canon_proposal(
    proposal_id: str,
    req: ProposalReviewRequest,
):
    try:
        return _get_manager().review_canon_proposal(
            proposal_id,
            "rejected",
            req.reviewer_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/projects/{project_id}/chapters/{chapter_number}")
async def delete_chapter(project_id: str, chapter_number: int):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    mgr.delete_chapter(project_id, chapter_number)
    return {"ok": True}


@router.post("/projects/{project_id}/chapters/{chapter_number}/write")
async def write_chapter(project_id: str, chapter_number: int):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get outline for this chapter
    outline_items = mgr.get_outline(project_id)
    chapter_outline = ""
    for item in outline_items:
        if item["chapter_number"] == chapter_number:
            chapter_outline = item.get("summary", "") or item.get("title", "")
            break
    if not chapter_outline:
        chapter_outline = f"第{chapter_number}章"

    # Mark the run durably before starting the graph without clearing a draft.
    existing_chapter = mgr.get_chapter(project_id, chapter_number)
    if existing_chapter is None:
        mgr.save_chapter(project_id, chapter_number, status=ChapterStatus.WRITING.value)
    elif existing_chapter.get("status") != ChapterStatus.APPROVED.value:
        mgr.save_chapter(
            project_id,
            chapter_number,
            outline=existing_chapter.get("outline", ""),
            draft_content=existing_chapter.get("draft_content", ""),
            status=ChapterStatus.WRITING.value,
            editor_report=existing_chapter.get("editor_report", "{}"),
            continuity_report=existing_chapter.get("continuity_report", "{}"),
            version=existing_chapter.get("version", 0),
            evolution_summary=existing_chapter.get("evolution_summary", "{}"),
        )
    mgr.update_outline_item(project_id, chapter_number, status=OutlineStatus.WRITING.value)

    # Build context
    ctx = ContextCompiler(mgr).compile(project_id, chapter_number).to_state()

    # Build initial state
    persist_dir = str(_get_persist_dir())
    graph = await build_chapter_graph_async(persist_dir=persist_dir)
    queue: asyncio.Queue = asyncio.Queue()
    # 清理同章节旧会话，避免 approve/reject 命中失效 session
    stale = session_store.find_session(project_id, chapter_number)
    if stale:
        session_store.remove(stale)
    session_id = session_store.create(graph, queue)

    # Use deterministic thread_id so checkpoints survive server restarts
    thread_id = f"{project_id}:ch{chapter_number}"
    config = {"configurable": {"thread_id": thread_id, "stream_queue": queue}}

    # An interrupted checkpoint is the durable source of truth across restarts.
    existing = await graph.aget_state(config)
    if existing and existing.next:
        runs = mgr.list_writing_runs(project_id, chapter_number)
        run = next(
            (
                item
                for item in runs
                if item["status"]
                in {"queued", "running", "waiting_review", "waiting_user", "retrying"}
            ),
            None,
        )
        if run is None:
            try:
                run = mgr.create_writing_run(project_id, chapter_number)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        session_store.set_config(session_id, config)
        session_store.set_context(session_id, project_id, chapter_number, run["id"])
        return StreamingResponse(
            replay_review(existing.values or {}, chapter_number),
            media_type="text/event-stream",
        )

    try:
        run = mgr.create_writing_run(project_id, chapter_number)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    initial_state = {
        "project_id": project_id,
        "writing_run_id": run["id"],
        "chapter_number": chapter_number,
        "chapter_outline": chapter_outline,
        "story_length": project.get("story_length", "long"),
        "target_chapter_words": project.get("target_chapter_words", 3000),
        "narrative_mode": project.get("narrative_mode"),
        "narrative_perspective": project.get("narrative_perspective", ""),
        "character_context": ctx.get("character_context", ""),
        "world_context": ctx.get("world_context", ""),
        "recent_summary": ctx.get("recent_summary", ""),
        "unresolved_foreshadowings": ctx.get("unresolved_foreshadowings", []),
        "context_packet_hash": ctx.get("context_packet_hash", ""),
        "context_packet": ctx.get("context_packet", {}),
        "scene_first": True,
        "existing_world_entities": mgr.get_all_world_entities(project_id),
        "persist_dir": persist_dir,
        "retry_count": 0,
    }

    return StreamingResponse(
        create_sse_stream(
            session_store,
            session_id,
            graph,
            initial_state,
            config,
            mgr,
            project_id,
            chapter_number,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/projects/{project_id}/chapters/{chapter_number}/approve")
async def approve_chapter(project_id: str, chapter_number: int):
    session_id = session_store.find_session(project_id, chapter_number)
    if not session_id:
        session_id = await _restore_session(project_id, chapter_number)
    if not session_id:
        raise HTTPException(status_code=404, detail="No active writing session")

    mgr = _get_manager()

    return StreamingResponse(
        resume_graph(
            session_store,
            session_id,
            feedback={"action": "approve", "comments": ""},
            mgr=mgr,
            project_id=project_id,
            chapter_number=chapter_number,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/projects/{project_id}/chapters/{chapter_number}/reject")
async def reject_chapter(project_id: str, chapter_number: int, req: RejectRequest):
    session_id = session_store.find_session(project_id, chapter_number)
    if not session_id:
        session_id = await _restore_session(project_id, chapter_number)
    if not session_id:
        raise HTTPException(status_code=404, detail="No active writing session")

    mgr = _get_manager()

    return StreamingResponse(
        resume_graph(
            session_store,
            session_id,
            feedback={"action": "reject", "comments": req.comments},
            mgr=mgr,
            project_id=project_id,
            chapter_number=chapter_number,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.put("/projects/{project_id}/chapters/{chapter_number}/draft")
async def save_draft(project_id: str, chapter_number: int, req: SaveDraftRequest):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    existing = mgr.get_chapter(project_id, chapter_number)
    status = (
        existing["status"]
        if existing and existing.get("status") == ChapterStatus.APPROVED.value
        else ChapterStatus.DRAFT.value
    )
    mgr.save_chapter(
        project_id=project_id,
        chapter_number=chapter_number,
        draft_content=req.content,
        status=status,
    )
    return {"ok": True}


def build_export_content(
    project: dict,
    chapters: list[dict],
    outlines: list[dict],
    fmt: str,
) -> str:
    """Build formatted export content."""
    title = project.get("title") or project.get("name", "Untitled")
    genre = project.get("genre", "")
    length_label = "长篇"
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Build chapter title map from outlines
    title_map: dict[int, str] = {}
    for o in outlines:
        title_map[o["chapter_number"]] = o.get("title", "")

    # Filter to chapters that have content (guard against None)
    written = [c for c in chapters if (c.get("draft_content") or "").strip()]
    total = len(written)

    if fmt == "txt":
        lines = [f"《{title}》", ""]
        meta_parts = []
        if genre:
            meta_parts.append(f"题材：{genre}")
        if length_label:
            meta_parts.append(f"篇幅：{length_label}")
        meta_parts.append(f"共 {total} 章")
        meta_parts.append(f"导出时间：{now}")
        lines.append(" | ".join(meta_parts))
        lines.append("")
        lines.append("─" * 40)
        lines.append("")

        for ch in written:
            cn = ch["chapter_number"]
            ch_title = title_map.get(cn, f"第{cn}章")
            lines.append(f"第{cn}章 {ch_title}")
            lines.append("")
            lines.append((ch.get("draft_content") or "").strip())
            lines.append("")
            lines.append("─" * 40)
            lines.append("")

        return "\n".join(lines)

    # Markdown (default)
    lines = [f"# 《{title}》", ""]
    meta_parts = []
    if genre:
        meta_parts.append(f"题材：{genre}")
    if length_label:
        meta_parts.append(f"篇幅：{length_label}")
    meta_parts.append(f"共 {total} 章")
    meta_parts.append(f"导出时间：{now}")
    lines.append("> " + " | ".join(meta_parts))
    lines.append("")
    lines.append("---")
    lines.append("")

    for ch in written:
        cn = ch["chapter_number"]
        ch_title = title_map.get(cn, "")
        heading = f"## 第{cn}章" + (f" {ch_title}" if ch_title else "")
        lines.append(heading)
        lines.append("")
        lines.append((ch.get("draft_content") or "").strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames."""
    return re.sub(r'[\\/:*?"<>|]', "", name)


@router.get("/projects/{project_id}/export")
async def export_novel(
    project_id: str,
    fmt: str = Query("md", alias="format"),
    preview: bool = Query(False),
):
    mgr = _get_manager()
    project = mgr.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    chapters = mgr.get_all_chapters(project_id)
    outlines = mgr.get_outline(project_id)
    title = project.get("title") or project.get("name", "Untitled")

    if fmt not in ("md", "txt"):
        fmt = "md"

    content = build_export_content(project, chapters, outlines, fmt)

    if preview:
        total = len([c for c in chapters if (c.get("draft_content") or "").strip()])
        return {
            "title": title,
            "content": content,
            "format": fmt,
            "chapter_count": total,
        }

    media_type = "text/markdown; charset=utf-8" if fmt == "md" else "text/plain; charset=utf-8"
    ext = "md" if fmt == "md" else "txt"
    safe_name = _sanitize_filename(title) or "novel"
    encoded_name = quote(f"{safe_name}.{ext}")

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}",
        },
    )
