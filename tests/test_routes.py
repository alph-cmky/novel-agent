"""Tests for durable writing-session recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from novel_agent.api.routes import (
    CreateRunRequest,
    CreateVersionRequest,
    SceneRewriteRequest,
    _restore_session,
    commit_writing_run,
    create_chapter_version,
    create_writing_run,
    diff_chapter_version,
    get_writing_run,
    list_chapter_scenes,
    review_scene_version,
    rewrite_scene,
    session_store,
)
from novel_agent.storage.manager import ProjectManager


def test_restore_session_rebuilds_handle_from_pending_checkpoint(tmp_path):
    graph = SimpleNamespace(
        aget_state=AsyncMock(
            return_value=SimpleNamespace(next=("human_review",), values={"draft_content": "正文"})
        )
    )
    session_store._sessions.clear()

    with patch("novel_agent.api.routes._get_persist_dir", return_value=tmp_path), \
         patch("novel_agent.api.routes.build_chapter_graph_async", AsyncMock(return_value=graph)):
        session_id = asyncio.run(_restore_session("project", 3))

    session = session_store.get(session_id)
    assert session["config"] == {"configurable": {"thread_id": "project:ch3"}}
    assert session["project_id"] == "project"
    assert session["chapter_number"] == 3
    session_store.remove(session_id)


def test_v2_run_routes_create_version_commit(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    with patch("novel_agent.api.routes._get_manager", return_value=mgr):
        run = asyncio.run(
            create_writing_run(project_id, 1, CreateRunRequest())
        )
        version = asyncio.run(
            create_chapter_version(
                run["id"], CreateVersionRequest(content="正文")
            )
        )
        assert version["version_number"] == 1
        assert asyncio.run(get_writing_run(run["id"]))["status"] == "waiting_review"
        committed = asyncio.run(commit_writing_run(run["id"]))

    assert committed["status"] == "approved"
    assert mgr.get_chapter(project_id, 1)["draft_content"] == "正文"


def test_v2_run_route_rejects_concurrent_run(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    with patch("novel_agent.api.routes._get_manager", return_value=mgr):
        asyncio.run(create_writing_run(project_id, 1, CreateRunRequest()))
        with pytest.raises(Exception) as exc_info:
            asyncio.run(create_writing_run(project_id, 1, CreateRunRequest()))

    assert exc_info.value.status_code == 409


def test_scene_rewrite_and_diff_routes(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)
    version = mgr.create_chapter_version(
        project_id,
        1,
        "一\n\n二",
        run_id=run["id"],
        scene_plan=[{"scene_index": 1}, {"scene_index": 2}],
        scene_drafts=["一", "二"],
    )
    mgr.update_writing_run(run["id"], current_version_id=version["id"])

    with patch("novel_agent.api.routes._get_manager", return_value=mgr):
        rewritten = asyncio.run(
            rewrite_scene(run["id"], 2, SceneRewriteRequest(content="二改"))
        )
        scenes_before_commit = asyncio.run(list_chapter_scenes(project_id, 1))
        diff = asyncio.run(diff_chapter_version(rewritten["id"]))
        asyncio.run(commit_writing_run(run["id"]))
        scenes_after_commit = asyncio.run(list_chapter_scenes(project_id, 1))

    assert rewritten["content"] == "一\n\n二改"
    assert scenes_before_commit == []  # candidate is not formal until commit
    assert scenes_after_commit[1]["content"] == "二改"
    assert any("二改" in line for line in diff["diff"])


def test_scene_review_runs_editor_and_continuity_on_scene_only(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p", narrative_mode="linear")
    version = mgr.create_chapter_version(
        project_id,
        1,
        "一",
        scene_plan=[{"scene_index": 1, "outline": "冲突"}],
        scene_drafts=["Scene 正文"],
    )

    class FakeEditor:
        def __init__(self, config):
            self.config = config

        async def review(self, **kwargs):
            assert kwargs["draft_content"] == "Scene 正文"
            return {"overall_score": 80}, None

    class FakeContinuity:
        def __init__(self, config, chapter_store, project_id):
            self.project_id = project_id

        async def audit(self, **kwargs):
            assert kwargs["draft_content"] == "Scene 正文"
            return {"overall_score": 90}, None

    with patch("novel_agent.api.routes._get_manager", return_value=mgr), \
         patch("novel_agent.api.routes.EditorAgent", FakeEditor), \
         patch("novel_agent.api.routes.ContinuityAgent", FakeContinuity), \
         patch("novel_agent.api.routes._config_for", return_value=object()), \
         patch("novel_agent.api.routes._get_chapter_store", return_value=object()):
        result = asyncio.run(review_scene_version(version["id"], 1))

    assert result["valid"] is True
    assert result["editor_report"]["overall_score"] == 80
    assert result["continuity_report"]["overall_score"] == 90
