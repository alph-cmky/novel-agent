"""Tests for SSE SessionStore — 会话生命周期与清理。

锁住修复：``create_sse_stream`` 在 done / error 路径都调用 ``store.remove(session_id)``。
SessionStore 必须提供幂等的 remove 能力，供会话完成 / 异常后清理，防内存泄漏。
"""

import asyncio
from unittest.mock import MagicMock, patch

from novel_agent.api.sse import (
    SessionStore,
    _save_chapter_result,
    create_sse_stream,
    replay_review,
)
from novel_agent.storage.manager import ProjectManager


class TestSessionStore:
    def test_create_and_get(self):
        store = SessionStore()
        queue = asyncio.Queue()
        sid = store.create(queue)

        assert len(sid) == 8
        session = store.get(sid)
        assert session["queue"] is queue
        assert session["project_id"] is None

    def test_remove_is_idempotent(self):
        store = SessionStore()
        sid = store.create(asyncio.Queue())

        store.remove(sid)
        assert store.get(sid) is None

        # 幂等：重复 remove 不抛异常（error 路径可能 remove 不存在的会话）
        store.remove(sid)

    def test_get_queue(self):
        store = SessionStore()
        queue = asyncio.Queue()
        sid = store.create(queue)

        assert store.get_queue(sid) is queue
        assert store.get_queue("missing") is None

    def test_set_context(self):
        store = SessionStore()
        sid = store.create(asyncio.Queue())

        store.set_context(sid, "proj-1", 3)

        session = store.get(sid)
        assert session["project_id"] == "proj-1"
        assert session["chapter_number"] == 3

    def test_find_session_by_context(self):
        store = SessionStore()
        sid = store.create(asyncio.Queue())
        store.set_context(sid, "proj-1", 3)

        assert store.find_session("proj-1", 3) == sid
        assert store.find_session("proj-1", 4) is None

    def test_setters_ignore_unknown_session(self):
        store = SessionStore()
        # 对不存在的会话 set 不应抛异常
        store.set_context("missing", "p", 1)


async def test_replay_review_emits_persisted_checkpoint():
    events = [
        event
        async for event in replay_review(
            {"draft_content": "恢复正文", "editor_report": {"overall_score": 80}}, 2
        )
    ]

    assert "event: review_required" in events[-1]
    assert "恢复正文" in events[-1]


def _run_stream(stream):
    async def collect():
        return [event async for event in stream]

    return asyncio.run(collect())


def test_create_sse_stream_persists_waiting_review_run(tmp_path):
    with patch("novel_agent.storage.manager.ChapterStore") as mock_store:
        mock_store.return_value = MagicMock()
        mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)

    # Patch run_agent_loop to return a minimal result
    fake_values = {
        "writing_run_id": run["id"],
        "draft_content": "候选",
        "editor_report": {"overall_score": 80},
        "continuity_report": {"overall_score": 90},
        "worldbuilding_report": {},
        "orchestrator_strategy": {},
        "quality_gate_report": {"passed": True},
    }

    store = SessionStore()
    session_id = store.create(asyncio.Queue())

    with patch("novel_agent.api.sse.run_chapter_agent_loop") as mock_run:
        from novel_agent.graph.runner import ChapterOutcome

        mock_run.return_value = ChapterOutcome(values=fake_values, interrupted=False)
        events = _run_stream(
            create_sse_stream(
                store,
                session_id,
                {"writing_run_id": run["id"]},
                mgr,
                project_id,
                1,
            )
        )

    assert any("review_required" in event for event in events)
    assert mgr.get_writing_run(run["id"])["status"] == "waiting_review"


def test_create_sse_stream_marks_run_failed(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)
    store = SessionStore()
    session_id = store.create(asyncio.Queue())

    with patch("novel_agent.api.sse.run_chapter_agent_loop") as mock_run:
        mock_run.side_effect = RuntimeError("generation failed")
        events = _run_stream(
            create_sse_stream(
                store,
                session_id,
                {"writing_run_id": run["id"]},
                mgr,
                project_id,
                1,
            )
        )

    assert any("error" in event for event in events)
    assert mgr.get_writing_run(run["id"])["status"] == "failed"


def test_save_chapter_result_creates_and_commits_v2_version(tmp_path):
    with patch("novel_agent.storage.manager.ChapterStore") as mock_store:
        mock_store.return_value = MagicMock()
        mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)

    _save_chapter_result(
        mgr,
        project_id,
        1,
        {
            "writing_run_id": run["id"],
            "draft_content": "正文",
            "human_approved": True,
            "editor_report": {"overall_score": 80},
            "continuity_report": {"overall_score": 90},
            "worldbuilding_report": {
                "new_entities": [{"entity_type": "character", "name": "洛千秋"}],
                "chapter_events": ["秦照夜托付火种"],
            },
        },
    )

    versions = mgr.list_chapter_versions(project_id, 1)
    assert len(versions) == 1
    assert versions[0]["status"] == "approved"
    assert mgr.get_writing_run(run["id"])["status"] == "succeeded"
    proposals = mgr.list_canon_proposals(project_id, run_id=run["id"])
    assert proposals[0]["status"] == "committed"
