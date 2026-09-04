"""Tests for SSE SessionStore — 会话生命周期与清理。

锁住修复：``create_sse_stream`` 在 done / error 路径都调用 ``store.remove(session_id)``。
SessionStore 必须提供幂等的 remove 能力，供会话完成 / 异常后清理，防内存泄漏。
"""

import asyncio
from types import SimpleNamespace
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
        sid = store.create(graph=object(), queue=queue)

        assert len(sid) == 8
        session = store.get(sid)
        assert session["queue"] is queue
        assert session["config"] is None
        assert session["project_id"] is None

    def test_remove_is_idempotent(self):
        store = SessionStore()
        sid = store.create(object(), asyncio.Queue())

        store.remove(sid)
        assert store.get(sid) is None

        # 幂等：重复 remove 不抛异常（error 路径可能 remove 不存在的会话）
        store.remove(sid)

    def test_get_queue(self):
        store = SessionStore()
        queue = asyncio.Queue()
        sid = store.create(object(), queue)

        assert store.get_queue(sid) is queue
        assert store.get_queue("missing") is None

    def test_set_config_and_context(self):
        store = SessionStore()
        sid = store.create(object(), asyncio.Queue())

        store.set_config(sid, {"thread_id": "t1"})
        store.set_context(sid, "proj-1", 3)

        session = store.get(sid)
        assert session["config"] == {"thread_id": "t1"}
        assert session["project_id"] == "proj-1"
        assert session["chapter_number"] == 3

    def test_find_session_by_context(self):
        store = SessionStore()
        sid = store.create(object(), asyncio.Queue())
        store.set_context(sid, "proj-1", 3)

        assert store.find_session("proj-1", 3) == sid
        assert store.find_session("proj-1", 4) is None

    def test_setters_ignore_unknown_session(self):
        store = SessionStore()
        # 对不存在的会话 set 不应抛异常
        store.set_config("missing", {"x": 1})
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


class _LifecycleGraph:
    def __init__(self, state, error=None):
        self.state = state
        self.error = error

    async def astream_events(self, _input, _config, version):
        if self.error:
            raise self.error
        yield {
            "event": "on_chain_start",
            "name": "evolution_writer",
            "data": {},
        }

    async def aget_state(self, _config):
        return self.state


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
    state = SimpleNamespace(
        next=("human_review",),
        values={"writing_run_id": run["id"], "draft_content": "候选"},
    )
    store = SessionStore()
    session_id = store.create(_LifecycleGraph(state), asyncio.Queue())

    events = _run_stream(
        create_sse_stream(
            store,
            session_id,
            store.get(session_id)["graph"],
            {"writing_run_id": run["id"]},
            {"configurable": {}},
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
    session_id = store.create(
        _LifecycleGraph(None, RuntimeError("generation failed")), asyncio.Queue()
    )

    events = _run_stream(
        create_sse_stream(
            store,
            session_id,
            store.get(session_id)["graph"],
            {"writing_run_id": run["id"]},
            {"configurable": {}},
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
    assert mgr.get_all_world_entities(project_id)[0]["name"] == "洛千秋"
    assert mgr.get_story_events(project_id, 1)[0]["action"] == "秦照夜托付火种"


def test_save_unapproved_v2_result_keeps_proposal_pending(tmp_path):
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
            "draft_content": "候选正文",
            "human_approved": False,
            "worldbuilding_report": {"new_entities": [{"entity_type": "item", "name": "候选物"}]},
        },
    )

    versions = mgr.list_chapter_versions(project_id, 1)
    proposals = mgr.list_canon_proposals(project_id, run_id=run["id"])
    assert versions[0]["status"] == "candidate"
    assert proposals[0]["status"] == "proposed"
    assert mgr.get_all_world_entities(project_id) == []
