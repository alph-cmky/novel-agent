"""Tests for SSE SessionStore — 会话生命周期与清理。

锁住修复：``create_sse_stream`` 在 done / error 路径都调用 ``store.remove(session_id)``。
SessionStore 必须提供幂等的 remove 能力，供会话完成 / 异常后清理，防内存泄漏。
"""

import asyncio

from novel_agent.api.sse import SessionStore


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
