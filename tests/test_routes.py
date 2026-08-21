"""Tests for durable writing-session recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from novel_agent.api.routes import _restore_session, session_store


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
