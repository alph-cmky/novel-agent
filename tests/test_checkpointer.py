"""Tests for SqliteSaver checkpointer lifecycle and resume behavior.

两部分保障：
1. AsyncSqliteSaver connection lifecycle — aiosqlite 的 worker thread 是非守护线程，
   跑完不 close 会让短生命周期进程在退出时挂死。aclose_checkpointers() 负责关闭
   全部缓存连接并清空缓存。
2. Sync SqliteSaver resume — 清空 ``_checkpointer_cache`` 模拟进程重启后，从同一
   persist_dir 重建的 SqliteSaver 仍能读回 checkpoint（CLAUDE.md: thread_id 用
   ``{project_id}:ch{chapter_number}`` 确定性格式，重启可恢复）。
"""

import tempfile
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from novel_agent.graph.chapter import (
    _async_checkpointer_cache,
    _checkpointer_cache,
    _get_checkpointer,
    _get_checkpointer_async,
    aclose_checkpointers,
)


async def test_aclose_checkpointers_closes_and_clears_cache():
    _async_checkpointer_cache.clear()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            saver = await _get_checkpointer_async(tmp)
            conn = saver.conn
            assert len(_async_checkpointer_cache) == 1

            await aclose_checkpointers()

            assert len(_async_checkpointer_cache) == 0
            # aiosqlite.Connection.close() 会关底层 sqlite3.Connection 并置 None，
            # 同时 stop() 通知 worker thread 退出。
            assert conn._connection is None
            assert conn._running is False
            # 直接验证非守护 worker thread 真的退出了（否则进程退出会挂死）。
            conn._thread.join(timeout=5)
            assert not conn._thread.is_alive()
    finally:
        _async_checkpointer_cache.clear()


async def test_aclose_checkpointers_is_idempotent_on_empty_cache():
    _async_checkpointer_cache.clear()
    # 空缓存调用不应抛异常（服务 shutdown 时可能从未建过 checkpointer）。
    await aclose_checkpointers()
    assert len(_async_checkpointer_cache) == 0


# ── Sync SqliteSaver resume ──────────────────────────────


class _CountState(TypedDict, total=False):
    """Minimal state for resume tests — a single counter channel."""

    count: int


def _increment(state: _CountState) -> dict:
    return {"count": state.get("count", 0) + 1}


def _build_count_graph(checkpointer) -> StateGraph:
    """One-node graph compiled with the given checkpointer."""
    graph = StateGraph(_CountState)
    graph.add_node("inc", _increment)
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    return graph.compile(checkpointer=checkpointer)


class TestSyncCheckpointerResume:
    """Verify thread_id resumes from SqliteSaver across simulated restarts.

    清空 ``_checkpointer_cache`` 模拟进程重启：同一 persist_dir 重建出新的
    SqliteSaver 连接，但底层 ``checkpoints.db`` 不变，checkpoint 仍可读回。
    """

    def setup_method(self):
        _checkpointer_cache.clear()

    def teardown_method(self):
        _checkpointer_cache.clear()

    def test_same_persist_dir_returns_cached_saver(self, tmp_path):
        """Cache hit: 同一 persist_dir 返回同一 SqliteSaver 实例。"""
        saver_a = _get_checkpointer(str(tmp_path))
        saver_b = _get_checkpointer(str(tmp_path))
        assert saver_a is saver_b
        assert len(_checkpointer_cache) == 1

    def test_checkpoint_resumes_after_cache_clear(self, tmp_path):
        """重启模拟：清空缓存后从同一 db 重建的 saver 仍能读回 checkpoint。"""
        # {project_id}:ch{chapter_number} 确定性格式，跨重启可定位
        config = {"configurable": {"thread_id": "demo_proj:ch1"}}

        # 第一次"进程"：build + invoke，checkpoint 落盘到 checkpoints.db
        saver1 = _get_checkpointer(str(tmp_path))
        graph1 = _build_count_graph(saver1)
        result = graph1.invoke({"count": 0}, config=config)
        assert result["count"] == 1

        # 模拟重启：清空模块缓存 → 下次调用会建新 SqliteSaver 读同一 SQLite 文件
        _checkpointer_cache.clear()
        assert len(_checkpointer_cache) == 0

        # 第二次"进程"：从同一 persist_dir 重建
        saver2 = _get_checkpointer(str(tmp_path))
        assert saver2 is not saver1  # 新连接对象，非同一实例
        graph2 = _build_count_graph(saver2)

        # Resume：checkpoint 状态存活 — graph2 读到 count=1 而非空白
        snapshot = graph2.get_state(config)
        assert snapshot.values["count"] == 1
        assert snapshot.next == ()  # 已抵达 END，无待执行节点
