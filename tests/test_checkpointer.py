"""Tests for AsyncSqliteSaver connection lifecycle (close on shutdown).

回归保障：aiosqlite 的 worker thread 是非守护线程，跑完不 close 会让短生命周期
进程在退出时挂死。aclose_checkpointers() 负责关闭全部缓存连接并清空缓存。
"""

import tempfile

from novel_agent.graph.chapter import (
    _async_checkpointer_cache,
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
