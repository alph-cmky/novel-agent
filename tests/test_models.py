"""Tests for SQLite init_db — PRAGMA + DDL.

锁住修复：``PRAGMA busy_timeout=5000`` 防多连接写锁竞争
（WAL 下偶发 "database is locked"）。
"""

from novel_agent.storage.models import get_db_path, init_db


def test_busy_timeout_pragma_set(tmp_path):
    conn = init_db(tmp_path / "novel.db")
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_foreign_keys_enabled(tmp_path):
    conn = init_db(tmp_path / "novel.db")
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_schema_creates_all_tables(tmp_path):
    conn = init_db(tmp_path / "novel.db")
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()

    assert {
        "projects",
        "chapters",
        "world_entities",
        "world_relations",
        "foreshadowings",
        "outlines",
    } <= tables


def test_init_db_idempotent(tmp_path):
    """第二次 init 不报错（IF NOT EXISTS + _migrate 幂等）。"""
    db = tmp_path / "novel.db"
    conn1 = init_db(db)
    conn1.close()
    conn2 = init_db(db)
    conn2.close()


def test_get_db_path(tmp_path):
    assert get_db_path(tmp_path) == tmp_path / "novel.db"
