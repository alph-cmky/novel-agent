"""SQLite data models for project and chapter storage."""

import sqlite3
from pathlib import Path


def get_db_path(project_dir: Path) -> Path:
    return project_dir / "novel.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection):
    """Add columns that may be missing from older databases."""
    existing_projects = {row["name"] for row in conn.execute("PRAGMA table_info(projects)")}
    project_migrations = [
        ("story_length", "TEXT NOT NULL DEFAULT 'long'"),
        ("target_chapter_words", "INTEGER NOT NULL DEFAULT 3000"),
        ("world_setting", "TEXT NOT NULL DEFAULT ''"),
        ("narrative_mode", "TEXT"),  # NULL = legacy project
        ("narrative_perspective", "TEXT NOT NULL DEFAULT ''"),
    ]
    for col_name, col_def in project_migrations:
        if col_name not in existing_projects:
            conn.execute(f"ALTER TABLE projects ADD COLUMN {col_name} {col_def}")

    existing_chapters = {row["name"] for row in conn.execute("PRAGMA table_info(chapters)")}
    chapter_migrations = [
        ("worldbuilding_report", "TEXT NOT NULL DEFAULT '{}'"),
        ("version", "INTEGER NOT NULL DEFAULT 0"),
        ("evolution_summary", "TEXT NOT NULL DEFAULT '{}'"),
    ]
    for col_name, col_def in chapter_migrations:
        if col_name not in existing_chapters:
            conn.execute(f"ALTER TABLE chapters ADD COLUMN {col_name} {col_def}")

    existing_fs = {row["name"] for row in conn.execute("PRAGMA table_info(foreshadowings)")}
    foreshadowing_migrations = [
        ("risk_level", "TEXT NOT NULL DEFAULT 'medium'"),
        ("action_needed", "TEXT NOT NULL DEFAULT 'maintain'"),
        ("reader_knows", "INTEGER NOT NULL DEFAULT 0"),
        ("characters_aware", "TEXT NOT NULL DEFAULT '[]'"),
        ("characters_unaware", "TEXT NOT NULL DEFAULT '[]'"),
    ]
    for col_name, col_def in foreshadowing_migrations:
        if col_name not in existing_fs:
            conn.execute(f"ALTER TABLE foreshadowings ADD COLUMN {col_name} {col_def}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    genre TEXT NOT NULL DEFAULT '',
    outline TEXT NOT NULL DEFAULT '',
    story_length TEXT NOT NULL DEFAULT 'long',
    target_chapter_words INTEGER NOT NULL DEFAULT 3000,
    world_setting TEXT NOT NULL DEFAULT '',
    narrative_mode TEXT,
    narrative_perspective TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    outline TEXT NOT NULL DEFAULT '',
    draft_content TEXT NOT NULL DEFAULT '',
    editor_report TEXT NOT NULL DEFAULT '{}',
    continuity_report TEXT NOT NULL DEFAULT '{}',
    worldbuilding_report TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',

    -- 进化元数据 (v2)
    version INTEGER NOT NULL DEFAULT 0,
    evolution_summary TEXT NOT NULL DEFAULT '{}',

    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, chapter_number)
);

CREATE TABLE IF NOT EXISTS world_entities (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties TEXT NOT NULL DEFAULT '{}',
    first_appearance_chapter INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS world_relations (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    first_appearance_chapter INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, source, target, relation_type)
);

CREATE TABLE IF NOT EXISTS foreshadowings (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    planted_chapter INTEGER NOT NULL,
    expected_resolve_chapter INTEGER,
    resolved_chapter INTEGER,
    status TEXT NOT NULL DEFAULT 'open',
    risk_level TEXT NOT NULL DEFAULT 'medium',
    action_needed TEXT NOT NULL DEFAULT 'maintain',
    reader_knows INTEGER NOT NULL DEFAULT 0,
    characters_aware TEXT NOT NULL DEFAULT '[]',
    characters_unaware TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS outlines (
    project_id TEXT NOT NULL,
    chapter_number INTEGER NOT NULL,
    title TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    sort_order INTEGER DEFAULT 0,
    PRIMARY KEY (project_id, chapter_number)
);

CREATE INDEX IF NOT EXISTS idx_chapters_project ON chapters(project_id);
CREATE INDEX IF NOT EXISTS idx_world_entities_project ON world_entities(project_id);
CREATE INDEX IF NOT EXISTS idx_world_relations_project ON world_relations(project_id);
CREATE INDEX IF NOT EXISTS idx_foreshadowings_project ON foreshadowings(project_id);
CREATE INDEX IF NOT EXISTS idx_outlines_project ON outlines(project_id);
"""
