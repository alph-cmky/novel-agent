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
        ("current_version_id", "TEXT"),
        ("approved_version_id", "TEXT"),
    ]
    for col_name, col_def in chapter_migrations:
        if col_name not in existing_chapters:
            conn.execute(f"ALTER TABLE chapters ADD COLUMN {col_name} {col_def}")

    existing_versions = {row["name"] for row in conn.execute("PRAGMA table_info(chapter_versions)")}
    if "scene_manifest" not in existing_versions:
        conn.execute(
            "ALTER TABLE chapter_versions ADD COLUMN scene_manifest TEXT NOT NULL DEFAULT '[]'"
        )

    existing_runs = {row["name"] for row in conn.execute("PRAGMA table_info(writing_runs)")}
    if "input_snapshot_id" not in existing_runs:
        conn.execute("ALTER TABLE writing_runs ADD COLUMN input_snapshot_id TEXT")

    existing_outbox = {row["name"] for row in conn.execute("PRAGMA table_info(outbox_events)")}
    for col_name, col_def in (
        ("lease_owner", "TEXT"),
        ("lease_expires_at", "TEXT"),
    ):
        if col_name not in existing_outbox:
            conn.execute(f"ALTER TABLE outbox_events ADD COLUMN {col_name} {col_def}")

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

CREATE TABLE IF NOT EXISTS writing_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    run_type TEXT NOT NULL DEFAULT 'draft',
    workflow_version TEXT NOT NULL DEFAULT 'v2',
    status TEXT NOT NULL DEFAULT 'queued',
    current_node TEXT NOT NULL DEFAULT '',
    current_version_id TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    input_snapshot_id TEXT REFERENCES canon_snapshots(id),
    lease_owner TEXT,
    lease_expires_at TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS canon_snapshots (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    payload TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chapter_versions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    run_id TEXT REFERENCES writing_runs(id) ON DELETE SET NULL,
    version_number INTEGER NOT NULL,
    parent_version_id TEXT REFERENCES chapter_versions(id) ON DELETE SET NULL,
    content TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    word_count INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'initial_generation',
    status TEXT NOT NULL DEFAULT 'candidate',
    scene_manifest TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, chapter_number, version_number)
);

CREATE TABLE IF NOT EXISTS canon_proposals (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    run_id TEXT REFERENCES writing_runs(id) ON DELETE SET NULL,
    version_id TEXT REFERENCES chapter_versions(id) ON DELETE SET NULL,
    proposal_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'proposed',
    reviewer_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    reviewed_at TEXT,
    committed_at TEXT
);

CREATE TABLE IF NOT EXISTS outbox_events (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    aggregate_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    lease_owner TEXT,
    lease_expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS story_events (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chapter_number INTEGER NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'chapter_event',
    subject TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    object_value TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    story_time TEXT NOT NULL DEFAULT '',
    causality TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '',
    source_version_id TEXT REFERENCES chapter_versions(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, chapter_number, event_type, subject, action, object_value)
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
CREATE INDEX IF NOT EXISTS idx_writing_runs_project ON writing_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_writing_runs_chapter ON writing_runs(project_id, chapter_number);
CREATE INDEX IF NOT EXISTS idx_canon_snapshots_project
    ON canon_snapshots(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chapter_versions_project ON chapter_versions(project_id);
CREATE INDEX IF NOT EXISTS idx_chapter_versions_chapter
    ON chapter_versions(project_id, chapter_number);
CREATE INDEX IF NOT EXISTS idx_canon_proposals_project
    ON canon_proposals(project_id);
CREATE INDEX IF NOT EXISTS idx_canon_proposals_run
    ON canon_proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_outbox_events_status
    ON outbox_events(status, created_at);
CREATE INDEX IF NOT EXISTS idx_story_events_project_chapter
    ON story_events(project_id, chapter_number);
CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_events_idempotency
    ON outbox_events(event_type, aggregate_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_writing_run
    ON writing_runs(project_id, chapter_number)
    WHERE status IN ('queued', 'running', 'waiting_review', 'waiting_user', 'retrying');
CREATE INDEX IF NOT EXISTS idx_world_entities_project ON world_entities(project_id);
CREATE INDEX IF NOT EXISTS idx_world_relations_project ON world_relations(project_id);
CREATE INDEX IF NOT EXISTS idx_foreshadowings_project ON foreshadowings(project_id);
CREATE INDEX IF NOT EXISTS idx_outlines_project ON outlines(project_id);
"""
