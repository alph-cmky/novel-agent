"""Project manager — ties together SQLite + ChromaDB for a project."""

import uuid
from pathlib import Path

from novel_agent.memory.embeddings import ChapterStore
from novel_agent.storage.models import get_db_path, init_db


class ProjectManager:
    """Manages the lifecycle of a novel project."""

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.db_path = get_db_path(self.project_dir)
        self.chapter_store = ChapterStore(self.project_dir / "chroma_data")

    # ── Database ──────────────────────────────────────

    def _get_conn(self):
        return init_db(self.db_path)

    def init_project(
        self,
        name: str,
        title: str = "",
        genre: str = "",
        story_length: str = "long",
        target_chapter_words: int = 3000,
    ) -> str:
        """Create a new project. Returns project_id."""
        project_id = str(uuid.uuid4())[:8]
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO projects (id, name, title, genre, story_length, target_chapter_words) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, title or name, genre, story_length, target_chapter_words),
        )
        conn.commit()
        conn.close()
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_or_create_default_project(self) -> str:
        """Get the first project or create a default one."""
        projects = self.list_projects()
        if projects:
            return projects[0]["id"]
        return self.init_project(name="default", title="未命名小说")

    # ── Chapters ──────────────────────────────────────

    def save_chapter(
        self,
        project_id: str,
        chapter_number: int,
        outline: str = "",
        draft_content: str = "",
        status: str = "draft",
    ) -> str:
        """Save a chapter. Returns chapter_id."""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM chapters WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_number),
        ).fetchone()

        if existing:
            chapter_id = existing["id"]
            conn.execute(
                """UPDATE chapters
                   SET outline = ?, draft_content = ?, status = ?,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (outline, draft_content, status, chapter_id),
            )
        else:
            chapter_id = str(uuid.uuid4())[:8]
            conn.execute(
                """INSERT INTO chapters
                   (id, project_id, chapter_number, outline, draft_content, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chapter_id, project_id, chapter_number, outline, draft_content, status),
            )

        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?",
            (project_id,),
        )
        conn.commit()
        conn.close()

        # Index in ChromaDB
        if draft_content:
            self.chapter_store.index_chapter(project_id, chapter_number, draft_content)

        return chapter_id

    def get_chapter(self, project_id: str, chapter_number: int) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM chapters WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_number),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_chapters(self, project_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chapters WHERE project_id = ? ORDER BY chapter_number",
            (project_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def build_context(
        self,
        project_id: str,
        chapter_number: int,
        max_recent_chapters: int = 3,
    ) -> dict[str, str]:
        """Build context for writing a chapter: recent summary + character/world info."""
        chapters = self.get_all_chapters(project_id)
        recent = [c for c in chapters if c["chapter_number"] < chapter_number]

        # Recent summary from last N chapters
        recent_summary_parts = []
        for c in recent[-max_recent_chapters:]:
            draft = c.get("draft_content", "")
            if draft:
                recent_summary_parts.append(
                    f"第{c['chapter_number']}章: {draft[:300]}..."
                    if len(draft) > 300 else f"第{c['chapter_number']}章: {draft}"
                )
        recent_summary = "\n\n".join(recent_summary_parts) if recent_summary_parts else ""

        # Character context from world_entities
        conn = self._get_conn()
        chars = conn.execute(
            "SELECT * FROM world_entities WHERE project_id = ? AND entity_type = 'character'",
            (project_id,),
        ).fetchall()
        world_ents = conn.execute(
            "SELECT * FROM world_entities WHERE project_id = ? AND entity_type != 'character'",
            (project_id,),
        ).fetchall()
        conn.close()

        character_context = "\n".join(
            f"- {c['name']}: {c['properties']}" for c in chars
        ) if chars else ""

        world_context = "\n".join(
            f"- [{e['entity_type']}] {e['name']}: {e['properties']}"
            for e in world_ents
        ) if world_ents else ""

        return {
            "recent_summary": recent_summary,
            "character_context": character_context,
            "world_context": world_context,
        }

    def get_all_world_entities(self, project_id: str) -> list[dict]:
        """Get all world entities for a project (for state population)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM world_entities WHERE project_id = ? ORDER BY entity_type, name",
            (project_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def save_world_entities(
        self,
        project_id: str,
        worldbuilding_report: dict,
    ) -> int:
        """Persist extracted world entities to SQLite. Returns count of saved entities."""
        import json

        new_entities = worldbuilding_report.get("new_entities", [])
        if not new_entities:
            return 0

        conn = self._get_conn()
        saved = 0
        for entity in new_entities:
            entity_type = entity.get("entity_type", "unknown")
            name = entity.get("name", "")
            if not name:
                continue
            props = json.dumps(
                entity.get("properties", {}), ensure_ascii=False
            )
            chapter = entity.get("first_appearance_chapter", 0)

            existing = conn.execute(
                "SELECT id FROM world_entities "
                "WHERE project_id = ? AND entity_type = ? AND name = ?",
                (project_id, entity_type, name),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE world_entities "
                    "SET properties = ?, first_appearance_chapter = ? "
                    "WHERE id = ?",
                    (props, chapter, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO world_entities "
                    "(id, project_id, entity_type, name, properties, first_appearance_chapter) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4())[:8], project_id, entity_type, name, props, chapter),
                )
            saved += 1

        conn.commit()
        conn.close()
        return saved
