"""Project manager — ties together SQLite + ChromaDB for a project."""

import uuid
from pathlib import Path

from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.enums import ChapterStatus, OutlineStatus
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
        world_setting: str = "",
        outline: str = "",
        narrative_mode: str | None = None,
        narrative_perspective: str = "",
    ) -> str:
        """Create a new project. Returns project_id."""
        project_id = str(uuid.uuid4())[:8]
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO projects (id, name, title, genre, story_length, "
            "target_chapter_words, world_setting, outline, narrative_mode, "
            "narrative_perspective) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, name, title or name, genre, story_length,
             target_chapter_words, world_setting, outline,
             narrative_mode, narrative_perspective),
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
        status: str = ChapterStatus.DRAFT.value,
        editor_report: str = "{}",
        continuity_report: str = "{}",
        version: int = 0,
        evolution_summary: str = "{}",
    ) -> str:
        """Save a chapter. Returns chapter_id."""
        conn = self._get_conn()
        chapter_id = str(uuid.uuid4())[:8]

        conn.execute(
            """INSERT INTO chapters
               (id, project_id, chapter_number, outline, draft_content, status,
                editor_report, continuity_report, version, evolution_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id, chapter_number) DO UPDATE SET
               outline = excluded.outline,
               draft_content = excluded.draft_content,
               status = excluded.status,
               editor_report = excluded.editor_report,
               continuity_report = excluded.continuity_report,
               version = excluded.version,
               evolution_summary = excluded.evolution_summary,
               updated_at = datetime('now')""",
            (chapter_id, project_id, chapter_number, outline, draft_content, status,
             editor_report, continuity_report, version, evolution_summary),
        )

        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?",
            (project_id,),
        )
        # Update outline status if exists — map chapter status ("draft"/"approved")
        # onto the outline lifecycle ("drafted"/"approved") shown in the UI.
        outline_status = (
            OutlineStatus.DRAFTED.value if status == ChapterStatus.DRAFT else status
        )
        conn.execute(
            """UPDATE outlines SET status = ?
               WHERE project_id = ? AND chapter_number = ?""",
            (outline_status, project_id, chapter_number),
        )
        conn.commit()
        conn.close()

        # Index in ChromaDB
        if draft_content:
            self.chapter_store.index_chapter(project_id, chapter_number, draft_content)

        return chapter_id

    def update_project(self, project_id: str, **fields) -> None:
        """Update project fields. Only whitelisted fields are accepted."""
        allowed = {"name", "title", "genre", "story_length",
                   "target_chapter_words", "world_setting", "outline",
                   "narrative_mode", "narrative_perspective"}
        updates = {k: v for k, v in fields.items()
                   if k in allowed and v is not None}
        if not updates:
            return
        conn = self._get_conn()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [project_id]
        conn.execute(
            f"UPDATE projects SET {set_clause}, "
            "updated_at = datetime('now') WHERE id = ?",
            values,
        )
        conn.commit()
        conn.close()

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

    def get_chapter_worldbuilding(self, project_id: str) -> list[dict]:
        """Lightweight fetch of chapter_number + worldbuilding_report only.

        Used by the graph builder to extract conflicts without pulling
        draft_content (正文全文) into memory.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT chapter_number, worldbuilding_report FROM chapters "
            "WHERE project_id = ? ORDER BY chapter_number",
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

    def save_world_relations(
        self,
        project_id: str,
        chapter_number: int,
        worldbuilding_report: dict,
    ) -> int:
        """Persist entity relationships (edges) to world_relations. Returns count."""
        new_entities = worldbuilding_report.get("new_entities", [])
        if not new_entities:
            return 0

        conn = self._get_conn()
        saved = 0
        for entity in new_entities:
            if not isinstance(entity, dict):
                continue
            source = entity.get("name", "")
            if not source:
                continue
            for rel in entity.get("relationships", []) or []:
                if not isinstance(rel, dict):
                    continue
                target = rel.get("target", "")
                if not target:
                    continue
                # INSERT OR IGNORE 依赖 UNIQUE(project_id, source, target, relation_type)
                # 去重并保留首次 first_appearance_chapter
                conn.execute(
                    "INSERT OR IGNORE INTO world_relations "
                    "(id, project_id, source, target, relation_type, first_appearance_chapter) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4())[:8], project_id, source, target,
                     rel.get("relation", "related_to"), chapter_number),
                )
                saved += 1

        conn.commit()
        conn.close()
        return saved

    def get_all_world_relations(self, project_id: str) -> list[dict]:
        """Get all relationships (edges) for a project."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM world_relations WHERE project_id = ? "
            "ORDER BY first_appearance_chapter",
            (project_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def backfill_world_relations(self, project_id: str) -> int:
        """Backfill edges from existing chapters' worldbuilding_report (idempotent)."""
        import json
        chapters = self.get_chapter_worldbuilding(project_id)
        total = 0
        for ch in chapters:
            try:
                wb = json.loads(ch.get("worldbuilding_report", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                wb = {}
            total += self.save_world_relations(
                project_id, ch["chapter_number"], wb
            )
        return total

    # ── Outline ───────────────────────────────────────

    def save_outline(self, project_id: str, chapters: list[dict]) -> None:
        """Batch upsert outline chapters."""
        conn = self._get_conn()
        for ch in chapters:
            conn.execute(
                """INSERT INTO outlines
                   (project_id, chapter_number, title, summary, status, sort_order)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_id, chapter_number) DO UPDATE SET
                   title = excluded.title,
                   summary = excluded.summary,
                   status = excluded.status,
                   sort_order = excluded.sort_order""",
                (
                    project_id,
                    ch.get("chapter_number", 0),
                    ch.get("title", ""),
                    ch.get("summary", ""),
                    ch.get("status", OutlineStatus.PENDING.value),
                    ch.get("sort_order", ch.get("chapter_number", 0)),
                ),
            )
        conn.commit()
        conn.close()

    def get_outline(self, project_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM outlines WHERE project_id = ? ORDER BY sort_order",
            (project_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_outline_item(self, project_id: str, chapter_number: int, **fields) -> None:
        allowed = {"title", "summary", "status", "sort_order"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        conn = self._get_conn()
        conn.execute(
            f"UPDATE outlines SET {set_clause} WHERE project_id = ? AND chapter_number = ?",
            [*values, project_id, chapter_number],
        )
        conn.commit()
        conn.close()

    def delete_outline_item(self, project_id: str, chapter_number: int) -> None:
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM outlines WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_number),
        )
        conn.commit()
        conn.close()

    # ── Foreshadowings ──────────────────────────────────

    def add_foreshadowing(
        self,
        project_id: str,
        description: str,
        planted_chapter: int,
        expected_resolve_chapter: int | None = None,
        risk_level: str = "medium",
        reader_knows: bool = False,
        characters_aware: list[str] | None = None,
        characters_unaware: list[str] | None = None,
    ) -> str:
        """Record a new foreshadowing. Returns foreshadowing_id."""
        import json as _json
        import uuid as _uuid
        fid = str(_uuid.uuid4())[:8]
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO foreshadowings "
            "(id, project_id, description, planted_chapter, "
            "expected_resolve_chapter, status, risk_level, reader_knows, "
            "characters_aware, characters_unaware) "
            "VALUES (?, ?, ?, ?, ?, 'planted', ?, ?, ?, ?)",
            (fid, project_id, description, planted_chapter,
             expected_resolve_chapter, risk_level,
             1 if reader_knows else 0,
             _json.dumps(characters_aware or [], ensure_ascii=False),
             _json.dumps(characters_unaware or [], ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
        return fid

    def update_foreshadowing_status(
        self,
        project_id: str,
        description: str,
        planted_chapter: int,
        **kwargs,
    ) -> bool:
        """Update foreshadowing lifecycle fields (risk_level, action_needed, etc.)."""
        allowed = {"risk_level", "action_needed", "status", "reader_knows",
                    "characters_aware", "characters_unaware",
                    "resolved_chapter", "expected_resolve_chapter"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        import json as _json
        conn = self._get_conn()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        # JSON-serialize list fields
        for i, (k, v) in enumerate(list(updates.items())):
            if k in ("characters_aware", "characters_unaware") and isinstance(v, list):
                values[i] = _json.dumps(v, ensure_ascii=False)
        values.extend([project_id, description, planted_chapter])
        conn.execute(
            f"UPDATE foreshadowings SET {set_clause} "
            f"WHERE project_id = ? AND description = ? AND planted_chapter = ?",
            values,
        )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return updated

    def get_foreshadowings(self, project_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM foreshadowings WHERE project_id = ? "
            "ORDER BY planted_chapter", (project_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Project helpers ───────────────────────────────

    def get_chapter_count(self, project_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chapters WHERE project_id = ? AND status != ?",
            (project_id, ChapterStatus.DRAFT.value),
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0

    def get_project_with_progress(self, project_id: str) -> dict | None:
        project = self.get_project(project_id)
        if not project:
            return None
        project["chapter_count"] = self.get_chapter_count(project_id)
        outline = self.get_outline(project_id)
        project["total_chapters"] = len(outline)
        return project

    def list_projects_with_progress(self) -> list[dict]:
        projects = self.list_projects()
        for p in projects:
            p["chapter_count"] = self.get_chapter_count(p["id"])
            outline = self.get_outline(p["id"])
            p["total_chapters"] = len(outline)
        return projects

    def update_chapter_worldbuilding(
        self, project_id: str, chapter_number: int, report: dict
    ) -> None:
        import json
        conn = self._get_conn()
        conn.execute(
            """UPDATE chapters SET worldbuilding_report = ?
               WHERE project_id = ? AND chapter_number = ?""",
            (json.dumps(report, ensure_ascii=False), project_id, chapter_number),
        )
        conn.commit()
        conn.close()

    def delete_chapter(self, project_id: str, chapter_number: int) -> None:
        """Delete a chapter and its outline entry."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM chapters WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_number),
        )
        conn.execute(
            "DELETE FROM outlines WHERE project_id = ? AND chapter_number = ?",
            (project_id, chapter_number),
        )
        conn.commit()
        conn.close()

    def delete_project(self, project_id: str) -> None:
        """Delete a project and all its related data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chapters WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM outlines WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM world_entities WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM world_relations WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM foreshadowings WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        # Also remove the project's vector memory collection (ChromaDB).
        self.chapter_store.delete_collection(project_id)
