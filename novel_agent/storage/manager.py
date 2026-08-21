"""Project manager — ties together SQLite + ChromaDB for a project."""

import uuid
from contextlib import contextmanager
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

    @contextmanager
    def _conn(self):
        """Connection context manager: commit on success, rollback on error, always close."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

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
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, title, genre, story_length, "
                "target_chapter_words, world_setting, outline, narrative_mode, "
                "narrative_perspective) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, name, title or name, genre, story_length,
                 target_chapter_words, world_setting, outline,
                 narrative_mode, narrative_perspective),
            )
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
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
        index: bool = True,
    ) -> str:
        """Save a chapter. Returns chapter_id."""
        chapter_id = str(uuid.uuid4())[:8]

        with self._conn() as conn:
            row = conn.execute(
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
                   updated_at = datetime('now')
                   RETURNING id""",
                (chapter_id, project_id, chapter_number, outline, draft_content, status,
                 editor_report, continuity_report, version, evolution_summary),
            ).fetchone()
            chapter_id = row["id"] if row else chapter_id

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

        # Index in ChromaDB
        if draft_content and index:
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
        with self._conn() as conn:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [project_id]
            conn.execute(
                f"UPDATE projects SET {set_clause}, "
                "updated_at = datetime('now') WHERE id = ?",
                values,
            )

    def get_chapter(self, project_id: str, chapter_number: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            ).fetchone()
        return dict(row) if row else None

    def mark_chapter_failed(self, project_id: str, chapter_number: int) -> None:
        """Persist a failed run without discarding an existing approved chapter."""
        existing = self.get_chapter(project_id, chapter_number)
        if existing and existing.get("status") == ChapterStatus.APPROVED.value:
            return
        if existing:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE chapters SET status = ?, updated_at = datetime('now') "
                    "WHERE project_id = ? AND chapter_number = ?",
                    (ChapterStatus.FAILED.value, project_id, chapter_number),
                )
                conn.execute(
                    "UPDATE outlines SET status = ? WHERE project_id = ? "
                    "AND chapter_number = ?",
                    (OutlineStatus.FAILED.value, project_id, chapter_number),
                )
            return
        self.save_chapter(
            project_id,
            chapter_number,
            status=ChapterStatus.FAILED.value,
        )

    def get_all_chapters(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters WHERE project_id = ? ORDER BY chapter_number",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chapter_worldbuilding(self, project_id: str) -> list[dict]:
        """Lightweight fetch of chapter_number + worldbuilding_report only.

        Used by the graph builder to extract conflicts without pulling
        draft_content (正文全文) into memory.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chapter_number, worldbuilding_report FROM chapters "
                "WHERE project_id = ? ORDER BY chapter_number",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chapter_reports(self, project_id: str) -> list[dict]:
        """Lightweight fetch of chapter_number + reports only (no draft_content).

        Used by the orchestrator to summarize past chapter performance without
        pulling 正文全文 into memory.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chapter_number, editor_report, continuity_report FROM chapters "
                "WHERE project_id = ? ORDER BY chapter_number",
                (project_id,),
            ).fetchall()
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
        with self._conn() as conn:
            chars = conn.execute(
                "SELECT * FROM world_entities WHERE project_id = ? AND entity_type = 'character'",
                (project_id,),
            ).fetchall()
            world_ents = conn.execute(
                "SELECT * FROM world_entities WHERE project_id = ? AND entity_type != 'character'",
                (project_id,),
            ).fetchall()

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
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM world_entities WHERE project_id = ? ORDER BY entity_type, name",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_world_entities(
        self,
        project_id: str,
        worldbuilding_report: dict,
        chapter_number: int | None = None,
    ) -> int:
        """Persist new and updated entities, preserving history and merging properties."""
        import json

        new_entities = worldbuilding_report.get("new_entities", []) or []
        updated_entities = worldbuilding_report.get("updated_entities", []) or []
        if not new_entities and not updated_entities:
            return 0

        with self._conn() as conn:
            saved = 0
            entities = list(new_entities) + list(updated_entities)
            for entity in entities:
                if isinstance(entity, str):
                    entity = {"name": entity}
                if not isinstance(entity, dict):
                    continue
                entity_type = entity.get("entity_type", "unknown")
                name = entity.get("name", "")
                if not name:
                    continue
                incoming = entity.get("properties", {})
                incoming = incoming if isinstance(incoming, dict) else {}
                chapter = entity.get("first_appearance_chapter") or chapter_number or 0

                if entity_type == "unknown":
                    existing = conn.execute(
                        "SELECT id, properties, first_appearance_chapter FROM world_entities "
                        "WHERE project_id = ? AND name = ? ORDER BY id LIMIT 1",
                        (project_id, name),
                    ).fetchone()
                else:
                    existing = conn.execute(
                        "SELECT id, properties, first_appearance_chapter FROM world_entities "
                        "WHERE project_id = ? AND entity_type = ? AND name = ?",
                        (project_id, entity_type, name),
                    ).fetchone()

                if existing:
                    try:
                        current = json.loads(existing["properties"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        current = {}
                    merged = current if isinstance(current, dict) else {}
                    merged.update(incoming)
                    first_chapter = existing["first_appearance_chapter"] or chapter
                    conn.execute(
                        "UPDATE world_entities "
                        "SET properties = ?, first_appearance_chapter = ? "
                        "WHERE id = ?",
                        (json.dumps(merged, ensure_ascii=False), first_chapter, existing["id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO world_entities "
                        "(id, project_id, entity_type, name, properties, first_appearance_chapter) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4())[:8], project_id, entity_type, name,
                         json.dumps(incoming, ensure_ascii=False), chapter),
                    )
                saved += 1

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

        with self._conn() as conn:
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
                    cur = conn.execute(
                        "INSERT OR IGNORE INTO world_relations "
                        "(id, project_id, source, target, relation_type, first_appearance_chapter) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4())[:8], project_id, source, target,
                         rel.get("relation", "related_to"), chapter_number),
                    )
                    saved += cur.rowcount

        return saved

    def get_all_world_relations(self, project_id: str) -> list[dict]:
        """Get all relationships (edges) for a project."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM world_relations WHERE project_id = ? "
                "ORDER BY first_appearance_chapter",
                (project_id,),
            ).fetchall()
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
        with self._conn() as conn:
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

    def get_outline(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM outlines WHERE project_id = ? ORDER BY sort_order",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_outline_item(self, project_id: str, chapter_number: int, **fields) -> None:
        allowed = {"title", "summary", "status", "sort_order"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        with self._conn() as conn:
            conn.execute(
                f"UPDATE outlines SET {set_clause} WHERE project_id = ? AND chapter_number = ?",
                [*values, project_id, chapter_number],
            )

    def delete_outline_item(self, project_id: str, chapter_number: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM outlines WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            )

    # ── Foreshadowings ──────────────────────────────────

    def add_foreshadowing(
        self,
        project_id: str,
        description: str,
        planted_chapter: int,
        expected_resolve_chapter: int | None = None,
        risk_level: str = "medium",
        action_needed: str = "maintain",
        reader_knows: bool = False,
        characters_aware: list[str] | None = None,
        characters_unaware: list[str] | None = None,
    ) -> str:
        """Record a new foreshadowing. Returns foreshadowing_id."""
        import json as _json
        import uuid as _uuid
        fid = str(_uuid.uuid4())[:8]
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO foreshadowings "
                "(id, project_id, description, planted_chapter, "
                 "expected_resolve_chapter, status, risk_level, action_needed, reader_knows, "
                 "characters_aware, characters_unaware) "
                 "VALUES (?, ?, ?, ?, ?, 'planted', ?, ?, ?, ?, ?)",
                 (fid, project_id, description, planted_chapter,
                  expected_resolve_chapter, risk_level, action_needed,
                 1 if reader_knows else 0,
                 _json.dumps(characters_aware or [], ensure_ascii=False),
                 _json.dumps(characters_unaware or [], ensure_ascii=False)),
            )
        return fid

    def update_foreshadowing_status(
        self,
        project_id: str,
        description: str,
        planted_chapter: int | None = None,
        **kwargs,
    ) -> bool:
        """Update foreshadowing lifecycle fields (risk_level, action_needed, etc.)."""
        allowed = {"risk_level", "action_needed", "status", "reader_knows",
                    "characters_aware", "characters_unaware",
                    "resolved_chapter", "expected_resolve_chapter"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return False
        import json as _json
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        # JSON-serialize list fields
        for i, (k, v) in enumerate(list(updates.items())):
            if k in ("characters_aware", "characters_unaware") and isinstance(v, list):
                values[i] = _json.dumps(v, ensure_ascii=False)
        values.extend([project_id, description])
        where = "project_id = ? AND description = ?"
        if planted_chapter is not None:
            values.append(planted_chapter)
            where += " AND planted_chapter = ?"
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE foreshadowings SET {set_clause} "
                f"WHERE {where}",
                values,
            )
            updated = cur.rowcount > 0
        return updated

    def get_foreshadowings(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM foreshadowings WHERE project_id = ? "
                "ORDER BY planted_chapter", (project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Project helpers ───────────────────────────────

    def get_chapter_count(self, project_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM chapters WHERE project_id = ? AND status = ?",
                (project_id, ChapterStatus.APPROVED.value),
            ).fetchone()
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
        with self._conn() as conn:
            conn.execute(
                """UPDATE chapters SET worldbuilding_report = ?
                   WHERE project_id = ? AND chapter_number = ?""",
                (json.dumps(report, ensure_ascii=False), project_id, chapter_number),
            )

    def _cleanup_checkpoints(self, thread_id: str, prefix: bool = False) -> None:
        """Delete LangGraph checkpoint rows for a thread_id (or prefix)."""
        checkpoints_db = self.project_dir / "checkpoints.db"
        if not checkpoints_db.exists():
            return
        import sqlite3 as _sqlite3
        pattern = f"{thread_id}%" if prefix else thread_id
        op = "LIKE" if prefix else "="
        conn = _sqlite3.connect(str(checkpoints_db))
        try:
            conn.execute(f"DELETE FROM checkpoints WHERE thread_id {op} ?", (pattern,))
            conn.execute(f"DELETE FROM writes WHERE thread_id {op} ?", (pattern,))
            conn.commit()
        finally:
            conn.close()

    def delete_chapter(self, project_id: str, chapter_number: int) -> None:
        """Delete a chapter and its outline entry."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM chapters WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            )
            conn.execute(
                "DELETE FROM outlines WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            )
        self._cleanup_checkpoints(f"{project_id}:ch{chapter_number}")

    def delete_project(self, project_id: str) -> None:
        """Delete a project and all its related data."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chapters WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM outlines WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM world_entities WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM world_relations WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM foreshadowings WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        # Also remove the project's vector memory collection (ChromaDB).
        self.chapter_store.delete_collection(project_id)
        self._cleanup_checkpoints(f"{project_id}:", prefix=True)
