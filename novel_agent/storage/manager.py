"""Project manager — ties together SQLite + ChromaDB for a project."""

import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from novel_agent.memory.embeddings import ChapterStore
from novel_agent.schema.enums import (
    ChapterStatus,
    OutboxStatus,
    OutlineStatus,
    ProposalStatus,
    RunStatus,
)
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
                (
                    project_id,
                    name,
                    title or name,
                    genre,
                    story_length,
                    target_chapter_words,
                    world_setting,
                    outline,
                    narrative_mode,
                    narrative_perspective,
                ),
            )
        return project_id

    def get_project(self, project_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
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
                (
                    chapter_id,
                    project_id,
                    chapter_number,
                    outline,
                    draft_content,
                    status,
                    editor_report,
                    continuity_report,
                    version,
                    evolution_summary,
                ),
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
        allowed = {
            "name",
            "title",
            "genre",
            "story_length",
            "target_chapter_words",
            "world_setting",
            "outline",
            "narrative_mode",
            "narrative_perspective",
        }
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        with self._conn() as conn:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [project_id]
            conn.execute(
                f"UPDATE projects SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
                values,
            )

    def get_chapter(self, project_id: str, chapter_number: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapters WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            ).fetchone()
        return dict(row) if row else None

    # ── Durable V2 runs and immutable chapter versions ──

    def _create_canon_snapshot(self, conn, project_id: str) -> str:
        snapshot_id = str(uuid.uuid4())
        chapters = conn.execute(
            "SELECT chapter_number, approved_version_id, draft_content "
            "FROM chapters WHERE project_id = ? AND status = 'approved' "
            "ORDER BY chapter_number",
            (project_id,),
        ).fetchall()
        payload = {
            "entities": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM world_entities WHERE project_id = ? ORDER BY id",
                    (project_id,),
                ).fetchall()
            ],
            "relations": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM world_relations WHERE project_id = ? ORDER BY id",
                    (project_id,),
                ).fetchall()
            ],
            "foreshadowings": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM foreshadowings WHERE project_id = ? ORDER BY id",
                    (project_id,),
                ).fetchall()
            ],
            "story_events": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM story_events WHERE project_id = ? ORDER BY chapter_number, id",
                    (project_id,),
                ).fetchall()
            ],
            "chapters": [dict(row) for row in chapters],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        conn.execute(
            "INSERT INTO canon_snapshots (id, project_id, payload, content_hash) "
            "VALUES (?, ?, ?, ?)",
            (snapshot_id, project_id, encoded, hashlib.sha256(encoded.encode()).hexdigest()),
        )
        return snapshot_id

    def get_canon_snapshot(self, snapshot_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM canon_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        if not row:
            return None
        snapshot = dict(row)
        snapshot["payload"] = json.loads(snapshot["payload"])
        return snapshot

    def create_writing_run(
        self,
        project_id: str,
        chapter_number: int,
        run_type: str = "draft",
        workflow_version: str = "v2",
    ) -> dict:
        """Create one durable run and reject concurrent runs for a chapter."""
        run_id = str(uuid.uuid4())
        try:
            with self._conn() as conn:
                snapshot_id = self._create_canon_snapshot(conn, project_id)
                conn.execute(
                    "INSERT INTO writing_runs "
                    "(id, project_id, chapter_number, run_type, workflow_version, status, "
                    "input_snapshot_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        project_id,
                        chapter_number,
                        run_type,
                        workflow_version,
                        RunStatus.QUEUED.value,
                        snapshot_id,
                    ),
                )
        except Exception as exc:
            if "UNIQUE constraint failed: writing_runs.project_id" in str(exc):
                raise ValueError(
                    f"chapter {project_id}:{chapter_number} already has an active run"
                ) from exc
            raise
        return self.get_writing_run(run_id)

    def get_writing_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM writing_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_writing_runs(self, project_id: str, chapter_number: int | None = None) -> list[dict]:
        with self._conn() as conn:
            if chapter_number is None:
                rows = conn.execute(
                    "SELECT * FROM writing_runs WHERE project_id = ? ORDER BY created_at DESC",
                    (project_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM writing_runs WHERE project_id = ? "
                    "AND chapter_number = ? ORDER BY created_at DESC",
                    (project_id, chapter_number),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_writing_run(self, run_id: str, **fields) -> None:
        allowed = {
            "status",
            "current_node",
            "current_version_id",
            "retry_count",
            "lease_owner",
            "lease_expires_at",
            "error_code",
            "error_message",
            "started_at",
            "finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{key} = ?" for key in updates)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE writing_runs SET {set_clause} WHERE id = ?",
                [*updates.values(), run_id],
            )

    def transition_writing_run(self, run_id: str, new_status: str, **fields) -> dict:
        """Apply one whitelisted lifecycle transition atomically."""
        transitions = {
            RunStatus.QUEUED.value: {
                RunStatus.RUNNING.value,
                RunStatus.CANCELLED.value,
                RunStatus.EXPIRED.value,
                RunStatus.WAITING_REVIEW.value,
            },
            RunStatus.RUNNING.value: {
                RunStatus.WAITING_REVIEW.value,
                RunStatus.WAITING_USER.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
                RunStatus.EXPIRED.value,
            },
            RunStatus.WAITING_REVIEW.value: {
                RunStatus.SUCCEEDED.value,
                RunStatus.RETRYING.value,
                RunStatus.CANCELLED.value,
            },
            RunStatus.WAITING_USER.value: {
                RunStatus.RUNNING.value,
                RunStatus.RETRYING.value,
                RunStatus.CANCELLED.value,
            },
            RunStatus.RETRYING.value: {
                RunStatus.QUEUED.value,
                RunStatus.RUNNING.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            },
            RunStatus.FAILED.value: {
                RunStatus.RETRYING.value,
                RunStatus.CANCELLED.value,
            },
        }
        allowed = {
            "current_node",
            "current_version_id",
            "retry_count",
            "lease_owner",
            "lease_expires_at",
            "error_code",
            "error_message",
            "started_at",
            "finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = new_status
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM writing_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Run not found")
            old_status = row["status"]
            if new_status not in transitions.get(old_status, set()):
                raise ValueError(f"invalid Run transition: {old_status} -> {new_status}")
            set_clause = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"UPDATE writing_runs SET {set_clause} WHERE id = ? AND status = ?",
                [*updates.values(), run_id, old_status],
            )
        return self.get_writing_run(run_id)

    def create_chapter_version(
        self,
        project_id: str,
        chapter_number: int,
        content: str,
        *,
        run_id: str | None = None,
        parent_version_id: str | None = None,
        origin: str = "initial_generation",
        status: str = "candidate",
        scene_plan: list[dict] | None = None,
        scene_drafts: list[str] | None = None,
    ) -> dict:
        """Append an immutable candidate version for a chapter."""
        import json

        version_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        scene_manifest = []
        for index, draft in enumerate(scene_drafts or [], start=1):
            plan = (scene_plan or [])[index - 1] if scene_plan and index <= len(scene_plan) else {}
            scene_manifest.append(
                {
                    "scene_index": plan.get("scene_index", index),
                    "outline": plan.get("outline", ""),
                    "target_words": plan.get("target_words", 0),
                    "content": draft,
                }
            )
        with self._conn() as conn:
            if parent_version_id is None:
                parent = conn.execute(
                    "SELECT id FROM chapter_versions "
                    "WHERE project_id = ? AND chapter_number = ? "
                    "ORDER BY version_number DESC LIMIT 1",
                    (project_id, chapter_number),
                ).fetchone()
                parent_version_id = parent["id"] if parent else None
            previous = conn.execute(
                "SELECT COALESCE(MAX(version_number), 0) AS version_number "
                "FROM chapter_versions WHERE project_id = ? AND chapter_number = ?",
                (project_id, chapter_number),
            ).fetchone()
            version_number = int(previous["version_number"]) + 1
            conn.execute(
                "INSERT INTO chapter_versions "
                "(id, project_id, chapter_number, run_id, version_number, "
                "parent_version_id, content, content_hash, word_count, origin, status, "
                "scene_manifest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version_id,
                    project_id,
                    chapter_number,
                    run_id,
                    version_number,
                    parent_version_id,
                    content,
                    content_hash,
                    len(content),
                    origin,
                    status,
                    json.dumps(scene_manifest, ensure_ascii=False),
                ),
            )
        return self.get_chapter_version(version_id)

    def create_scene_revision(
        self,
        version_id: str,
        scene_index: int,
        content: str,
        *,
        run_id: str | None = None,
    ) -> dict:
        """Replace one scene and append a complete chapter candidate version."""
        import json

        version = self.get_chapter_version(version_id)
        if not version:
            raise ValueError(f"chapter version not found: {version_id}")
        try:
            manifest = json.loads(version.get("scene_manifest") or "[]")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("chapter version has no valid scene manifest") from exc
        target = next(
            (scene for scene in manifest if scene.get("scene_index") == scene_index),
            None,
        )
        if target is None:
            raise ValueError(f"scene not found: {scene_index}")
        target["content"] = content
        assembled = "\n\n".join(
            scene.get("content", "").strip()
            for scene in manifest
            if scene.get("content", "").strip()
        )
        return self.create_chapter_version(
            version["project_id"],
            version["chapter_number"],
            assembled,
            run_id=run_id,
            parent_version_id=version_id,
            origin="scene_rewrite",
            scene_plan=manifest,
            scene_drafts=[scene.get("content", "") for scene in manifest],
        )

    def get_chapter_version(self, version_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chapter_versions WHERE id = ?", (version_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_scene_manifest(self, version_id: str) -> list[dict]:
        import json

        version = self.get_chapter_version(version_id)
        if not version:
            return []
        try:
            manifest = json.loads(version.get("scene_manifest") or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
        return manifest if isinstance(manifest, list) else []

    def list_chapter_versions(self, project_id: str, chapter_number: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapter_versions WHERE project_id = ? "
                "AND chapter_number = ? ORDER BY version_number",
                (project_id, chapter_number),
            ).fetchall()
        return [dict(row) for row in rows]

    def commit_chapter_version(self, version_id: str) -> dict:
        """Publish a candidate version and finish its run in one SQLite transaction."""
        existing_version = self.get_chapter_version(version_id)
        if not existing_version:
            raise ValueError(f"chapter version not found: {version_id}")
        if existing_version["status"] == "approved":
            return existing_version
        with self._conn() as conn:
            version = conn.execute(
                "SELECT * FROM chapter_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if not version:
                raise ValueError(f"chapter version not found: {version_id}")
            conn.execute(
                "INSERT INTO chapters "
                "(id, project_id, chapter_number, draft_content, status, version, "
                "current_version_id, approved_version_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(project_id, chapter_number) DO UPDATE SET "
                "draft_content = excluded.draft_content, "
                "status = excluded.status, version = excluded.version, "
                "current_version_id = excluded.current_version_id, "
                "approved_version_id = excluded.approved_version_id, "
                "updated_at = datetime('now')",
                (
                    str(uuid.uuid4())[:8],
                    version["project_id"],
                    version["chapter_number"],
                    version["content"],
                    ChapterStatus.APPROVED.value,
                    version["version_number"],
                    version["id"],
                    version["id"],
                ),
            )
            conn.execute(
                "UPDATE chapter_versions SET status = 'approved' WHERE id = ?",
                (version_id,),
            )
            if version["run_id"]:
                conn.execute(
                    "UPDATE writing_runs SET status = ?, current_version_id = ?, "
                    "finished_at = datetime('now') WHERE id = ?",
                    (RunStatus.SUCCEEDED.value, version_id, version["run_id"]),
                )
            conn.execute(
                "UPDATE outlines SET status = ? WHERE project_id = ? AND chapter_number = ?",
                (
                    OutlineStatus.APPROVED.value,
                    version["project_id"],
                    version["chapter_number"],
                ),
            )
            conn.execute(
                "UPDATE projects SET updated_at = datetime('now') WHERE id = ?",
                (version["project_id"],),
            )
            import json

            conn.execute(
                "INSERT OR IGNORE INTO outbox_events "
                "(id, project_id, aggregate_type, aggregate_id, event_type, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    version["project_id"],
                    "chapter_version",
                    version["id"],
                    "chapter_committed",
                    json.dumps(
                        {
                            "version_id": version["id"],
                            "chapter_number": version["chapter_number"],
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        return self.get_chapter_version(version_id)

    def commit_run(self, run_id: str) -> dict:
        """Commit a chapter version and its accepted Canon proposals atomically."""
        run = self.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        version_id = run.get("current_version_id")
        if not version_id:
            raise ValueError("Run has no candidate version")
        if run["status"] not in {RunStatus.WAITING_REVIEW.value, RunStatus.SUCCEEDED.value}:
            raise ValueError("Run is not waiting for review")
        with self._conn() as conn:
            version = conn.execute(
                "SELECT * FROM chapter_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if not version:
                raise ValueError(f"chapter version not found: {version_id}")
            self._commit_version_in_conn(conn, version)
            self._commit_canon_proposals_in_conn(conn, run_id)
        return self.get_chapter_version(version_id)

    def _commit_version_in_conn(self, conn, version) -> None:
        conn.execute(
            "INSERT INTO chapters "
            "(id, project_id, chapter_number, draft_content, status, version, "
            "current_version_id, approved_version_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, chapter_number) DO UPDATE SET "
            "draft_content = excluded.draft_content, status = excluded.status, "
            "version = excluded.version, current_version_id = excluded.current_version_id, "
            "approved_version_id = excluded.approved_version_id, updated_at = datetime('now')",
            (
                str(uuid.uuid4())[:8],
                version["project_id"],
                version["chapter_number"],
                version["content"],
                ChapterStatus.APPROVED.value,
                version["version_number"],
                version["id"],
                version["id"],
            ),
        )
        conn.execute(
            "UPDATE chapter_versions SET status = 'approved' WHERE id = ?",
            (version["id"],),
        )
        if version["run_id"]:
            conn.execute(
                "UPDATE writing_runs SET status = ?, current_version_id = ?, "
                "finished_at = datetime('now') WHERE id = ?",
                (RunStatus.SUCCEEDED.value, version["id"], version["run_id"]),
            )
        conn.execute(
            "UPDATE outlines SET status = ? WHERE project_id = ? AND chapter_number = ?",
            (OutlineStatus.APPROVED.value, version["project_id"], version["chapter_number"]),
        )
        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?",
            (version["project_id"],),
        )
        conn.execute(
            "INSERT OR IGNORE INTO outbox_events "
            "(id, project_id, aggregate_type, aggregate_id, event_type, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                version["project_id"],
                "chapter_version",
                version["id"],
                "chapter_committed",
                json.dumps(
                    {"version_id": version["id"], "chapter_number": version["chapter_number"]},
                    ensure_ascii=False,
                ),
            ),
        )

    # ── Outbox / derived indexes ──

    def get_outbox_event(self, event_id: str) -> dict | None:
        import json

        with self._conn() as conn:
            row = conn.execute("SELECT * FROM outbox_events WHERE id = ?", (event_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result["payload"] = json.loads(result["payload"] or "{}")
        except (TypeError, json.JSONDecodeError):
            result["payload"] = {}
        return result

    def list_outbox_events(
        self,
        project_id: str,
        *,
        status: str | None = None,
    ) -> list[dict]:
        query = "SELECT id FROM outbox_events WHERE project_id = ?"
        params: list = [project_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self.get_outbox_event(row["id"]) for row in rows]

    def list_all_outbox_events(self, *, status: str | None = None) -> list[dict]:
        query = "SELECT id FROM outbox_events"
        params: list = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self.get_outbox_event(row["id"]) for row in rows]

    def process_outbox_event(self, event_id: str, owner: str | None = None) -> dict:
        """Process one derived-state event; failures remain retryable."""
        event = self.get_outbox_event(event_id)
        if not event:
            raise ValueError(f"outbox event not found: {event_id}")
        if event["status"] == OutboxStatus.DONE.value:
            return event
        lease_owner = event.get("lease_owner")
        if event["status"] == OutboxStatus.PROCESSING.value and not lease_owner:
            raise ValueError("outbox event is processing without an owner")
        with self._conn() as conn:
            if owner and lease_owner != owner:
                raise ValueError("outbox event lease is owned by another worker")
            effective_owner = lease_owner or owner
            if effective_owner:
                current = conn.execute(
                    "SELECT lease_owner FROM outbox_events WHERE id = ?",
                    (event_id,),
                ).fetchone()
                if not current or current["lease_owner"] != effective_owner:
                    raise ValueError("outbox event lease is owned by another worker")
            conn.execute(
                "UPDATE outbox_events SET status = ?, lease_owner = ?, "
                "lease_expires_at = NULL WHERE id = ?",
                (OutboxStatus.PROCESSING.value, effective_owner, event_id),
            )
        try:
            if event["event_type"] == "chapter_committed":
                version = self.get_chapter_version(event["payload"]["version_id"])
                if not version:
                    raise ValueError("committed chapter version is missing")
                self.chapter_store.index_chapter(
                    event["project_id"],
                    version["chapter_number"],
                    version["content"],
                )
            with self._conn() as conn:
                conn.execute(
                    "UPDATE outbox_events SET status = ?, "
                    "processed_at = datetime('now'), last_error = '', "
                    "lease_owner = NULL, lease_expires_at = NULL WHERE id = ?",
                    (OutboxStatus.DONE.value, event_id),
                )
        except Exception as exc:  # noqa: BLE001 - persist retryable delivery failure
            with self._conn() as conn:
                conn.execute(
                    "UPDATE outbox_events SET status = ?, retry_count = retry_count + 1, "
                    "last_error = ?, lease_owner = NULL, lease_expires_at = NULL "
                    "WHERE id = ?",
                    (
                        OutboxStatus.STALE.value
                        if event["event_type"] == "chapter_committed"
                        else OutboxStatus.FAILED.value,
                        str(exc),
                        event_id,
                    ),
                )
        return self.get_outbox_event(event_id)

    def claim_outbox_events(
        self,
        owner: str,
        *,
        limit: int = 10,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> list[dict]:
        """Atomically claim pending, retryable, or expired events."""
        now = datetime.now(UTC).replace(tzinfo=None)
        now_text = now.strftime("%Y-%m-%d %H:%M:%S")
        expires_text = (now + timedelta(seconds=max(lease_seconds, 1))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        with self._conn() as conn:
            rows = conn.execute(
                "UPDATE outbox_events SET status = ?, lease_owner = ?, "
                "lease_expires_at = ? WHERE id IN ("
                "SELECT id FROM outbox_events WHERE "
                "(status IN (?, ?) AND retry_count < ?) OR "
                "(status = ? AND (lease_expires_at IS NULL OR lease_expires_at <= ?)) "
                "ORDER BY created_at LIMIT ?"
                ") RETURNING id",
                (
                    OutboxStatus.PROCESSING.value,
                    owner,
                    expires_text,
                    OutboxStatus.PENDING.value,
                    OutboxStatus.STALE.value,
                    max_retries,
                    OutboxStatus.PROCESSING.value,
                    now_text,
                    max(limit, 1),
                ),
            ).fetchall()
            ids = [row["id"] for row in rows]
        return [self.get_outbox_event(event_id) for event_id in ids]

    def retry_outbox_event(self, event_id: str) -> dict:
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE outbox_events SET status = ?, last_error = '', "
                "lease_owner = NULL, lease_expires_at = NULL "
                "WHERE id = ? AND status IN (?, ?)",
                (
                    OutboxStatus.PENDING.value,
                    event_id,
                    OutboxStatus.FAILED.value,
                    OutboxStatus.STALE.value,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("outbox event is missing or not failed")
        return self.get_outbox_event(event_id)

    # ── Canon proposals ──

    def create_canon_proposal(
        self,
        project_id: str,
        chapter_number: int,
        proposal_type: str,
        payload: dict,
        *,
        run_id: str | None = None,
        version_id: str | None = None,
    ) -> dict:
        import json

        proposal_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO canon_proposals "
                "(id, project_id, chapter_number, run_id, version_id, "
                "proposal_type, payload, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal_id,
                    project_id,
                    chapter_number,
                    run_id,
                    version_id,
                    proposal_type,
                    json.dumps(payload, ensure_ascii=False),
                    ProposalStatus.PROPOSED.value,
                ),
            )
        return self.get_canon_proposal(proposal_id)

    def get_canon_proposal(self, proposal_id: str) -> dict | None:
        import json

        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM canon_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result["payload"] = json.loads(result["payload"] or "{}")
        except (TypeError, json.JSONDecodeError):
            result["payload"] = {}
        return result

    def list_canon_proposals(
        self,
        project_id: str | None,
        *,
        run_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        query = "SELECT id FROM canon_proposals WHERE 1 = 1"
        params: list = []
        if project_id is not None:
            query += " AND project_id = ?"
            params.append(project_id)
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self.get_canon_proposal(row["id"]) for row in rows]

    def review_canon_proposal(
        self,
        proposal_id: str,
        status: str,
        reviewer_note: str = "",
    ) -> dict:
        if status not in {
            ProposalStatus.ACCEPTED.value,
            ProposalStatus.REJECTED.value,
        }:
            raise ValueError(f"invalid proposal review status: {status}")
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE canon_proposals SET status = ?, reviewer_note = ?, "
                "reviewed_at = datetime('now') WHERE id = ? "
                "AND status = ?",
                (
                    status,
                    reviewer_note,
                    proposal_id,
                    ProposalStatus.PROPOSED.value,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError("proposal is missing or no longer reviewable")
        return self.get_canon_proposal(proposal_id)

    def commit_canon_proposals(self, run_id: str) -> list[dict]:
        """Apply accepted proposals atomically to the Canon projection."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM canon_proposals WHERE run_id = ? AND status = ? ORDER BY created_at",
                (run_id, ProposalStatus.ACCEPTED.value),
            ).fetchall()
            for row in rows:
                proposal = dict(row)
                proposal["payload"] = json.loads(proposal["payload"] or "{}")
                if proposal["proposal_type"] == "worldbuilding":
                    self._apply_worldbuilding_proposal_in_conn(
                        conn,
                        proposal["project_id"],
                        proposal["chapter_number"],
                        proposal["payload"],
                        proposal.get("version_id"),
                    )
                conn.execute(
                    "UPDATE canon_proposals SET status = ?, "
                    "committed_at = datetime('now') WHERE id = ? AND status = ?",
                    (
                        ProposalStatus.COMMITTED.value,
                        proposal["id"],
                        ProposalStatus.ACCEPTED.value,
                    ),
                )
        return [self.get_canon_proposal(row["id"]) for row in rows]

    def _commit_canon_proposals_in_conn(self, conn, run_id: str) -> None:
        rows = conn.execute(
            "SELECT * FROM canon_proposals WHERE run_id = ? AND status = ? ORDER BY created_at",
            (run_id, ProposalStatus.ACCEPTED.value),
        ).fetchall()
        for row in rows:
            proposal = dict(row)
            proposal["payload"] = json.loads(proposal["payload"] or "{}")
            if proposal["proposal_type"] == "worldbuilding":
                self._apply_worldbuilding_proposal_in_conn(
                    conn,
                    proposal["project_id"],
                    proposal["chapter_number"],
                    proposal["payload"],
                    proposal.get("version_id"),
                )
            conn.execute(
                "UPDATE canon_proposals SET status = ?, committed_at = datetime('now') "
                "WHERE id = ? AND status = ?",
                (
                    ProposalStatus.COMMITTED.value,
                    proposal["id"],
                    ProposalStatus.ACCEPTED.value,
                ),
            )

    def _apply_worldbuilding_proposal(
        self,
        project_id: str,
        chapter_number: int,
        report: dict,
        source_version_id: str | None = None,
    ) -> None:
        with self._conn() as conn:
            self._apply_worldbuilding_proposal_in_conn(
                conn, project_id, chapter_number, report, source_version_id
            )

    def _apply_worldbuilding_proposal_in_conn(
        self,
        conn,
        project_id: str,
        chapter_number: int,
        report: dict,
        source_version_id: str | None = None,
    ) -> None:
        for event in report.get("chapter_events", []) or []:
            if isinstance(event, str):
                event = {"action": event}
            if not isinstance(event, dict):
                continue
            action = str(event.get("action") or event.get("description") or "")
            if action:
                conn.execute(
                    "INSERT OR IGNORE INTO story_events "
                    "(id, project_id, chapter_number, event_type, subject, action, "
                    "object_value, location, story_time, causality, evidence, source_version_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        project_id,
                        chapter_number,
                        str(event.get("event_type", "chapter_event")),
                        str(event.get("subject", "")),
                        action,
                        str(event.get("object", event.get("object_value", ""))),
                        str(event.get("location", "")),
                        str(event.get("time", event.get("story_time", ""))),
                        str(event.get("causality", "")),
                        str(event.get("evidence", "")),
                        source_version_id,
                    ),
                )

        entities = (report.get("new_entities", []) or []) + (
            report.get("updated_entities", []) or []
        )
        for entity in entities:
            if isinstance(entity, str):
                entity = {"name": entity}
            if not isinstance(entity, dict) or not entity.get("name"):
                continue
            entity_type = entity.get("entity_type", "unknown")
            name = entity["name"]
            incoming = entity.get("properties", {})
            incoming = incoming if isinstance(incoming, dict) else {}
            query = (
                "SELECT id, properties, first_appearance_chapter FROM world_entities "
                "WHERE project_id = ? AND name = ? ORDER BY id LIMIT 1"
                if entity_type == "unknown"
                else "SELECT id, properties, first_appearance_chapter FROM world_entities "
                "WHERE project_id = ? AND entity_type = ? AND name = ?"
            )
            params = (
                (project_id, name) if entity_type == "unknown" else (project_id, entity_type, name)
            )
            existing = conn.execute(query, params).fetchone()
            if existing:
                try:
                    current = json.loads(existing["properties"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    current = {}
                current = current if isinstance(current, dict) else {}
                current.update(incoming)
                conn.execute(
                    "UPDATE world_entities SET properties = ? WHERE id = ?",
                    (json.dumps(current, ensure_ascii=False), existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO world_entities "
                    "(id, project_id, entity_type, name, properties, first_appearance_chapter) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4())[:8],
                        project_id,
                        entity_type,
                        name,
                        json.dumps(incoming, ensure_ascii=False),
                        entity.get("first_appearance_chapter") or chapter_number,
                    ),
                )

        for entity in report.get("new_entities", []) or []:
            if not isinstance(entity, dict) or not entity.get("name"):
                continue
            for relation in entity.get("relationships", []) or []:
                if isinstance(relation, dict) and relation.get("target"):
                    conn.execute(
                        "INSERT OR IGNORE INTO world_relations "
                        "(id, project_id, source, target, relation_type, first_appearance_chapter) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4())[:8],
                            project_id,
                            entity["name"],
                            relation["target"],
                            relation.get("relation", "related_to"),
                            chapter_number,
                        ),
                    )

        for item in report.get("foreshadowings", []) or []:
            if not isinstance(item, dict) or not item.get("description"):
                continue
            description = str(item["description"])
            planted = int(item.get("planted_chapter", chapter_number))
            values = (
                "open",
                item.get("expected_resolve_chapter"),
                str(item.get("risk_level", "medium")),
                str(item.get("action_needed", "maintain")),
                project_id,
                description,
                planted,
            )
            updated = conn.execute(
                "UPDATE foreshadowings SET status = ?, expected_resolve_chapter = ?, "
                "risk_level = ?, action_needed = ? WHERE project_id = ? "
                "AND description = ? AND planted_chapter = ?",
                values,
            ).rowcount
            if not updated:
                conn.execute(
                    "INSERT INTO foreshadowings "
                    "(id, project_id, description, planted_chapter, expected_resolve_chapter, "
                    "status, risk_level, action_needed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4())[:8],
                        project_id,
                        description,
                        planted,
                        item.get("expected_resolve_chapter"),
                        "open",
                        str(item.get("risk_level", "medium")),
                        str(item.get("action_needed", "maintain")),
                    ),
                )
        for item in report.get("resolved_foreshadowings", []) or []:
            if isinstance(item, dict) and item.get("description"):
                params = [chapter_number, project_id, str(item["description"])]
                where = "project_id = ? AND description = ?"
                if item.get("planted_chapter") is not None:
                    params.append(item["planted_chapter"])
                    where += " AND planted_chapter = ?"
                conn.execute(
                    f"UPDATE foreshadowings SET status = 'resolved', "
                    f"resolved_chapter = ? WHERE {where}",
                    params,
                )

    def save_story_events(
        self,
        project_id: str,
        chapter_number: int,
        events: list,
        *,
        source_version_id: str | None = None,
    ) -> list[dict]:
        """Persist normalized chapter events idempotently."""
        normalized = []
        for event in events:
            if isinstance(event, str):
                normalized.append({"action": event})
            elif isinstance(event, dict):
                normalized.append(event)
        with self._conn() as conn:
            for event in normalized:
                action = str(event.get("action") or event.get("description") or "")
                if not action:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO story_events "
                    "(id, project_id, chapter_number, event_type, subject, action, "
                    "object_value, location, story_time, causality, evidence, source_version_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        project_id,
                        chapter_number,
                        str(event.get("event_type", "chapter_event")),
                        str(event.get("subject", "")),
                        action,
                        str(event.get("object", event.get("object_value", ""))),
                        str(event.get("location", "")),
                        str(event.get("time", event.get("story_time", ""))),
                        str(event.get("causality", "")),
                        str(event.get("evidence", "")),
                        source_version_id,
                    ),
                )
        return self.get_story_events(project_id, chapter_number)

    def get_story_events(
        self,
        project_id: str,
        chapter_number: int | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM story_events WHERE project_id = ?"
        params: list = [project_id]
        if chapter_number is not None:
            query += " AND chapter_number = ?"
            params.append(chapter_number)
        query += " ORDER BY chapter_number, created_at"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

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
                    "UPDATE outlines SET status = ? WHERE project_id = ? AND chapter_number = ?",
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

    def get_recent_chapters(
        self,
        project_id: str,
        before: int,
        limit: int = 5,
    ) -> list[dict]:
        """Most recent chapters before a chapter number, newest first.

        Ordered by chapter_number DESC in SQL so long projects don't load the
        full chapter table into Python just to slice the tail.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chapters WHERE project_id = ? AND chapter_number < ? "
                "AND status != 'failed' ORDER BY chapter_number DESC LIMIT ?",
                (project_id, before, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_chapters(self, project_id: str, before: int | None = None) -> int:
        """Count stored (non-failed) chapters, optionally only before a chapter number."""
        query = "SELECT COUNT(*) FROM chapters WHERE project_id = ? AND status != 'failed'"
        params: list = [project_id]
        if before is not None:
            query += " AND chapter_number < ?"
            params.append(before)
        with self._conn() as conn:
            (count,) = conn.execute(query, params).fetchone()
        return count

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

    def build_context(
        self,
        project_id: str,
        chapter_number: int,
        max_recent_chapters: int = 3,
    ) -> dict[str, str]:
        """Build context for writing a chapter: recent summary + character/world info."""
        # SQL-bounded window read — get_all_chapters would load every draft
        # in the project just to keep the last few summaries.
        recent = list(
            reversed(
                self.get_recent_chapters(
                    project_id,
                    before=chapter_number,
                    limit=max_recent_chapters,
                )
            )
        )

        # Recent summary from last N chapters
        recent_summary_parts = []
        for c in recent[-max_recent_chapters:]:
            draft = c.get("draft_content", "")
            if draft:
                recent_summary_parts.append(
                    f"第{c['chapter_number']}章: {draft[:300]}..."
                    if len(draft) > 300
                    else f"第{c['chapter_number']}章: {draft}"
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

        character_context = (
            "\n".join(f"- {c['name']}: {c['properties']}" for c in chars) if chars else ""
        )

        world_context = (
            "\n".join(f"- [{e['entity_type']}] {e['name']}: {e['properties']}" for e in world_ents)
            if world_ents
            else ""
        )

        return {
            "recent_summary": recent_summary,
            "character_context": character_context,
            "world_context": world_context,
        }

    def build_context_from_snapshot(
        self, snapshot: dict, chapter_number: int, max_recent_chapters: int = 3
    ) -> dict[str, str]:
        """Build context only from the immutable Canon snapshot payload."""
        payload = snapshot.get("payload", snapshot)
        chapters = [
            chapter
            for chapter in payload.get("chapters", [])
            if chapter.get("chapter_number", 0) < chapter_number
        ]
        recent_parts = []
        for chapter in chapters[-max_recent_chapters:]:
            draft = chapter.get("draft_content", "")
            if draft:
                recent_parts.append(
                    f"第{chapter['chapter_number']}章: {draft[:300]}..."
                    if len(draft) > 300
                    else f"第{chapter['chapter_number']}章: {draft}"
                )
        characters = [
            entity
            for entity in payload.get("entities", [])
            if entity.get("entity_type") == "character"
        ]
        world_entities = [
            entity
            for entity in payload.get("entities", [])
            if entity.get("entity_type") != "character"
        ]
        return {
            "recent_summary": "\n\n".join(recent_parts),
            "character_context": "\n".join(
                f"- {entity['name']}: {entity['properties']}" for entity in characters
            ),
            "world_context": "\n".join(
                f"- [{entity['entity_type']}] {entity['name']}: {entity['properties']}"
                for entity in world_entities
            ),
        }

    def get_all_world_entities(self, project_id: str) -> list[dict]:
        """Get all world entities for a project (for state population)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM world_entities WHERE project_id = ? ORDER BY entity_type, name",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_relevant_world_entities(
        self,
        project_id: str,
        draft_text: str,
        limit: int = 60,
    ) -> list[dict]:
        """Entities referenced by name in the draft — conflict candidates.

        Worldbuilding conflict detection only needs entities the current
        chapter actually touches; sending the full entity table into the
        extraction prompt grows unbounded on long projects. Small projects
        (within ``limit``) pass through unchanged.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM world_entities WHERE project_id = ? "
                "ORDER BY first_appearance_chapter, name",
                (project_id,),
            ).fetchall()
        entities = [dict(r) for r in rows]
        if len(entities) <= limit:
            return entities
        mentioned = [e for e in entities if e.get("name") and e["name"] in draft_text]
        return mentioned[:limit]

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
                        (
                            str(uuid.uuid4())[:8],
                            project_id,
                            entity_type,
                            name,
                            json.dumps(incoming, ensure_ascii=False),
                            chapter,
                        ),
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
                        (
                            str(uuid.uuid4())[:8],
                            project_id,
                            source,
                            target,
                            rel.get("relation", "related_to"),
                            chapter_number,
                        ),
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
                (
                    fid,
                    project_id,
                    description,
                    planted_chapter,
                    expected_resolve_chapter,
                    risk_level,
                    action_needed,
                    1 if reader_knows else 0,
                    _json.dumps(characters_aware or [], ensure_ascii=False),
                    _json.dumps(characters_unaware or [], ensure_ascii=False),
                ),
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
        allowed = {
            "risk_level",
            "action_needed",
            "status",
            "reader_knows",
            "characters_aware",
            "characters_unaware",
            "resolved_chapter",
            "expected_resolve_chapter",
        }
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
                f"UPDATE foreshadowings SET {set_clause} WHERE {where}",
                values,
            )
            updated = cur.rowcount > 0
        return updated

    def get_foreshadowings(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM foreshadowings WHERE project_id = ? ORDER BY planted_chapter",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_relevant_foreshadowings(
        self,
        project_id: str,
        current_chapter: int,
        limit: int = 25,
    ) -> list[dict]:
        """Unresolved foreshadowings ranked by deterministic relevance.

        Priority: risk level → urgency (distance to expected resolution) →
        planting recency. SQL-side filter + rank, so long projects never read
        the full foreshadowings table into the context packet.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM foreshadowings WHERE project_id = ? "
                "AND status IN ('open', 'planted', 'hinted', 'advanced') "
                "ORDER BY "
                "CASE risk_level WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
                "CASE WHEN expected_resolve_chapter IS NOT NULL "
                "     THEN ABS(expected_resolve_chapter - ?) ELSE 99999 END, "
                "planted_chapter DESC, created_at DESC "
                "LIMIT ?",
                (project_id, current_chapter, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_relevant_story_events(
        self,
        project_id: str,
        current_chapter: int,
        window: int = 30,
        limit: int = 90,
    ) -> list[dict]:
        """Events from the recent chapter window, ascending order.

        Bounded SQL read (chapters within ``window`` back, LIMIT rows) instead
        of the full story_events table. Distant history (e.g. a death planted
        100 chapters ago) is intentionally out of the v1 window.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM story_events WHERE project_id = ? "
                "AND chapter_number >= ? AND chapter_number < ? "
                "ORDER BY chapter_number DESC, created_at DESC LIMIT ?",
                (project_id, max(current_chapter - window, 0), current_chapter, limit),
            ).fetchall()
        rows.reverse()
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
