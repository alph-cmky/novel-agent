"""Application service for the durable V2 chapter run lifecycle."""


class ChapterRunService:
    """Keep Run/Version lifecycle operations out of transport adapters."""

    ACTIVE_STATUSES = {
        "queued", "running", "waiting_review", "waiting_user", "retrying",
    }

    def __init__(self, manager):
        self.manager = manager

    def create_run(self, project_id: str, chapter_number: int, **kwargs) -> dict:
        return self.manager.create_writing_run(project_id, chapter_number, **kwargs)

    def attach_candidate(self, run_id: str, content: str, **kwargs) -> dict:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] not in self.ACTIVE_STATUSES:
            raise ValueError("Run is not active")
        version = self.manager.create_chapter_version(
            run["project_id"], run["chapter_number"], content,
            run_id=run_id, **kwargs,
        )
        self.manager.transition_writing_run(
            run_id,
            "waiting_review",
            current_version_id=version["id"],
        )
        return version

    def commit(self, run_id: str) -> dict:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        version_id = run.get("current_version_id")
        if not version_id:
            raise ValueError("Run has no candidate version")
        pending = self.manager.list_canon_proposals(
            run["project_id"], run_id=run_id, status="proposed"
        )
        if pending:
            raise ValueError("Run has unreviewed Canon proposals")
        return self.manager.commit_run(run_id)

    def rewrite_scene(self, run_id: str, scene_index: int, content: str) -> dict:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        version_id = run.get("current_version_id")
        if not version_id:
            raise ValueError("Run has no current version")
        version = self.manager.create_scene_revision(
            version_id,
            scene_index,
            content,
            run_id=run_id,
        )
        self.manager.transition_writing_run(
            run_id,
            "waiting_review",
            current_version_id=version["id"],
            current_node="scene_review",
        )
        return version

    def cancel(self, run_id: str) -> dict:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] not in self.ACTIVE_STATUSES:
            raise ValueError("Run is not active")
        return self.manager.transition_writing_run(run_id, "cancelled")

    def retry(self, run_id: str) -> dict:
        run = self.manager.get_writing_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] != "failed":
            raise ValueError("Only failed runs can be retried")
        self.manager.transition_writing_run(
            run_id,
            "retrying",
            retry_count=(run.get("retry_count") or 0) + 1,
        )
        return self.manager.transition_writing_run(run_id, "queued")

    def list_versions(self, project_id: str, chapter_number: int) -> list[dict]:
        return self.manager.list_chapter_versions(project_id, chapter_number)

    def restore_version(self, version_id: str) -> dict:
        version = self.manager.get_chapter_version(version_id)
        if not version:
            raise ValueError("Version not found")
        new_version = self.manager.create_chapter_version(
            version["project_id"],
            version["chapter_number"],
            version["content"],
            parent_version_id=version_id,
            origin="restored",
        )
        return self.manager.commit_chapter_version(new_version["id"])
