"""Integration tests for V2 Run lifecycle — P0 coverage.

Tests the durable run state machine end-to-end through ProjectManager
and ChapterRunService, covering:
  1. Full approve → commit path
  2. Failed run does not corrupt previously approved data
  3. Concurrent runs rejected
  4. Canon proposal isolation (before/after commit, rejected never applied)
  5. Version immutability and auto-increment chain
"""

from unittest.mock import MagicMock, patch

import pytest

from novel_agent.api.run_service import ChapterRunService
from novel_agent.schema.enums import (
    ChapterStatus,
    OutboxStatus,
    ProposalStatus,
    RunStatus,
)
from novel_agent.storage.manager import ProjectManager


def _make_manager(tmp_path, chapter_store=None) -> ProjectManager:
    with patch("novel_agent.storage.manager.ChapterStore") as mock_store:
        mock_store.return_value = chapter_store or MagicMock()
        return ProjectManager(tmp_path)


# ── 1. Full approve → commit path ────────────────────────────────


class TestRunFullApproveCommitPath:
    def test_run_full_approve_commit_path(self, tmp_path):
        """End-to-end: queued → running → waiting_review → commit → succeeded."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="测试小说")
        mgr.save_outline(pid, [{"chapter_number": 1, "title": "开篇"}])

        # Create writing run — starts queued
        run = mgr.create_writing_run(pid, 1)
        assert run["status"] == RunStatus.QUEUED.value

        # Agent picks up work → running
        mgr.update_writing_run(run["id"], status=RunStatus.RUNNING.value)
        assert mgr.get_writing_run(run["id"])["status"] == RunStatus.RUNNING.value

        # Agent produces a candidate version
        version = mgr.create_chapter_version(pid, 1, "第一章正文内容", run_id=run["id"])
        assert version["status"] == "candidate"
        assert version["version_number"] == 1

        # Agent creates a canon proposal (worldbuilding)
        proposal = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {"new_entities": [{"entity_type": "character", "name": "林风"}]},
            run_id=run["id"],
        )
        assert proposal["status"] == ProposalStatus.PROPOSED.value

        # Agent moves to waiting_review
        mgr.update_writing_run(
            run["id"],
            status=RunStatus.WAITING_REVIEW.value,
            current_version_id=version["id"],
        )
        run_updated = mgr.get_writing_run(run["id"])
        assert run_updated["status"] == RunStatus.WAITING_REVIEW.value

        # User approves: commit the version
        committed = mgr.commit_chapter_version(version["id"])
        assert committed["status"] == "approved"

        # Run auto-transitions to succeeded
        assert mgr.get_writing_run(run["id"])["status"] == RunStatus.SUCCEEDED.value

        # Commit canon proposals for this run
        mgr.review_canon_proposal(proposal["id"], ProposalStatus.ACCEPTED.value, "确认")
        applied = mgr.commit_canon_proposals(run["id"])
        assert len(applied) == 1
        assert applied[0]["status"] == ProposalStatus.COMMITTED.value

        # Outbox event created
        events = mgr.list_outbox_events(pid)
        assert len(events) >= 1
        assert events[0]["event_type"] == "chapter_committed"
        assert events[0]["status"] == OutboxStatus.PENDING.value

        # Old chapters table projection updated
        chapter = mgr.get_chapter(pid, 1)
        assert chapter is not None
        assert chapter["approved_version_id"] == version["id"]
        assert chapter["status"] == ChapterStatus.APPROVED.value
        assert chapter["draft_content"] == "第一章正文内容"


# ── 2. Failed run does not corrupt approved data ─────────────────


class TestRunFailedDoesNotCorruptApproved:
    def test_run_failed_does_not_corrupt_approved(self, tmp_path):
        """A failed run2 must not affect data committed by run1."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        # run1: succeed and commit
        run1 = mgr.create_writing_run(pid, 1)
        v1 = mgr.create_chapter_version(pid, 1, "run1正文", run_id=run1["id"])
        mgr.update_writing_run(
            run1["id"],
            status=RunStatus.WAITING_REVIEW.value,
            current_version_id=v1["id"],
        )
        mgr.commit_chapter_version(v1["id"])
        assert mgr.get_writing_run(run1["id"])["status"] == RunStatus.SUCCEEDED.value

        # run2: starts then fails
        run2 = mgr.create_writing_run(pid, 1)
        mgr.update_writing_run(run2["id"], status=RunStatus.RUNNING.value)
        mgr.create_chapter_version(pid, 1, "run2候选", run_id=run2["id"])
        mgr.update_writing_run(
            run2["id"],
            status=RunStatus.FAILED.value,
            error_message="LLM timeout",
        )

        # run2 is failed
        assert mgr.get_writing_run(run2["id"])["status"] == RunStatus.FAILED.value

        # run1's approved chapter is intact
        chapter = mgr.get_chapter(pid, 1)
        assert chapter["approved_version_id"] == v1["id"]
        assert chapter["draft_content"] == "run1正文"
        assert chapter["status"] == ChapterStatus.APPROVED.value

        # v1 content unchanged
        assert mgr.get_chapter_version(v1["id"])["content"] == "run1正文"
        assert mgr.get_chapter_version(v1["id"])["status"] == "approved"


# ── 3. Concurrent runs rejected ──────────────────────────────────


class TestConcurrentRunsRejected:
    def test_concurrent_runs_rejected(self, tmp_path):
        """Only one active run per chapter; second attempt raises ValueError."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        run1 = mgr.create_writing_run(pid, 1)
        assert run1["status"] == RunStatus.QUEUED.value

        # Attempt a second active run → rejected
        with pytest.raises(ValueError, match="active run"):
            mgr.create_writing_run(pid, 1)

        # Complete run1
        v1 = mgr.create_chapter_version(pid, 1, "内容", run_id=run1["id"])
        mgr.update_writing_run(
            run1["id"],
            status=RunStatus.WAITING_REVIEW.value,
            current_version_id=v1["id"],
        )
        mgr.commit_chapter_version(v1["id"])
        assert mgr.get_writing_run(run1["id"])["status"] == RunStatus.SUCCEEDED.value

        # Now a second run succeeds
        run2 = mgr.create_writing_run(pid, 1)
        assert run2["status"] == RunStatus.QUEUED.value
        assert run2["id"] != run1["id"]

    def test_cancelled_run_unblocks_new_run(self, tmp_path):
        """Cancelling an active run should allow a new run to be created."""
        mgr = _make_manager(tmp_path)
        svc = ChapterRunService(mgr)
        pid = mgr.init_project(name="p")

        run1 = svc.create_run(pid, 1)
        cancelled = svc.cancel(run1["id"])
        assert cancelled["status"] == RunStatus.CANCELLED.value

        run2 = svc.create_run(pid, 1)
        assert run2["status"] == RunStatus.QUEUED.value

    def test_failed_run_allows_retry_then_new_run(self, tmp_path):
        """Failed → retry cycle through ChapterRunService."""
        mgr = _make_manager(tmp_path)
        svc = ChapterRunService(mgr)
        pid = mgr.init_project(name="p")

        run = svc.create_run(pid, 1)
        mgr.update_writing_run(run["id"], status=RunStatus.FAILED.value)

        retried = svc.retry(run["id"])
        assert retried["status"] == RunStatus.QUEUED.value
        assert retried["retry_count"] == 1


# ── 4. Canon proposal isolation ──────────────────────────────────


class TestCanonProposalIsolation:
    def test_proposals_isolated_until_commit(self, tmp_path):
        """Entities from proposals must not appear in world_entities until commit."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)

        proposal = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {
                "new_entities": [
                    {"entity_type": "character", "name": "洛千秋"},
                    {"entity_type": "item", "name": "星辰剑"},
                ]
            },
            run_id=run["id"],
        )

        # Before commit: entities NOT in world_entities
        assert mgr.get_all_world_entities(pid) == []

        # Accept and commit
        mgr.review_canon_proposal(proposal["id"], "accepted")
        committed = mgr.commit_canon_proposals(run["id"])
        assert len(committed) == 1
        assert committed[0]["status"] == ProposalStatus.COMMITTED.value

        # After commit: entities ARE in world_entities
        entities = mgr.get_all_world_entities(pid)
        names = {e["name"] for e in entities}
        assert "洛千秋" in names
        assert "星辰剑" in names

    def test_rejected_proposals_never_enter_world_entities(self, tmp_path):
        """Rejected proposals must never be applied to world_entities."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)

        accepted_prop = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {"new_entities": [{"entity_type": "character", "name": "好角色"}]},
            run_id=run["id"],
        )
        rejected_prop = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {"new_entities": [{"entity_type": "character", "name": "坏角色"}]},
            run_id=run["id"],
        )

        mgr.review_canon_proposal(accepted_prop["id"], "accepted")
        mgr.review_canon_proposal(rejected_prop["id"], "rejected", "不符合设定")

        mgr.commit_canon_proposals(run["id"])

        entities = mgr.get_all_world_entities(pid)
        names = {e["name"] for e in entities}
        assert "好角色" in names
        assert "坏角色" not in names

        # Rejected proposal stays rejected
        rejected = mgr.get_canon_proposal(rejected_prop["id"])
        assert rejected["status"] == ProposalStatus.REJECTED.value


# ── 5. Version immutability and chain ────────────────────────────


class TestVersionImmutability:
    def test_version_content_immutable_after_new_version(self, tmp_path):
        """Creating v2 must not alter v1's content."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        v1 = mgr.create_chapter_version(pid, 1, "第一版内容")
        assert v1["version_number"] == 1
        assert v1["content"] == "第一版内容"

        v2 = mgr.create_chapter_version(pid, 1, "第二版内容")
        assert v2["version_number"] == 2

        # v1 content unchanged
        v1_reloaded = mgr.get_chapter_version(v1["id"])
        assert v1_reloaded["content"] == "第一版内容"
        assert v1_reloaded["version_number"] == 1

    def test_version_number_auto_increments(self, tmp_path):
        """version_number must monotonically increase per chapter."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        v1 = mgr.create_chapter_version(pid, 1, "v1")
        v2 = mgr.create_chapter_version(pid, 1, "v2")
        v3 = mgr.create_chapter_version(pid, 1, "v3")

        assert v1["version_number"] == 1
        assert v2["version_number"] == 2
        assert v3["version_number"] == 3

    def test_parent_version_chain(self, tmp_path):
        """parent_version_id auto-links to previous version."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        v1 = mgr.create_chapter_version(pid, 1, "v1")
        v2 = mgr.create_chapter_version(pid, 1, "v2")
        v3 = mgr.create_chapter_version(pid, 1, "v3")

        assert v1["parent_version_id"] is None
        assert v2["parent_version_id"] == v1["id"]
        assert v3["parent_version_id"] == v2["id"]

    def test_different_chapters_have_independent_version_numbers(self, tmp_path):
        """Versions for chapter 1 and chapter 2 are numbered independently."""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        ch1_v1 = mgr.create_chapter_version(pid, 1, "ch1-v1")
        ch1_v2 = mgr.create_chapter_version(pid, 1, "ch1-v2")
        ch2_v1 = mgr.create_chapter_version(pid, 2, "ch2-v1")

        assert ch1_v1["version_number"] == 1
        assert ch1_v2["version_number"] == 2
        assert ch2_v1["version_number"] == 1
        assert ch2_v1["parent_version_id"] is None


# ── 6. Version restore via ChapterRunService ─────────────────────


class TestVersionRestore:
    def test_restore_creates_new_approved_version(self, tmp_path):
        """Restoring a version creates a new committed version with origin='restored'."""
        mgr = _make_manager(tmp_path)
        svc = ChapterRunService(mgr)
        pid = mgr.init_project(name="p")

        # Create and commit v1
        v1 = mgr.create_chapter_version(pid, 1, "原始内容v1")
        mgr.commit_chapter_version(v1["id"])

        # Create and commit v2 (overwrites chapter)
        v2 = mgr.create_chapter_version(pid, 1, "修改内容v2")
        mgr.commit_chapter_version(v2["id"])
        assert mgr.get_chapter(pid, 1)["draft_content"] == "修改内容v2"

        # Restore v1
        restored = svc.restore_version(v1["id"])
        assert restored["status"] == "approved"
        assert restored["content"] == "原始内容v1"

        # Chapter now reflects restored content
        chapter = mgr.get_chapter(pid, 1)
        assert chapter["draft_content"] == "原始内容v1"
        assert chapter["approved_version_id"] == restored["id"]

        # Restored version is a NEW version (v3), not v1 itself
        assert restored["version_number"] == 3
        assert restored["id"] != v1["id"]

        # Original v1 and v2 are untouched
        assert mgr.get_chapter_version(v1["id"])["content"] == "原始内容v1"
        assert mgr.get_chapter_version(v2["id"])["content"] == "修改内容v2"

    def test_restore_nonexistent_version_raises(self, tmp_path):
        """Restoring a non-existent version raises ValueError."""
        mgr = _make_manager(tmp_path)
        svc = ChapterRunService(mgr)

        with pytest.raises(ValueError, match="Version not found"):
            svc.restore_version("nonexistent-id")
