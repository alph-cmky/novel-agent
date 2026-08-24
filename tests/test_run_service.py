from unittest.mock import patch

import pytest

from novel_agent.api.run_service import ChapterRunService
from novel_agent.schema.enums import RunStatus
from novel_agent.storage.manager import ProjectManager


def test_service_commit_rejects_unreviewed_proposal(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)
    version = mgr.create_chapter_version(project_id, 1, "正文", run_id=run["id"])
    mgr.update_writing_run(
        run["id"],
        status=RunStatus.WAITING_REVIEW.value,
        current_version_id=version["id"],
    )
    mgr.create_canon_proposal(
        project_id,
        1,
        "worldbuilding",
        {"new_entities": [{"entity_type": "character", "name": "甲"}]},
        run_id=run["id"],
    )

    with pytest.raises(ValueError, match="unreviewed Canon proposals"):
        ChapterRunService(mgr).commit(run["id"])

    assert mgr.get_chapter_version(version["id"])["status"] == "candidate"


def test_transition_rejects_invalid_terminal_transition(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)
    mgr.transition_writing_run(run["id"], RunStatus.CANCELLED.value)

    with pytest.raises(ValueError, match="invalid Run transition"):
        mgr.transition_writing_run(run["id"], RunStatus.RUNNING.value)


def test_commit_run_rolls_back_version_when_canon_commit_fails(tmp_path):
    mgr = ProjectManager(tmp_path)
    project_id = mgr.init_project(name="p")
    run = mgr.create_writing_run(project_id, 1)
    version = mgr.create_chapter_version(project_id, 1, "正文", run_id=run["id"])
    mgr.update_writing_run(
        run["id"],
        status=RunStatus.WAITING_REVIEW.value,
        current_version_id=version["id"],
    )

    with patch.object(
        mgr, "_commit_canon_proposals_in_conn", side_effect=RuntimeError("canon failure")
    ):
        with pytest.raises(RuntimeError, match="canon failure"):
            ChapterRunService(mgr).commit(run["id"])

    assert mgr.get_chapter_version(version["id"])["status"] == "candidate"
    assert mgr.get_chapter(project_id, 1) is None
    assert mgr.get_writing_run(run["id"])["status"] == RunStatus.WAITING_REVIEW.value
