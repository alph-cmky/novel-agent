"""Tests for ProjectManager CRUD — SQLite 数据访问层。

锁住覆盖缺口：manager 是 novel-agent 的数据主干，但此前零测试。
用 tmp_path 隔离 SQLite，mock ChapterStore 避免真实 ChromaDB 落盘。
"""

from unittest.mock import MagicMock, patch

import pytest

from novel_agent.schema.enums import ChapterStatus, RunStatus
from novel_agent.storage.manager import ProjectManager


def _make_manager(tmp_path, chapter_store=None) -> ProjectManager:
    with patch("novel_agent.storage.manager.ChapterStore") as mock_store:
        mock_store.return_value = chapter_store or MagicMock()
        return ProjectManager(tmp_path)


class TestProjectCRUD:
    def test_commit_is_idempotent_and_emits_outbox_event(self, tmp_path):
        store = MagicMock()
        mgr = _make_manager(tmp_path, chapter_store=store)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        version = mgr.create_chapter_version(pid, 1, "正文", run_id=run["id"])

        mgr.commit_chapter_version(version["id"])
        mgr.commit_chapter_version(version["id"])

        events = mgr.list_outbox_events(pid)
        assert len(events) == 1
        assert events[0]["status"] == "pending"
        processed = mgr.process_outbox_event(events[0]["id"])
        assert processed["status"] == "done"
        store.index_chapter.assert_called_once_with(pid, 1, "正文")

    def test_writing_run_binds_immutable_canon_snapshot(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_world_entities(
            pid,
            {"new_entities": [{"entity_type": "character", "name": "甲"}]},
            1,
        )
        run = mgr.create_writing_run(pid, 2)
        snapshot = mgr.get_canon_snapshot(run["input_snapshot_id"])
        assert snapshot["payload"]["entities"][0]["name"] == "甲"
        mgr.save_world_entities(
            pid,
            {"new_entities": [{"entity_type": "character", "name": "乙"}]},
            2,
        )
        assert [
            entity["name"]
            for entity in mgr.get_canon_snapshot(run["input_snapshot_id"])["payload"]["entities"]
        ] == ["甲"]

    def test_outbox_failure_is_retryable(self, tmp_path):
        store = MagicMock()
        store.index_chapter.side_effect = RuntimeError("index unavailable")
        mgr = _make_manager(tmp_path, chapter_store=store)
        pid = mgr.init_project(name="p")
        version = mgr.create_chapter_version(pid, 1, "正文")
        mgr.commit_chapter_version(version["id"])
        event = mgr.list_outbox_events(pid)[0]

        failed = mgr.process_outbox_event(event["id"])
        assert failed["status"] == "stale"
        assert failed["retry_count"] == 1
        retried = mgr.retry_outbox_event(event["id"])
        assert retried["status"] == "pending"

    def test_outbox_claim_is_exclusive_and_expired_lease_is_reclaimable(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        version = mgr.create_chapter_version(pid, 1, "正文")
        mgr.commit_chapter_version(version["id"])
        event_id = mgr.list_outbox_events(pid)[0]["id"]

        first = mgr.claim_outbox_events("worker-a", lease_seconds=60)
        second = mgr.claim_outbox_events("worker-b", lease_seconds=60)
        assert [event["id"] for event in first] == [event_id]
        assert second == []
        with pytest.raises(ValueError, match="another worker"):
            mgr.process_outbox_event(event_id, owner="worker-b")

        with mgr._conn() as conn:
            conn.execute(
                "UPDATE outbox_events SET lease_expires_at = '2000-01-01 00:00:00' WHERE id = ?",
                (event_id,),
            )
        reclaimed = mgr.claim_outbox_events("worker-b", lease_seconds=60)
        assert [event["id"] for event in reclaimed] == [event_id]

    def test_init_and_get_project(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="测试", title="", genre="玄幻")

        assert len(pid) == 8
        proj = mgr.get_project(pid)
        assert proj["name"] == "测试"
        assert proj["title"] == "测试"  # title 为空 → 回退 name
        assert proj["genre"] == "玄幻"

    def test_list_projects(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.init_project(name="A")
        mgr.init_project(name="B")
        assert len(mgr.list_projects()) == 2

    def test_get_or_create_default_project(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.get_or_create_default_project()
        assert mgr.get_project(pid)["name"] == "default"


class TestChapterCRUD:
    def test_save_and_get_chapter(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        cid = mgr.save_chapter(pid, 1, outline="大纲", draft_content="正文")

        ch = mgr.get_chapter(pid, 1)
        assert ch["id"] == cid
        assert ch["draft_content"] == "正文"
        assert ch["outline"] == "大纲"

    def test_save_chapter_upserts(self, tmp_path):
        """同 project+chapter_number 重复 save 是 update 而非新增。"""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        cid1 = mgr.save_chapter(pid, 1, draft_content="v1")
        cid2 = mgr.save_chapter(pid, 1, draft_content="v2")

        assert cid1 == cid2  # ON CONFLICT ... RETURNING id 返回既有 id
        chapters = mgr.get_all_chapters(pid)
        assert len(chapters) == 1
        assert chapters[0]["draft_content"] == "v2"

    def test_get_recent_chapters_sql_tail_slice(self, tmp_path):
        """get_recent_chapters 在 SQL 侧取尾部切片：latest-first、排除 failed、不含当前章。"""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        for n in range(1, 6):
            mgr.save_chapter(pid, n, draft_content=f"第{n}章")
        mgr.mark_chapter_failed(pid, 2)
        mgr.save_chapter(pid, 7, draft_content="当前章")

        recent = mgr.get_recent_chapters(pid, before=7, limit=2)

        assert [c["chapter_number"] for c in recent] == [5, 4]
        assert mgr.count_chapters(pid, before=7) == 4
        assert mgr.count_chapters(pid) == 5

    def test_get_relevant_foreshadowings_ranks_and_filters(self, tmp_path):
        """Phase C: relevance 查询排除 resolved，按 risk → 紧迫度排序，LIMIT 生效。"""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.add_foreshadowing(pid, "已解决", planted_chapter=1, risk_level="high")
        mgr.update_foreshadowing_status(pid, "已解决", planted_chapter=1, status="resolved")
        mgr.add_foreshadowing(pid, "远期低危", planted_chapter=1, risk_level="low")
        mgr.add_foreshadowing(
            pid, "近章高危", planted_chapter=2, expected_resolve_chapter=5, risk_level="high"
        )
        mgr.add_foreshadowing(
            pid, "近章中危", planted_chapter=2, expected_resolve_chapter=5, risk_level="medium"
        )
        mgr.add_foreshadowing(
            pid, "另一高危", planted_chapter=3, expected_resolve_chapter=9, risk_level="high"
        )

        relevant = mgr.get_relevant_foreshadowings(pid, current_chapter=5)

        descriptions = [f["description"] for f in relevant]
        assert "已解决" not in descriptions  # resolved 伏笔不进入上下文
        assert descriptions[0] == "近章高危"  # high risk + 距回收章最近
        assert descriptions.index("近章高危") < descriptions.index("近章中危")
        assert len(mgr.get_relevant_foreshadowings(pid, 5, limit=1)) == 1

    def test_get_relevant_story_events_windowed_ascending(self, tmp_path):
        """Phase C: story_events 只取章节窗口内的行，升序返回，排除当前章。"""
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_story_events(pid, 1, [{"action": "第1章事件", "subject": "甲"}])
        mgr.save_story_events(pid, 9, [{"action": "第9章事件", "subject": "乙"}])
        mgr.save_story_events(pid, 10, [{"action": "当前章事件", "subject": "丙"}])

        events = mgr.get_relevant_story_events(pid, current_chapter=10, window=5)

        chapters = [e["chapter_number"] for e in events]
        assert 1 not in chapters  # 超出窗口的历史不加载
        assert 10 not in chapters  # 当前章（重写场景的旧事件）不加载
        assert chapters == [9]
        assert events[0]["action"] == "第9章事件"

    def test_v2_run_lock_and_immutable_versions(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        assert run["status"] == RunStatus.QUEUED.value

        try:
            mgr.create_writing_run(pid, 1)
        except ValueError as exc:
            assert "active run" in str(exc)
        else:
            raise AssertionError("concurrent chapter run was not rejected")

        version = mgr.create_chapter_version(pid, 1, "第一版正文", run_id=run["id"])
        assert version["version_number"] == 1
        assert version["content_hash"]
        mgr.update_writing_run(
            run["id"],
            status=RunStatus.WAITING_REVIEW.value,
            current_version_id=version["id"],
        )
        committed = mgr.commit_chapter_version(version["id"])
        assert committed["status"] == "approved"
        assert mgr.get_chapter(pid, 1)["approved_version_id"] == version["id"]
        assert mgr.get_writing_run(run["id"])["status"] == RunStatus.SUCCEEDED.value

        second = mgr.create_chapter_version(pid, 1, "第二版正文")
        assert second["version_number"] == 2
        assert second["parent_version_id"] == version["id"]
        assert mgr.get_chapter_version(version["id"])["content"] == "第一版正文"

    def test_canon_proposal_isolated_until_commit(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        proposal = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {"new_entities": [{"entity_type": "character", "name": "洛千秋"}]},
            run_id=run["id"],
        )
        assert mgr.get_all_world_entities(pid) == []
        accepted = mgr.review_canon_proposal(proposal["id"], "accepted", "确认")
        assert accepted["status"] == "accepted"

        committed = mgr.commit_canon_proposals(run["id"])
        assert committed[0]["status"] == "committed"
        assert mgr.get_all_world_entities(pid)[0]["name"] == "洛千秋"

    def test_rejected_canon_proposal_is_never_committed(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        proposal = mgr.create_canon_proposal(
            pid,
            1,
            "worldbuilding",
            {"new_entities": [{"entity_type": "item", "name": "禁物"}]},
            run_id=run["id"],
        )

        mgr.review_canon_proposal(proposal["id"], "rejected", "不符合设定")

        assert mgr.commit_canon_proposals(run["id"]) == []
        assert mgr.get_all_world_entities(pid) == []
        assert mgr.get_canon_proposal(proposal["id"])["status"] == "rejected"

    def test_canon_proposal_commit_rolls_back_as_one_transaction(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        for name in ("甲", "乙"):
            proposal = mgr.create_canon_proposal(
                pid,
                1,
                "worldbuilding",
                {"new_entities": [{"entity_type": "character", "name": name}]},
                run_id=run["id"],
            )
            mgr.review_canon_proposal(proposal["id"], "accepted")

        original = mgr._apply_worldbuilding_proposal_in_conn
        calls = 0

        def fail_after_first(conn, *args):
            nonlocal calls
            calls += 1
            original(conn, *args)
            if calls == 2:
                raise RuntimeError("canon failure")

        with patch.object(
            mgr, "_apply_worldbuilding_proposal_in_conn", side_effect=fail_after_first
        ):
            with pytest.raises(RuntimeError, match="canon failure"):
                mgr.commit_canon_proposals(run["id"])

        assert mgr.get_all_world_entities(pid) == []
        assert all(
            item["status"] == "accepted" for item in mgr.list_canon_proposals(pid, run_id=run["id"])
        )

    def test_story_events_are_normalized_and_idempotent(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        events = mgr.save_story_events(
            pid,
            3,
            [
                "主角进入城门",
                {"subject": "主角", "action": "发现", "object": "密道"},
            ],
        )
        again = mgr.save_story_events(
            pid,
            3,
            ["主角进入城门", {"subject": "主角", "action": "发现", "object": "密道"}],
        )

        assert len(events) == 2
        assert len(again) == 2
        assert mgr.get_story_events(pid, 3)[0]["chapter_number"] == 3

    def test_unapproved_v2_version_stays_candidate(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        version = mgr.create_chapter_version(pid, 1, "候选", run_id=run["id"])

        assert version["status"] == "candidate"
        assert mgr.get_chapter(pid, 1) is None
        assert mgr.get_writing_run(run["id"])["status"] == "queued"

    def test_scene_revision_rejects_missing_scene(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        version = mgr.create_chapter_version(
            pid,
            1,
            "正文",
            scene_plan=[{"scene_index": 1}],
            scene_drafts=["正文"],
        )

        with pytest.raises(ValueError, match="scene not found"):
            mgr.create_scene_revision(version["id"], 2, "不存在")

    def test_scene_revision_creates_child_version(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        run = mgr.create_writing_run(pid, 1)
        version = mgr.create_chapter_version(
            pid,
            1,
            "第一场\n\n第二场",
            run_id=run["id"],
            scene_plan=[
                {"scene_index": 1, "outline": "冲突", "target_words": 100},
                {"scene_index": 2, "outline": "转折", "target_words": 100},
            ],
            scene_drafts=["第一场", "第二场"],
        )

        revision = mgr.create_scene_revision(version["id"], 2, "改写后的第二场", run_id=run["id"])

        assert revision["parent_version_id"] == version["id"]
        assert revision["origin"] == "scene_rewrite"
        assert revision["content"] == "第一场\n\n改写后的第二场"
        assert mgr.get_scene_manifest(revision["id"])[1]["content"] == "改写后的第二场"

    def test_get_chapter_count_excludes_draft(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_chapter(pid, 1, status=ChapterStatus.DRAFT.value)
        mgr.save_chapter(pid, 2, status=ChapterStatus.APPROVED.value)
        assert mgr.get_chapter_count(pid) == 1  # 只数 approved

    def test_mark_chapter_failed_updates_chapter_and_outline(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_outline(pid, [{"chapter_number": 1, "title": "开篇"}])

        mgr.mark_chapter_failed(pid, 1)

        assert mgr.get_chapter(pid, 1)["status"] == "failed"
        assert mgr.get_outline(pid)[0]["status"] == "failed"
        assert mgr.get_chapter_count(pid) == 0


class TestForeshadowing:
    def test_lifecycle(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        fid = mgr.add_foreshadowing(
            pid,
            description="神秘信物",
            planted_chapter=1,
            risk_level="high",
            reader_knows=True,
            characters_aware=["主角"],
        )
        assert len(fid) == 8

        fs = mgr.get_foreshadowings(pid)
        assert len(fs) == 1
        assert fs[0]["status"] == "planted"
        assert fs[0]["reader_knows"] == 1

        ok = mgr.update_foreshadowing_status(
            pid,
            description="神秘信物",
            planted_chapter=1,
            status="resolved",
            resolved_chapter=3,
        )
        assert ok is True
        assert mgr.get_foreshadowings(pid)[0]["status"] == "resolved"

    def test_update_can_match_resolution_without_planted_chapter(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.add_foreshadowing(pid, "暗门", planted_chapter=2)
        assert mgr.update_foreshadowing_status(pid, "暗门", status="resolved", resolved_chapter=6)
        fs = mgr.get_foreshadowings(pid)[0]
        assert fs["status"] == "resolved"
        assert fs["resolved_chapter"] == 6


class TestWorldEntities:
    def test_save_and_dedup(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        report = {
            "new_entities": [
                {
                    "entity_type": "character",
                    "name": "林风",
                    "properties": {"age": 20},
                    "first_appearance_chapter": 1,
                },
            ]
        }
        assert mgr.save_world_entities(pid, report) == 1

        # 同名实体重复保存 → 更新而非新增
        report2 = {
            "new_entities": [
                {
                    "entity_type": "character",
                    "name": "林风",
                    "properties": {"age": 21},
                    "first_appearance_chapter": 1,
                },
            ]
        }
        mgr.save_world_entities(pid, report2)

        ents = mgr.get_all_world_entities(pid)
        assert len(ents) == 1
        assert '"age": 21' in ents[0]["properties"]

    def test_updates_merge_properties_and_preserve_first_appearance(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_world_entities(
            pid,
            {
                "new_entities": [
                    {"entity_type": "item", "name": "玉佩", "properties": {"颜色": "青"}}
                ],
            },
            chapter_number=2,
        )
        mgr.save_world_entities(
            pid,
            {
                "updated_entities": [
                    {"entity_type": "item", "name": "玉佩", "properties": {"主人": "林风"}}
                ],
            },
            chapter_number=8,
        )
        entity = mgr.get_all_world_entities(pid)[0]
        assert entity["first_appearance_chapter"] == 2
        assert '"颜色": "青"' in entity["properties"]
        assert '"主人": "林风"' in entity["properties"]


class TestOutline:
    def test_save_and_get(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_outline(
            pid,
            [
                {"chapter_number": 1, "title": "开篇", "summary": "s"},
                {"chapter_number": 2, "title": "发展"},
            ],
        )

        outline = mgr.get_outline(pid)
        assert len(outline) == 2
        assert outline[0]["title"] == "开篇"


class TestDelete:
    def test_delete_project_cascades(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_chapter(pid, 1, draft_content="正文")
        mgr.add_foreshadowing(pid, "伏笔", planted_chapter=1)
        mgr.save_world_entities(pid, {"new_entities": [{"entity_type": "character", "name": "A"}]})

        mgr.delete_project(pid)

        assert mgr.get_project(pid) is None
        assert mgr.get_all_chapters(pid) == []
        assert mgr.get_foreshadowings(pid) == []
        assert mgr.get_all_world_entities(pid) == []
