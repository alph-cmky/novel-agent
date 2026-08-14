"""Tests for ProjectManager CRUD — SQLite 数据访问层。

锁住覆盖缺口：manager 是 novel-agent 的数据主干，但此前零测试。
用 tmp_path 隔离 SQLite，mock ChapterStore 避免真实 ChromaDB 落盘。
"""

from unittest.mock import MagicMock, patch

from novel_agent.schema.enums import ChapterStatus
from novel_agent.storage.manager import ProjectManager


def _make_manager(tmp_path, chapter_store=None) -> ProjectManager:
    with patch("novel_agent.storage.manager.ChapterStore") as mock_store:
        mock_store.return_value = chapter_store or MagicMock()
        return ProjectManager(tmp_path)


class TestProjectCRUD:
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

    def test_get_chapter_count_excludes_draft(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_chapter(pid, 1, status=ChapterStatus.DRAFT.value)
        mgr.save_chapter(pid, 2, status=ChapterStatus.APPROVED.value)
        assert mgr.get_chapter_count(pid) == 1  # 只数 approved


class TestForeshadowing:
    def test_lifecycle(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        fid = mgr.add_foreshadowing(
            pid, description="神秘信物", planted_chapter=1,
            risk_level="high", reader_knows=True, characters_aware=["主角"],
        )
        assert len(fid) == 8

        fs = mgr.get_foreshadowings(pid)
        assert len(fs) == 1
        assert fs[0]["status"] == "planted"
        assert fs[0]["reader_knows"] == 1

        ok = mgr.update_foreshadowing_status(
            pid, description="神秘信物", planted_chapter=1,
            status="resolved", resolved_chapter=3,
        )
        assert ok is True
        assert mgr.get_foreshadowings(pid)[0]["status"] == "resolved"


class TestWorldEntities:
    def test_save_and_dedup(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")

        report = {"new_entities": [
            {"entity_type": "character", "name": "林风",
             "properties": {"age": 20}, "first_appearance_chapter": 1},
        ]}
        assert mgr.save_world_entities(pid, report) == 1

        # 同名实体重复保存 → 更新而非新增
        report2 = {"new_entities": [
            {"entity_type": "character", "name": "林风",
             "properties": {"age": 21}, "first_appearance_chapter": 1},
        ]}
        mgr.save_world_entities(pid, report2)

        ents = mgr.get_all_world_entities(pid)
        assert len(ents) == 1
        assert '"age": 21' in ents[0]["properties"]


class TestOutline:
    def test_save_and_get(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_outline(pid, [
            {"chapter_number": 1, "title": "开篇", "summary": "s"},
            {"chapter_number": 2, "title": "发展"},
        ])

        outline = mgr.get_outline(pid)
        assert len(outline) == 2
        assert outline[0]["title"] == "开篇"


class TestDelete:
    def test_delete_project_cascades(self, tmp_path):
        mgr = _make_manager(tmp_path)
        pid = mgr.init_project(name="p")
        mgr.save_chapter(pid, 1, draft_content="正文")
        mgr.add_foreshadowing(pid, "伏笔", planted_chapter=1)
        mgr.save_world_entities(pid, {"new_entities": [
            {"entity_type": "character", "name": "A"}]})

        mgr.delete_project(pid)

        assert mgr.get_project(pid) is None
        assert mgr.get_all_chapters(pid) == []
        assert mgr.get_foreshadowings(pid) == []
        assert mgr.get_all_world_entities(pid) == []
