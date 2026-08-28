# tests/test_storage_manager_canon_counts.py
"""C-3 canon growth counters：分表正确列名 + chapter 上界过滤。"""
from novel_agent.storage.manager import ProjectManager


def test_get_canon_counts_after_chapter(tmp_path):
    mgr = ProjectManager(str(tmp_path))
    pid = mgr.init_project(name="cc", title="cc", story_length="long", target_chapter_words=3000)

    # ch1: 1 角色 + 1 事件 + 1 伏笔
    mgr.save_world_entities(
        pid,
        {"new_entities": [
            {"entity_type": "character", "name": "林远", "properties": {"身份": "弟子"}}
        ]},
        chapter_number=1,
    )
    mgr.save_story_events(pid, 1, [{"action": "捡剑"}])
    mgr.add_foreshadowing(pid, "血契三日", planted_chapter=1)

    c1 = mgr.get_canon_counts(pid, after_chapter=1)
    assert c1["world_entities"] == 1
    assert c1["story_events"] == 1
    assert c1["foreshadowings_open"] == 1
    assert c1["chapters"] == 0  # 尚无 committed chapter

    # ch2 再加一个角色；after_chapter=1 不应计入
    mgr.save_world_entities(
        pid,
        {"new_entities": [{"entity_type": "character", "name": "苏晚晴", "properties": {}}]},
        chapter_number=2,
    )
    c2 = mgr.get_canon_counts(pid, after_chapter=2)
    assert c2["world_entities"] == 2

    # 不带上界：全量
    call = mgr.get_canon_counts(pid)
    assert call["world_entities"] == 2
    assert call["foreshadowings_open"] == 1
