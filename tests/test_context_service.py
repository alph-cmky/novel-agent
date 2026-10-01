from unittest.mock import MagicMock

from novel_agent.services.context import ContextCompiler, _compress_events, closing_repeats
from novel_agent.services.continuity import ContinuityService
from novel_agent.storage.manager import _excerpt, _excerpt_before_closing


def test_compress_events_keeps_reappearance_outside_the_window():
    events = [
        {"chapter_number": 1, "subject": "甲", "action": "甲死亡"},
        {"chapter_number": 2, "subject": "乙", "action": "赶路"},
        {"chapter_number": 5, "subject": "甲", "action": "甲又出现了"},
        {"chapter_number": 12, "subject": "乙", "action": "到达"},
        {"chapter_number": 15, "subject": "甲", "action": "当前章旧事件"},
    ]
    kept = _compress_events(events, chapter_number=15)
    actions = [event["action"] for event in kept]
    assert actions == ["甲死亡", "甲又出现了", "到达"]
    findings = ContinuityService.check_timeline(kept, [], current_chapter=15)["findings"]
    assert any(item["type"] == "dead_character_reappeared" for item in findings)


def test_discussing_death_does_not_keep_that_characters_history():
    events = [
        {"chapter_number": 1, "subject": "甲", "action": "讨论死亡"},
        {"chapter_number": 3, "subject": "甲", "action": "赶路"},
        {"chapter_number": 12, "subject": "乙", "action": "到达"},
    ]
    kept = _compress_events(events, chapter_number=15)
    assert [event["action"] for event in kept] == ["到达"]
    findings = ContinuityService.check_timeline(events, [], current_chapter=15)["findings"]
    assert not any(item["type"] == "dead_character_reappeared" for item in findings)


def test_excerpt_starts_after_a_sentence_boundary():
    text = "甲" * 100 + "。乙" * 200
    excerpt = _excerpt(text, 400)
    assert excerpt.startswith("…")
    assert excerpt[1] == "乙"
    assert len(excerpt) <= 401


def test_recent_excerpt_stops_before_the_closing_window():
    story = "苏迟核对名单，在表格末行写下待查。" * 80
    closing = "她把手机翻过去，屏幕朝下扣在桌上。窗外电线又往下滴了一滴水。"
    excerpt = _excerpt_before_closing(story + closing, 400)
    assert "翻过去" not in excerpt
    assert "待查" in excerpt


def test_closing_repeats_flags_the_same_pose_only():
    pose = "声音从半截卷闸门下面拐进来，落在折叠桌上。窗外电线又往下滴了一滴水。"
    previous = ("前文情节。" * 40) + pose
    same = ("这一章的新事。" * 40) + pose + "她没有抬头。"
    fresh = ("这一章的新事。" * 40) + "她走出巷口，洒水车从面前开过去，鸽子从屋檐上飞起来。"
    assert closing_repeats(previous, same)
    assert not closing_repeats(previous, fresh)


def test_context_packet_single_contract():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "- 主角: 冷静",
        "world_context": "- 北墙: 黑曜石",
        "recent_summary": "第1章：主角出城",
    }
    manager.get_context_story_events.return_value = []
    manager.get_relevant_foreshadowings.return_value = [
        {"description": "神秘信物", "planted_chapter": 1, "status": "open"},
        {"description": "已解决", "planted_chapter": 1, "status": "resolved"},
    ]

    packet = ContextCompiler(manager).compile("p", 2)

    assert packet.unresolved_foreshadowings == ["[第1章] 神秘信物"]
    # to_state() 只产出 context_packet 单键 —— 不再有平铺字段 / hash / observability
    state = packet.to_state()
    assert set(state.keys()) == {"context_packet"}
    assert state["context_packet"]["character_context"] == "- 主角: 冷静"
    assert "packet_hash" not in state["context_packet"]
    assert "sources" not in state["context_packet"]
    assert "token_budget" not in state["context_packet"]


def test_compile_uses_task_aware_retrieval_not_full_reads():
    """未回收伏笔先全部取出，再在 Python 里留下早期线和近期相关线。"""
    manager = MagicMock()
    manager.build_context.return_value = {}
    manager.get_relevant_foreshadowings.return_value = []
    manager.get_context_story_events.return_value = []

    ContextCompiler(manager).compile("p", 5)

    manager.get_relevant_foreshadowings.assert_called_once_with("p", 5, limit=None)
    manager.get_context_story_events.assert_called_once_with("p", 5, window=8)
    manager.get_relevant_story_events.assert_not_called()
    manager.get_foreshadowings.assert_not_called()
    manager.get_story_events.assert_not_called()


def test_compile_caps_entity_counts_and_keeps_recent_excerpt():
    """compile 限制人物和世界条目数量，前文仍只留结尾。"""
    manager = MagicMock()
    manager.build_context.return_value = {}
    manager.get_relevant_foreshadowings.return_value = []
    manager.get_context_story_events.return_value = []

    ContextCompiler(manager).compile("p", 5, task="orchestrator")

    manager.build_context.assert_called_once_with(
        "p",
        5,
        max_recent_chapters=3,
        max_entities=24,
        max_characters=16,
        excerpt_chars=400,
        property_chars=160,
        mention_text="",
    )


def test_prefer_foreshadowings_keeps_early_spine_and_recent_overlap():
    from novel_agent.services.context import _prefer_foreshadowings

    rows = [
        {
            "description": "别接龙王的单，别问第十八任去哪了",
            "planted_chapter": 1,
            "status": "open",
        }
    ]
    rows += [
        {
            "description": f"灰尘方块痕迹编号{index:02d}",
            "planted_chapter": index,
            "status": "open",
        }
        for index in range(2, 25)
    ]
    rows.append(
        {
            "description": "照壁胶痕与离职报告的折法相同",
            "planted_chapter": 30,
            "status": "open",
        }
    )
    events = [{"action": "苏迟发现照壁胶痕和离职报告折法相同"}]

    chosen = _prefer_foreshadowings(rows, 10, events)
    descriptions = [row["description"] for row in chosen]

    assert len(chosen) == 10
    assert descriptions[0].startswith("别接龙王的单")
    assert any("照壁胶痕" in item for item in descriptions)


def test_compile_snapshot_keeps_full_canon():
    """快照路径保留实体，伏笔截到上限，事件只留窗口。"""
    manager = MagicMock()
    manager.get_canon_snapshot.return_value = {
        "payload": {
            "entities": [
                {"entity_type": "character", "name": f"角色{i}", "properties": "{}"}
                for i in range(30)
            ],
            "foreshadowings": [
                {"status": "open", "description": f"伏笔{i}", "planted_chapter": i}
                for i in range(50)
            ],
            "story_events": [{"chapter_number": i, "action": f"事件{i}"} for i in range(100)],
            "chapters": [],
        }
    }
    manager.build_context_from_snapshot.return_value = {
        "character_context": "",
        "world_context": "",
        "recent_summary": "",
    }

    packet = ContextCompiler(manager).compile("p", 50, task="orchestrator", snapshot_id="snapshot")

    kwargs = manager.build_context_from_snapshot.call_args.kwargs
    assert kwargs["max_entities"] == 24
    assert kwargs["max_characters"] == 16
    assert "事件42" in kwargs["mention_text"]
    assert "伏笔0" in kwargs["mention_text"]
    assert len(packet.timeline_events) == 8
    assert packet.timeline_events[0]["chapter_number"] == 42
    assert packet.timeline_events[-1]["chapter_number"] == 49
    assert len(packet.unresolved_foreshadowings) == 40


def test_for_orchestrator_passes_compiled_packet_through():
    """角色投影不再二次切额度，compile 已有的字段原样交给 Orchestrator。"""
    packet = {
        "character_context": "甲: 主角",
        "world_context": "设定" * 4000,
        "recent_summary": "前情",
        "unresolved_foreshadowings": [f"伏笔{i}" for i in range(30)],
        "timeline_events": [{"event": f"事件{i}"} for i in range(30)],
        "timeline_findings": [f"警告{i}" for i in range(30)],
    }

    projected = ContextCompiler.for_orchestrator(packet)

    assert projected["world_context"] == packet["world_context"]
    assert projected["character_context"] == packet["character_context"]
    assert projected["recent_summary"] == "前情"
    assert projected["unresolved_foreshadowings"] == packet["unresolved_foreshadowings"]
    assert projected["timeline_events"] == packet["timeline_events"]
    assert projected["timeline_findings"] == packet["timeline_findings"]
    assert projected["timeline_events"][-1] == {"event": "事件29"}


def test_apply_context_needed_folds_orchestrator_demands_into_packet():
    """context_needed triggers real SQL retrieval, not text hints."""
    manager = MagicMock()
    # Characters: query returns matching entities
    manager.get_entities_by_names.side_effect = lambda pid, names, **kw: (
        [{"name": "甲", "entity_type": "character", "properties": "主角，冷静"}]
        if kw.get("entity_type") == "character"
        else [{"name": "北墙秘道", "entity_type": "location", "properties": "隐藏通道"}]
    )
    # Cross-timeline events
    manager.get_story_events_by_subjects.return_value = [
        {"id": "e1", "subject": "甲", "action": "坠崖", "chapter_number": 3}
    ]

    packet = {
        "character_context": "- 旧角色: 过时信息",
        "world_context": "- 旧设定: 过时",
        "recent_summary": "第1章：甲出城",
        "timeline_events": [],
    }
    compiler = ContextCompiler(manager)
    enriched = compiler.apply_context_needed(
        packet,
        {
            "characters": ["甲", "乙"],
            "world_elements": ["北墙秘道"],
            "perspective_specific": "乙不知道甲的身份",
            "cross_timeline_references": ["甲"],
            "recent_reference": "甲与乙的约定",
        },
        project_id="p",
        chapter_number=5,
    )

    assert "- 甲: 主角，冷静" in enriched["character_context"]
    assert "旧角色" in enriched["character_context"]
    assert "[视角特定信息: 乙不知道甲的身份]" in enriched["character_context"]

    assert "[location] 北墙秘道: 隐藏通道" in enriched["world_context"]
    assert "旧设定" in enriched["world_context"]

    # Cross-timeline events are merged into timeline_events
    assert any(e.get("action") == "坠崖" for e in enriched["timeline_events"])

    # POV metadata and recent_reference remain as text annotations
    assert "[本章需要回顾 — 甲与乙的约定]" in enriched["recent_summary"]

    # 原 packet 不被原地修改
    assert "旧角色" in packet["character_context"]


def test_apply_context_needed_clips_properties_and_keeps_deaths():
    manager = MagicMock()
    manager.get_entities_by_names.return_value = [
        {"name": "甲", "entity_type": "character", "properties": "甲" * 400}
    ]
    manager.get_story_events_by_subjects.return_value = [
        {"id": "death", "subject": "甲", "action": "甲死亡", "chapter_number": 1},
        {"id": "errand", "subject": "丙", "action": "买菜", "chapter_number": 2},
    ]
    packet = {
        "timeline_events": [
            {"id": f"e{n}", "subject": "乙", "action": "赶路", "chapter_number": n}
            for n in range(20, 50)
        ]
    }
    enriched = ContextCompiler(manager).apply_context_needed(
        packet,
        {"characters": ["甲"], "cross_timeline_references": ["甲"]},
        project_id="p",
        chapter_number=50,
    )
    assert enriched["character_context"].startswith("- 甲: 甲")
    assert "甲" * 400 not in enriched["character_context"]
    assert len(enriched["character_context"]) <= len("- 甲: ") + 160
    actions = [event["action"] for event in enriched["timeline_events"]]
    assert "甲死亡" in actions
    assert "买菜" not in actions
    assert "赶路" in actions


def test_apply_context_needed_empty_declaration_is_noop():
    """空声明（无消费者需求）→ packet 原样返回，不注入占位噪声。"""
    manager = MagicMock()
    packet = {"character_context": "ctx", "recent_summary": "sum"}
    compiler = ContextCompiler(manager)
    enriched = compiler.apply_context_needed(packet, {}, project_id="p", chapter_number=1)

    assert enriched == packet
    assert "暂无" not in enriched["character_context"]
    manager.get_entities_by_names.assert_not_called()
    manager.get_story_events_by_subjects.assert_not_called()


class TestContextNeededRetrieval:
    """Phase 2 (P0-2): context_needed triggers real SQL retrieval, not text hints."""

    def test_characters_query_replaces_not_appends(self):
        """context_needed.characters triggers get_entities_by_names, replacing packet content."""
        manager = MagicMock()
        manager.get_entities_by_names.return_value = [
            {"name": "甲", "entity_type": "character", "properties": "主角"},
        ]
        packet = {"character_context": "- 旧: 过时信息"}
        compiler = ContextCompiler(manager)
        result = compiler.apply_context_needed(
            packet, {"characters": ["甲"]}, project_id="p", chapter_number=1
        )
        assert "旧" in result["character_context"]
        assert "甲: 主角" in result["character_context"]
        manager.get_entities_by_names.assert_called_once_with("p", ["甲"], entity_type="character")

    def test_world_elements_query_replaces_not_appends(self):
        """context_needed.world_elements triggers get_entities_by_names for non-characters."""
        manager = MagicMock()
        manager.get_entities_by_names.return_value = [
            {"name": "北墙", "entity_type": "location", "properties": "黑曜石城墙"},
        ]
        packet = {"world_context": "- 旧设定: 过时"}
        compiler = ContextCompiler(manager)
        result = compiler.apply_context_needed(
            packet, {"world_elements": ["北墙"]}, project_id="p", chapter_number=1
        )
        assert "旧设定" in result["world_context"]
        assert "[location] 北墙: 黑曜石城墙" in result["world_context"]

    def test_cross_timeline_merges_into_timeline_events(self):
        """context_needed.cross_timeline_references triggers get_story_events_by_subjects."""
        manager = MagicMock()
        manager.get_story_events_by_subjects.return_value = [
            {"id": "e1", "subject": "甲", "action": "坠崖", "chapter_number": 3},
        ]
        packet = {"timeline_events": [{"id": "e0", "action": "出城"}]}
        compiler = ContextCompiler(manager)
        result = compiler.apply_context_needed(
            packet,
            {"cross_timeline_references": ["甲"]},
            project_id="p",
            chapter_number=10,
        )
        assert len(result["timeline_events"]) == 2
        manager.get_story_events_by_subjects.assert_called_once_with("p", ["甲"])

    def test_perspective_specific_remains_text_annotation(self):
        """perspective_specific is POV metadata, not entity retrieval."""
        manager = MagicMock()
        packet = {"character_context": "- 甲: 主角"}
        compiler = ContextCompiler(manager)
        result = compiler.apply_context_needed(
            packet,
            {"perspective_specific": "乙不知甲的身份"},
            project_id="p",
            chapter_number=1,
        )
        assert "[视角特定信息: 乙不知甲的身份]" in result["character_context"]
        manager.get_entities_by_names.assert_not_called()

    def test_no_retrieval_without_manager(self):
        """When mgr is available but context_needed is empty, no SQL calls."""
        manager = MagicMock()
        packet = {"character_context": "ctx", "world_context": "world"}
        compiler = ContextCompiler(manager)
        result = compiler.apply_context_needed(
            packet, {"characters": [], "world_elements": []}, project_id="p", chapter_number=1
        )
        assert result == packet
        manager.get_entities_by_names.assert_not_called()
        manager.get_story_events_by_subjects.assert_not_called()


def test_context_packet_keeps_full_sections():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "角色" * 1000,
        "world_context": "设定" * 1000,
        "recent_summary": "摘要" * 1000,
    }
    manager.get_all_world_entities.return_value = []
    manager.get_relevant_foreshadowings.return_value = []
    manager.get_context_story_events.return_value = []

    packet = ContextCompiler(manager).compile("p", 1)

    assert packet.character_context == "角色" * 1000
    assert packet.world_context == "设定" * 1000
    assert packet.recent_summary == "摘要" * 1000


class TestTaskAwareProjections:
    """角色投影原样传递 compile 后的包，不再按角色切额度。"""

    def _big_packet(self) -> dict:
        return {
            "character_context": "角色" * 5000,
            "world_context": "设定" * 5000,
            "recent_summary": "摘要" * 5000,
            "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
            "timeline_events": [{"event": f"事件{i}"} for i in range(30)],
            "timeline_findings": [{"finding": f"发现{i}"} for i in range(10)],
        }

    def test_for_writer_keeps_world_context(self):
        projected = ContextCompiler.for_writer(self._big_packet())
        assert projected["world_context"] == self._big_packet()["world_context"]

    def test_for_writer_all_keys_present(self):
        """所有 6 个 key 都存在，Writer.get() 不回退到旧 State。"""
        projected = ContextCompiler.for_writer(self._big_packet())
        for key in (
            "character_context",
            "world_context",
            "recent_summary",
            "unresolved_foreshadowings",
            "timeline_events",
            "timeline_findings",
        ):
            assert key in projected, f"missing key: {key}"

    def test_projections_do_not_truncate(self):
        packet = self._big_packet()
        for project in (
            ContextCompiler.for_writer,
            ContextCompiler.for_editor,
            ContextCompiler.for_continuity,
            ContextCompiler.for_orchestrator,
        ):
            projected = project(packet)
            assert projected["character_context"] == packet["character_context"]
            assert projected["world_context"] == packet["world_context"]
            assert projected["recent_summary"] == packet["recent_summary"]
            assert projected["unresolved_foreshadowings"] == packet["unresolved_foreshadowings"]
            assert projected["timeline_events"] == packet["timeline_events"]
            assert projected["timeline_findings"] == packet["timeline_findings"]

    def test_missing_fields_stay_empty(self):
        projected = ContextCompiler.for_writer({})
        assert projected["world_context"] == ""
        assert projected["character_context"] == ""
        assert projected["unresolved_foreshadowings"] == []
        assert projected["timeline_events"] == []
