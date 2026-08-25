from unittest.mock import MagicMock

from novel_agent.services.context import ContextCompiler


def test_context_packet_has_sources_budget_and_stable_hash():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "- 主角: 冷静",
        "world_context": "- 北墙: 黑曜石",
        "recent_summary": "第1章：主角出城",
    }
    manager.get_all_world_entities.return_value = [{"name": "主角"}]
    manager.get_story_events.return_value = []
    manager.get_foreshadowings.return_value = [
        {"description": "神秘信物", "planted_chapter": 1, "status": "open"},
        {"description": "已解决", "planted_chapter": 1, "status": "resolved"},
    ]

    packet = ContextCompiler(manager, max_context_chars=1000).compile("p", 2)

    assert packet.packet_hash
    assert packet.unresolved_foreshadowings == ["[第1章] 神秘信物"]
    assert packet.sources[0]["count"] == 1
    assert packet.token_budget["max_context_chars"] == 1000
    assert (
        packet.token_budget["character_chars"]
        + packet.token_budget["world_chars"]
        + packet.token_budget["recent_chars"]
        <= 1000
    )


def test_context_packet_bounds_each_section():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "角色" * 1000,
        "world_context": "设定" * 1000,
        "recent_summary": "摘要" * 1000,
    }
    manager.get_all_world_entities.return_value = []
    manager.get_foreshadowings.return_value = []
    manager.get_story_events.return_value = []

    packet = ContextCompiler(manager, max_context_chars=300).compile("p", 1)

    assert len(packet.character_context) <= 101
    assert len(packet.world_context) <= 101
    assert len(packet.recent_summary) <= 101
    assert packet.to_state()["context_packet_hash"] == packet.packet_hash
    assert packet.to_state()["context_packet"]["packet_hash"] == packet.packet_hash


def test_context_compiler_can_compile_from_run_snapshot():
    manager = MagicMock()
    manager.get_writing_run.return_value = {
        "project_id": "p",
        "chapter_number": 2,
        "input_snapshot_id": "snap",
    }
    manager.get_canon_snapshot.return_value = {
        "payload": {
            "entities": [
                {"entity_type": "character", "name": "甲", "properties": "{}"}
            ],
            "foreshadowings": [],
            "story_events": [],
            "chapters": [{"chapter_number": 1, "draft_content": "已批准正文"}],
        }
    }
    manager.build_context_from_snapshot.return_value = {
        "character_context": "- 甲: {}",
        "world_context": "",
        "recent_summary": "第1章: 已批准正文",
    }

    packet = ContextCompiler(manager).compile_for_run("run")

    assert "已批准正文" in packet.recent_summary
    assert packet.character_context == "- 甲: {}"
    manager.build_context.assert_not_called()
