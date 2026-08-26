from unittest.mock import MagicMock

from novel_agent.services.context import ContextCompiler, estimate_tokens


def test_context_packet_single_contract():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "- 主角: 冷静",
        "world_context": "- 北墙: 黑曜石",
        "recent_summary": "第1章：主角出城",
    }
    manager.get_relevant_story_events.return_value = []
    manager.get_relevant_foreshadowings.return_value = [
        {"description": "神秘信物", "planted_chapter": 1, "status": "open"},
        {"description": "已解决", "planted_chapter": 1, "status": "resolved"},
    ]

    packet = ContextCompiler(manager, max_context_chars=1000).compile("p", 2)

    assert packet.unresolved_foreshadowings == ["[第1章] 神秘信物"]
    # to_state() 只产出 context_packet 单键 —— 不再有平铺字段 / hash / observability
    state = packet.to_state()
    assert set(state.keys()) == {"context_packet"}
    assert state["context_packet"]["character_context"] == "- 主角: 冷静"
    assert "packet_hash" not in state["context_packet"]
    assert "sources" not in state["context_packet"]
    assert "token_budget" not in state["context_packet"]


def test_compile_uses_task_aware_retrieval_not_full_reads():
    """Phase C: compile 走 relevance 查询，禁止全表读取 foreshadowings/story_events。"""
    manager = MagicMock()
    manager.build_context.return_value = {}
    manager.get_relevant_foreshadowings.return_value = []
    manager.get_relevant_story_events.return_value = []

    ContextCompiler(manager).compile("p", 5)

    manager.get_relevant_foreshadowings.assert_called_once_with("p", 5)
    manager.get_relevant_story_events.assert_called_once_with("p", 5)
    manager.get_foreshadowings.assert_not_called()
    manager.get_story_events.assert_not_called()


def test_context_packet_bounds_each_section():
    manager = MagicMock()
    manager.build_context.return_value = {
        "character_context": "角色" * 1000,
        "world_context": "设定" * 1000,
        "recent_summary": "摘要" * 1000,
    }
    manager.get_all_world_entities.return_value = []
    manager.get_relevant_foreshadowings.return_value = []
    manager.get_relevant_story_events.return_value = []

    packet = ContextCompiler(manager, max_context_chars=300).compile("p", 1)

    assert len(packet.character_context) <= 101
    assert len(packet.world_context) <= 101
    assert len(packet.recent_summary) <= 101


def test_context_compiler_can_compile_from_run_snapshot():
    manager = MagicMock()
    manager.get_writing_run.return_value = {
        "project_id": "p",
        "chapter_number": 2,
        "input_snapshot_id": "snap",
    }
    manager.get_canon_snapshot.return_value = {
        "payload": {
            "entities": [{"entity_type": "character", "name": "甲", "properties": "{}"}],
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


class TestTaskAwareProjections:
    """Phase 3: for_writer / for_editor / for_continuity / for_evolution."""

    def _big_packet(self) -> dict:
        return {
            "character_context": "角色" * 5000,
            "world_context": "设定" * 5000,
            "recent_summary": "摘要" * 5000,
            "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
            "timeline_events": [{"event": f"事件{i}"} for i in range(30)],
            "timeline_findings": [{"finding": f"发现{i}"} for i in range(10)],
        }

    def test_for_writer_world_context_is_empty_string(self):
        """Writer projection 显式返回 world_context='' 防止 Writer fallback。"""
        projected = ContextCompiler.for_writer(self._big_packet())
        assert "world_context" in projected
        assert projected["world_context"] == ""

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

    def test_for_writer_caps_character_context(self):
        projected = ContextCompiler.for_writer(self._big_packet(), budget_chars=3000)
        assert len(projected["character_context"]) <= 1001

    def test_for_writer_limits_foreshadowings_to_5(self):
        projected = ContextCompiler.for_writer(self._big_packet())
        assert len(projected["unresolved_foreshadowings"]) == 5

    def test_for_writer_limits_timeline_events_to_5(self):
        projected = ContextCompiler.for_writer(self._big_packet())
        assert len(projected["timeline_events"]) == 5

    def test_for_editor_only_keeps_3_fields(self):
        projected = ContextCompiler.for_editor(self._big_packet())
        assert "world_context" not in projected
        assert "timeline_events" not in projected
        assert len(projected["unresolved_foreshadowings"]) == 3

    def test_for_continuity_keeps_timeline_and_drops_world(self):
        projected = ContextCompiler.for_continuity(self._big_packet())
        assert "world_context" not in projected
        assert len(projected["timeline_events"]) == 10
        assert len(projected["unresolved_foreshadowings"]) == 8

    def test_for_evolution_only_keeps_metrics(self):
        projected = ContextCompiler.for_evolution(
            current_scores={"editor_overall": 80},
            previous_scores={"editor_overall": 70},
            delta={"trend": "improving"},
            guard_report={"violations": ["length_regression"]},
            improvement_plan={"primary_instruction": "改进节奏"},
        )
        assert "current_scores" in projected
        assert "guard_violations" in projected
        assert "character_context" not in projected

    def test_context_metrics_returns_chars_tokens_utilization(self):
        ctx = {"character_context": "角色" * 100}
        metrics = ContextCompiler.context_metrics(ctx, budget_tokens=3500)
        assert "context_chars" in metrics
        assert "estimated_tokens" in metrics
        assert "budget_tokens" in metrics
        assert "utilization" in metrics
        assert metrics["context_chars"] > 0
        assert metrics["estimated_tokens"] > 0

    def test_estimate_tokens_chinese(self):
        assert estimate_tokens("字" * 300) == 200  # 300/1.5

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_for_extension_only_keeps_chars_and_foreshadowings(self):
        """Extension projection 只返回 character_context + 3 条伏笔。"""
        projected = ContextCompiler.for_extension(self._big_packet())
        assert "character_context" in projected
        assert "unresolved_foreshadowings" in projected
        assert len(projected["unresolved_foreshadowings"]) == 3
        assert "world_context" not in projected
        assert "recent_summary" not in projected
        assert "timeline_events" not in projected

    def test_for_evolution_has_no_draft_or_world(self):
        """Evolution projection 不含正文、world_context、character_context。"""
        projected = ContextCompiler.for_evolution(
            current_scores={"editor_overall": 80},
            previous_scores={"editor_overall": 70},
            delta={"trend": "improving"},
            guard_report={"violations": ["length_target_unmet"]},
            improvement_plan={"primary_instruction": "改进"},
        )
        assert "current_scores" in projected
        assert "delta" in projected
        assert "guard_violations" in projected
        assert "character_context" not in projected
        assert "world_context" not in projected
        assert "draft_content" not in projected
