"""Tests for WriterAgent strategy formatting — nested None safety."""

import asyncio
from unittest.mock import AsyncMock, patch

from novel_agent.agents.writer import WriterAgent
from novel_agent.schema.parser import strip_none


class TestStripNone:
    def test_removes_none_values_recursively(self):
        data = {
            "keep": 1,
            "drop": None,
            "nested": {"a": None, "b": 2},
            "items": [None, {"c": 3}, None],
        }
        assert strip_none(data) == {
            "keep": 1,
            "nested": {"b": 2},
            "items": [{"c": 3}],
        }

    def test_returns_scalars_unchanged(self):
        assert strip_none("x") == "x"
        assert strip_none(1) == 1
        assert strip_none(None) is None


class TestFormatStrategy:
    def test_nested_none_does_not_crash(self):
        """``tension_profile.variety_check: None`` used to crash with NoneType.get."""
        strategy = {
            "narrative_stage": "development",
            "chapter_strategy": {
                "tension_profile": {
                    "chapter_tension": 7,
                    "overall_trend": "rising",
                    "emotional_tone": "紧张",
                    "variety_check": None,
                },
                "pacing": "正常",
            },
        }
        text = WriterAgent()._format_strategy(strategy)
        # Tension profile is still rendered, but the None variety_check is skipped.
        assert "本章紧张度：7/10" in text
        assert "节奏提醒" not in text

    def test_none_chapter_strategy_is_empty(self):
        """chapter_strategy: None should degrade to an empty section."""
        assert WriterAgent()._format_strategy({"chapter_strategy": None}) == ""


class _FakeTool:
    name = "search_context"


class TestWriteToolHint:
    """write() 的工具提示只应在 search_context 工具真正注册时出现。

    无工具（project_id 为空）时提示「使用 search_context 工具」会让模型输出
    <search_context> 文本标签并提前终止，产出空正文（横评 bug）。
    """

    @staticmethod
    def _capture_user_prompt(writer: WriterAgent) -> str:
        with patch.object(
            writer, "run_with_tools", new=AsyncMock(return_value=("正文", None))
        ) as mocked:
            asyncio.run(writer.write(chapter_number=1, outline="大纲"))
        return mocked.call_args.args[0][1]["content"]

    def test_no_tool_uses_direct_output_hint(self):
        user = self._capture_user_prompt(WriterAgent())
        assert "search_context" not in user
        assert "直接输出章节正文" in user

    def test_tool_registered_uses_search_hint(self):
        writer = WriterAgent()
        writer.register_tool(_FakeTool())
        user = self._capture_user_prompt(writer)
        assert "search_context" in user
        assert "直接输出章节正文" not in user


class TestImprovementPlanFormatting:
    """测试演化迭代改进计划格式化，确保篇幅保护和结构硬约束不被丢弃。"""

    def test_improvement_plan_includes_length_and_structure_protection(self):
        from novel_agent.graph.chapter import _format_improvement_plan

        plan = {
            "focus_dimensions": ["dialogue", "ai_flavor"],
            "primary_instruction": "增强对白冲突，剔除套话",
            "secondary_instructions": ["增加环境声", "加快节奏"],
            "constraints": {
                "preserve": ["核心情节走向"],
                "avoid": ["大段哲学议论"],
            },
        }

        formatted = _format_improvement_plan(plan, version=1)

        # 验证重点维度与核心指令
        assert "第 2 次迭代" in formatted
        assert "对话" in formatted
        assert "AI味" in formatted
        assert "增强对白冲突，剔除套话" in formatted

        # 核心：必须包含全篇完整性与篇幅保护硬约束
        assert "篇幅与结构硬性要求" in formatted
        assert "必须输出完整的全章节正文" in formatted
        assert "绝对禁止只输出修改片段" in formatted
        assert "字数必须达标" in formatted

    def test_empty_plan_returns_empty_string(self):
        from novel_agent.graph.chapter import _format_improvement_plan
        assert _format_improvement_plan(None, 0) == ""
        assert _format_improvement_plan({}, 0) == ""
