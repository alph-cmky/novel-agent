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


def test_writer_prompt_prioritizes_canon_over_generic_style_rules():
    prompt = WriterAgent().system_prompt

    assert "Canon / 已批准事实" in prompt
    assert "不强制每章反转或 cliffhanger" in prompt
    assert "对话比例服从场景目标" in prompt


class TestWriterPrompt:
    """Verify the single writer prompt has required content."""

    def test_prompt_has_canon_priority(self):
        prompt = WriterAgent().system_prompt
        assert "长篇小说章节执行器" in prompt
        assert "Canon / 已批准事实" in prompt

    def test_prompt_has_paragraph_principles(self):
        prompt = WriterAgent().system_prompt
        assert "自然段以叙事单元而非单句为边界" in prompt
        assert "短句不等于短段" in prompt
        assert "对白自然独立成段" in prompt


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
    def _capture_user_prompt(writer: WriterAgent, **kwargs) -> str:
        with patch.object(
            writer, "run_with_tools", new=AsyncMock(return_value=("正文", None))
        ) as mocked:
            asyncio.run(writer.write(chapter_number=1, outline="大纲", **kwargs))
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

    def test_unresolved_foreshadowings_are_injected(self):
        user = self._capture_user_prompt(
            WriterAgent(),
            context_packet={
                "unresolved_foreshadowings": ["[第1章] 神秘信物"],
            },
        )
        assert "待回收伏笔" in user
        assert "神秘信物" in user

    def test_context_packet_character_is_injected(self):
        """context_packet 中的 character_context 出现在 user prompt 中。"""
        user = self._capture_user_prompt(
            WriterAgent(),
            context_packet={
                "character_context": "Packet 角色",
                "world_context": "Packet 设定",
                "recent_summary": "Packet 摘要",
                "unresolved_foreshadowings": [],
            },
        )
        assert "Packet 角色" in user
        assert "Packet 设定" in user
        assert "Packet 摘要" in user

    def test_context_packet_world_context_empty_does_not_inject(self):
        """context_packet 的 world_context='' 时，Writer 不输出世界观段落。"""
        user = self._capture_user_prompt(
            WriterAgent(),
            context_packet={
                "character_context": "角色",
                "world_context": "",
                "recent_summary": "",
                "unresolved_foreshadowings": [],
            },
        )
        assert "世界观设定" not in user


class TestNarrativeExtension:
    """Phase 1: Narrative Extension — 增量续写，不是 compensation 全文重写。"""

    @staticmethod
    def _capture_extension_prompt(**kwargs) -> str:
        writer = WriterAgent()
        with patch.object(
            writer, "run_with_tools", new=AsyncMock(return_value=("续写正文", None))
        ) as mocked:
            asyncio.run(writer.narrative_extension(**kwargs))
        return mocked.call_args.args[0][1]["content"]

    def test_extension_prompt_uses_ending_not_full_content(self):
        """Extension context 只含结尾 ~800 字，不含全文。"""
        full = "起" * 5000 + "结尾段落"
        user = self._capture_extension_prompt(
            current_content=full,
            chapter_number=1,
            chapter_outline="大纲",
            gap_words=500,
        )
        assert "结尾段落" in user
        assert "起" * 5000 not in user

    def test_extension_prompt_says_continue_not_expand(self):
        """Extension 指令是续写向前，不是充分展开已有内容。"""
        user = self._capture_extension_prompt(
            current_content="已有正文",
            chapter_number=1,
            chapter_outline="大纲",
            gap_words=500,
        )
        assert "续写" in user
        assert "不要回头重写" in user
        assert "充分展开" not in user

    def test_extension_returns_only_continuation(self):
        """返回值是续写文本，不是 (content, trace) 元组。"""
        writer = WriterAgent()
        with patch.object(writer, "run_with_tools", new=AsyncMock(return_value=("续写正文", None))):
            result = asyncio.run(
                writer.narrative_extension(
                    current_content="已有正文",
                    chapter_number=1,
                    chapter_outline="大纲",
                    gap_words=500,
                )
            )
        assert result == "续写正文"


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
