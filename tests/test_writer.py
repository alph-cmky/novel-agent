"""Tests for WriterAgent strategy formatting — nested None safety."""

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
