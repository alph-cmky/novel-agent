"""Tests for story length configuration."""

import pytest

from novel_agent.config import (
    StoryLength,
    get_length_config,
    get_length_defaults,
)


class TestStoryLength:
    def test_enum_values(self):
        assert StoryLength.SHORT.value == "short"
        assert StoryLength.NOVELLA.value == "novella"
        assert StoryLength.LONG.value == "long"

    def test_from_string(self):
        assert StoryLength("short") == StoryLength.SHORT
        assert StoryLength("novella") == StoryLength.NOVELLA
        assert StoryLength("long") == StoryLength.LONG

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            StoryLength("unknown")


class TestGetLengthConfig:
    def test_short_config(self):
        cfg = get_length_config(StoryLength.SHORT)
        assert cfg.default_chapter_words == 1500
        assert cfg.max_tokens == 2048
        assert "快速推进" in cfg.narrative_pacing

    def test_novella_config(self):
        cfg = get_length_config(StoryLength.NOVELLA)
        assert cfg.default_chapter_words == 3000
        assert cfg.max_tokens == 4096

    def test_long_config(self):
        cfg = get_length_config(StoryLength.LONG)
        assert cfg.default_chapter_words == 3000
        assert "渐进展开" in cfg.narrative_pacing

    def test_from_string(self):
        cfg = get_length_config("short")
        assert cfg.default_chapter_words == 1500


class TestGetLengthDefaults:
    def test_returns_dict(self):
        d = get_length_defaults(StoryLength.NOVELLA)
        assert d["story_length"] == "novella"
        assert d["default_chapter_words"] == 3000
        assert d["max_tokens"] == 4096

    def test_from_string(self):
        d = get_length_defaults("long")
        assert d["story_length"] == "long"
