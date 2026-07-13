"""Tests for story length configuration constants."""

from novel_agent.config import (
    DEFAULT_CHAPTER_WORDS,
    DEFAULT_MAX_TOKENS,
    NARRATIVE_PACING,
    TYPICAL_CHAPTERS,
)


class TestLengthConstants:
    def test_default_chapter_words(self):
        assert DEFAULT_CHAPTER_WORDS == 3000

    def test_default_max_tokens(self):
        assert DEFAULT_MAX_TOKENS == 4096

    def test_narrative_pacing(self):
        assert "渐进展开" in NARRATIVE_PACING
        assert "climax" in NARRATIVE_PACING

    def test_typical_chapters(self):
        assert "100章" in TYPICAL_CHAPTERS
