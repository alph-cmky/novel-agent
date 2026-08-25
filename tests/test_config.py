"""Tests for story length configuration constants."""

from novel_agent.config import (
    DEFAULT_CHAPTER_WORDS,
    DEFAULT_MAX_TOKENS,
    NARRATIVE_PACING,
    TYPICAL_CHAPTERS,
    ExecutionProfile,
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


class TestExecutionProfile:
    def test_full_profile_runs_all_stages(self):
        profile = ExecutionProfile.from_state({})

        assert profile.should_review(1) is True
        assert profile.should_worldbuild() is True
        assert profile.should_enrich_evolution() is True

    def test_fast_profile_preserves_existing_skip_rules(self):
        profile = ExecutionProfile.from_state(
            {
                "skip_reviews": True,
                "review_interval": 2,
                "skip_worldbuilding": True,
                "skip_evolution_enrichment": True,
            }
        )

        assert profile.should_review(1) is False
        assert profile.should_review(2) is False
        assert profile.should_worldbuild() is False
        assert profile.should_enrich_evolution() is False

    def test_review_interval_applies_only_to_review_stage(self):
        profile = ExecutionProfile.from_state({"review_interval": 2})

        assert profile.should_review(1) is False
        assert profile.should_review(2) is True
        assert profile.should_review(4) is True
