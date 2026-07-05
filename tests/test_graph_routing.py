"""Tests for LangGraph conditional routing logic and feedback loop."""

from novel_agent.graph.chapter import (
    CONTINUITY_PASS_SCORE,
    MAX_RETRIES,
    route_after_continuity,
    route_after_human,
)


def _state(continuity_score=0, criticals=None, retry=0,
           editor_score=0, wb_conflicts=0, approved=False,
           human_feedback=None):
    """Build a minimal NovelState dict for routing tests."""
    return {
        "continuity_report": {
            "overall_score": continuity_score,
            "inconsistencies": [
                {"severity": s} for s in (criticals or [])
            ],
        },
        "retry_count": retry,
        "editor_report": {"overall_score": editor_score},
        "worldbuilding_report": {
            "conflicts": [{}] * wb_conflicts,
        },
        "human_approved": approved,
        "human_feedback": human_feedback or {},
        "rewrite_instructions": "",
        "draft_content": "",
        "chapter_number": 1,
        "project_id": "test",
        "chapter_outline": "",
        "character_context": "",
        "world_context": "",
        "recent_summary": "",
        "unresolved_foreshadowings": [],
        "orchestrator_strategy": {},
        "existing_world_entities": [],
        "story_length": "long",
        "target_chapter_words": 3000,
        "persist_dir": "/tmp",
        "trace_id": "",
    }


class TestRouteAfterContinuity:
    def test_pass_to_worldbuilding(self):
        """High scores, no criticals → worldbuilding → human review."""
        result = route_after_continuity(
            _state(continuity_score=CONTINUITY_PASS_SCORE, editor_score=60)
        )
        assert result == "worldbuilding"

    def test_pass_above_threshold(self):
        result = route_after_continuity(
            _state(continuity_score=95, editor_score=60)
        )
        assert result == "worldbuilding"

    def test_low_editor_with_retries_goes_to_orchestrator_review(self):
        """Continuity passes but Editor below threshold → feedback loop."""
        result = route_after_continuity(
            _state(continuity_score=90, editor_score=20, retry=0)
        )
        assert result == "orchestrator_review"

    def test_critical_with_retries_left_goes_to_orchestrator_review(self):
        """Critical issues + retries remaining → orchestrator_review (feedback loop)."""
        result = route_after_continuity(
            _state(
                continuity_score=50,
                criticals=["critical"],
                retry=0,
            )
        )
        assert result == "orchestrator_review"

    def test_low_score_with_retries_left_goes_to_orchestrator_review(self):
        """Below threshold with retries → feedback loop."""
        result = route_after_continuity(
            _state(continuity_score=50, retry=1)
        )
        assert result == "orchestrator_review"

    def test_critical_no_retries_left_goes_to_worldbuilding(self):
        """Critical issues but no retries left → worldbuilding (human decides)."""
        result = route_after_continuity(
            _state(
                continuity_score=50,
                criticals=["critical"],
                retry=MAX_RETRIES,
            )
        )
        assert result == "worldbuilding"

    def test_low_score_no_retries_left_goes_to_worldbuilding(self):
        """Below threshold, no retries → worldbuilding → human review."""
        result = route_after_continuity(
            _state(continuity_score=50, retry=MAX_RETRIES)
        )
        assert result == "worldbuilding"

    def test_exactly_at_threshold_with_criticals(self):
        """Score at threshold but has criticals → feedback loop if retries left."""
        result = route_after_continuity(
            _state(
                continuity_score=CONTINUITY_PASS_SCORE,
                criticals=["critical"],
                retry=0,
            )
        )
        assert result == "orchestrator_review"


class TestRouteAfterHuman:
    def test_approved_ends_pipeline(self):
        result = route_after_human(_state(approved=True))
        assert result == "__end__"

    def test_rejected_with_retries_goes_to_orchestrator_review(self):
        """Human rejected + retries left → orchestrator_review for rewrite."""
        result = route_after_human(_state(approved=False, retry=0))
        assert result == "orchestrator_review"

    def test_rejected_with_feedback(self):
        """Human rejected with comments → orchestrator_review with feedback."""
        result = route_after_human(_state(
            approved=False,
            human_feedback={"action": "reject", "comments": "need more tension"},
            retry=0,
        ))
        assert result == "orchestrator_review"

    def test_rejected_no_retries_force_ends(self):
        """Human rejected but no retries left → force end (prevent infinite loop)."""
        result = route_after_human(_state(approved=False, retry=MAX_RETRIES))
        assert result == "__end__"
