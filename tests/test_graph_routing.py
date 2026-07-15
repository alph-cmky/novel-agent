"""Tests for LangGraph conditional routing logic and feedback loop."""

from novel_agent.graph.chapter import (
    CONTINUITY_PASS_SCORE,
    MAX_RETRIES,
    route_after_continuity,
    route_after_evolution,
    route_after_human_evolution,
    route_after_human_legacy,
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
        result = route_after_human_legacy(_state(approved=True))
        assert result == "__end__"

    def test_rejected_with_retries_goes_to_orchestrator_review(self):
        """Human rejected + retries left → orchestrator_review for rewrite."""
        result = route_after_human_legacy(_state(approved=False, retry=0))
        assert result == "orchestrator_review"

    def test_rejected_with_feedback(self):
        """Human rejected with comments → orchestrator_review with feedback."""
        result = route_after_human_legacy(_state(
            approved=False,
            human_feedback={"action": "reject", "comments": "need more tension"},
            retry=0,
        ))
        assert result == "orchestrator_review"

    def test_rejected_no_retries_force_ends(self):
        """Human rejected but no retries left → force end (prevent infinite loop)."""
        result = route_after_human_legacy(_state(approved=False, retry=MAX_RETRIES))
        assert result == "__end__"


class TestRouteAfterEvolution:
    """Tests for the evolution subgraph routing."""

    def _evo_state(self, termination="", evolution_round=0, max_rounds=5):
        """Build a minimal NovelState for evolution routing tests."""
        return {
            "evolution_termination": termination,
            "evolution_round": evolution_round,
            "evolution_max_rounds": max_rounds,
            "editor_report": {},
            "continuity_report": {},
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
            "human_approved": True,
            "human_feedback": {},
        }

    def test_continue_when_no_termination(self):
        """No termination → continue evolution."""
        result = route_after_evolution(self._evo_state(evolution_round=1))
        assert result == "evolution_writer"

    def test_select_best_when_terminated(self):
        """Termination set → select best version."""
        result = route_after_evolution(self._evo_state(termination="converged"))
        assert result == "evolution_select_best"

    def test_select_best_when_max_rounds(self):
        """Max rounds reached → select best version."""
        result = route_after_evolution(self._evo_state(evolution_round=5, max_rounds=5))
        assert result == "evolution_select_best"

    def test_select_best_when_over_max_rounds(self):
        """Over max rounds → select best version."""
        result = route_after_evolution(self._evo_state(evolution_round=6, max_rounds=5))
        assert result == "evolution_select_best"


class TestRouteAfterHumanEvolution:
    """Tests for the evolution-aware human review routing."""

    def _evo_state(self, approved=False, rejects=0):
        """Build a minimal NovelState for evolution human routing tests."""
        return {
            "human_approved": approved,
            "evolution_human_rejects": rejects,
            "editor_report": {},
            "continuity_report": {},
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
            "human_feedback": {},
        }

    def test_approved_ends(self):
        result = route_after_human_evolution(self._evo_state(approved=True))
        assert result == "__end__"

    def test_rejected_starts_evolution(self):
        result = route_after_human_evolution(self._evo_state(approved=False, rejects=0))
        assert result == "evolution_writer"

    def test_rejected_3_times_force_ends(self):
        """Safety valve: 3 rejects → force end."""
        result = route_after_human_evolution(self._evo_state(approved=False, rejects=3))
        assert result == "__end__"
