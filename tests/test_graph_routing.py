"""Tests for LangGraph evolution routing logic."""

from novel_agent.graph.chapter import (
    route_after_evolution,
    route_after_human_evolution,
)


def _evo_state(termination="", evolution_round=0, max_rounds=5):
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


class TestRouteAfterEvolution:
    """Tests for the evolution subgraph routing."""

    def test_continue_when_no_termination(self):
        """No termination → continue evolution."""
        result = route_after_evolution(_evo_state(evolution_round=1))
        assert result == "evolution_writer"

    def test_select_best_when_terminated(self):
        """Termination set → select best version."""
        result = route_after_evolution(_evo_state(termination="converged"))
        assert result == "evolution_select_best"

    def test_select_best_when_max_rounds(self):
        """Max rounds reached → select best version."""
        result = route_after_evolution(_evo_state(evolution_round=5, max_rounds=5))
        assert result == "evolution_select_best"

    def test_select_best_when_over_max_rounds(self):
        """Over max rounds → select best version."""
        result = route_after_evolution(_evo_state(evolution_round=6, max_rounds=5))
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
