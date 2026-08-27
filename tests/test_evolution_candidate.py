from novel_agent.services.evolution import (
    candidate_draft,
    candidate_from_state,
    candidate_to_state,
)


def test_evolution_candidate_round_trips_all_review_state():
    state = {
        "draft_content": "正文",
        "editor_report": {"overall_score": 80},
        "continuity_report": {"overall_score": 90},
        "worldbuilding_report": {"new_entities": []},
        "quality_gate_report": {"passed": True},
        "outline_coverage": 0.8,
        "required_facts_missing": 1,
    }
    candidate = candidate_from_state(
        state,
        version=2,
        scores={"editor_overall": 80, "composite": 85},
        quality_guard_report={"passed": True},
    )

    restored = candidate_to_state(candidate)
    assert candidate["version"] == 2
    assert candidate["content_length"] == 2
    assert restored["draft_content"] == "正文"
    assert restored["quality_gate_report"]["passed"] is True
    assert restored["worldbuilding_report"] == {"new_entities": []}


def test_candidate_conversion_normalizes_missing_report_values():
    candidate = candidate_from_state(
        {"draft_content": "正文", "editor_report": None, "continuity_report": None},
        version=1,
        scores={},
    )

    assert candidate["editor_report"] == {}
    assert candidate["continuity_report"] == {}
    assert candidate_to_state(candidate)["worldbuilding_report"] == {}


def test_persisted_candidate_carries_version_id_not_draft():
    """Phase 6: Storage-backed candidates keep only a runtime reference."""
    state = {
        "draft_content": "很长的正文" * 500,
        "editor_report": {"overall_score": 80},
        "quality_gate_report": {"passed": True},
    }
    candidate = candidate_from_state(state, version=3, scores={}, version_id="ver-123")

    assert candidate["version_id"] == "ver-123"
    assert "draft_content" not in candidate
    # content_length is retained for guard comparisons
    assert candidate["content_length"] == len("很长的正文" * 500)


def test_unpersisted_candidate_keeps_inline_draft_fallback():
    """Persistence failure → draft stays inline so rollback never loses data."""
    state = {"draft_content": "回退正文"}
    candidate = candidate_from_state(state, version=1, scores={}, version_id=None)

    assert candidate["draft_content"] == "回退正文"
    assert "version_id" not in candidate


def test_candidate_draft_prefers_inline_then_storage():
    inline = candidate_draft({"draft_content": "内联"})
    assert inline == "内联"

    loaded = candidate_draft(
        {"version_id": "ver-9"},
        loader=lambda vid: {"content": f"存储正文@{vid}"},
    )
    assert loaded == "存储正文@ver-9"

    # Inline wins over storage when both exist
    both = candidate_draft(
        {"draft_content": "内联", "version_id": "x"},
        loader=lambda vid: {"content": "存储"},
    )
    assert both == "内联"

    missing = candidate_draft({"version_id": "gone"}, loader=lambda vid: None)
    assert missing == ""

    no_loader = candidate_draft({"version_id": "ver-9"}, loader=None)
    assert no_loader == ""
