from novel_agent.services.evolution import candidate_from_state, candidate_to_state


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
