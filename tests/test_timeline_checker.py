from novel_agent.graph.timeline_checker import check_timeline


def test_timeline_checker_detects_order_and_dead_character_reappearance():
    result = check_timeline(
        [
            {"chapter_number": 2, "subject": "林风", "action": "回城"},
            {"chapter_number": 1, "subject": "林风", "action": "战死"},
            {"chapter_number": 3, "subject": "林风", "action": "再次出现"},
        ],
        [],
        current_chapter=3,
    )

    types = [finding["type"] for finding in result["findings"]]
    assert "event_order_violation" in types
    assert "dead_character_reappeared" in types
    assert result["passed"] is False


def test_timeline_checker_detects_overdue_and_dormant_foreshadowings():
    result = check_timeline(
        [],
        [
            {
                "description": "旧钟",
                "planted_chapter": 2,
                "expected_resolve_chapter": 5,
                "status": "open",
            },
            {"description": "暗门", "planted_chapter": 1, "status": "open"},
            {"description": "已回收", "planted_chapter": 1, "status": "resolved"},
        ],
        current_chapter=8,
    )

    types = [finding["type"] for finding in result["findings"]]
    assert types.count("overdue_foreshadowing") == 1
    assert types.count("dormant_foreshadowing") == 1
    assert result["passed"] is True
