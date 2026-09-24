import pytest

from novel_agent.services.continuity import ContinuityService, is_death_action


def test_timeline_checker_detects_order_and_dead_character_reappearance():
    result = ContinuityService.check_timeline(
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
    result = ContinuityService.check_timeline(
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


def test_continuity_service_does_not_fail_for_warnings_only():
    result = ContinuityService.check_timeline(
        [],
        [{"description": "未回收伏笔", "planted_chapter": 1, "status": "open"}],
        current_chapter=5,
    )

    assert result["passed"] is True
    assert result["findings"][0]["severity"] == "warning"


@pytest.mark.parametrize(
    ("action", "is_death"),
    [
        ("甲死亡", True),
        ("战死沙场", True),
        ("身死", True),
        ("毙命", True),
        ("he died", True),
        ("she dies", True),
        ("讨论死亡", False),
        ("关于死亡", False),
        ("未死亡", False),
        ("不死亡", False),
        ("没有死亡", False),
        ("不曾死去", False),
        ("仿佛死去", False),
        ("好像战死", False),
        ("提及毙命", False),
        ("not died", False),
        ("studies", False),
    ],
)
def test_is_death_action(action, is_death):
    assert is_death_action(action) is is_death


@pytest.mark.parametrize(
    ("subject", "action", "draft", "flagged"),
    [
        ("林风", "林风战死", "林风推门进来。", True),
        ("白", "白战死", "白天林风进门。", False),
        ("林风", "讨论死亡", "林风推门进来。", False),
    ],
)
def test_draft_reappearance_uses_a_whole_name(subject, action, draft, flagged):
    result = ContinuityService.check_draft_against_timeline(
        [{"chapter_number": 1, "subject": subject, "action": action}],
        draft,
        current_chapter=3,
    )
    assert result["passed"] is not flagged
    if flagged:
        assert result["findings"][0]["appearance_chapter"] == 3
        assert result["findings"][0]["subject"] == subject
    else:
        assert result["findings"] == []
