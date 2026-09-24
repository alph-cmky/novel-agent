from novel_agent.services.quality import QualityService


def test_quality_gate_rejects_empty_and_short_draft():
    result = QualityService.check_draft_hard_gates("", target_words=3000, chapter_outline="大纲")

    assert result["passed"] is False
    assert "empty_content" in result["violations"]
    assert "minimum_length" in result["violations"]


def test_quality_gate_accepts_target_length_with_outline():
    result = QualityService.check_draft_hard_gates(
        "正文" * 2600,
        target_words=3000,
        chapter_outline="主角进入城门",
    )

    assert result["passed"] is True
    assert result["violations"] == []


def test_quality_service_keeps_hard_gate_output_shape():
    result = QualityService.check_draft_hard_gates(
        "正文" * 600,
        target_words=600,
        chapter_outline="大纲",
    )

    assert result == {
        "passed": True,
        "violations": [],
        "content_units": 1200,
        "minimum_units": 600,
        "target_units": 600,
    }
