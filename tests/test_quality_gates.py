from novel_agent.graph.quality_gates import check_draft_hard_gates, check_story_integrity


def test_quality_gate_rejects_empty_and_short_draft():
    result = check_draft_hard_gates("", target_words=3000, chapter_outline="大纲")

    assert result["passed"] is False
    assert "empty_content" in result["violations"]
    assert "minimum_length" in result["violations"]


def test_quality_gate_accepts_target_length_with_outline():
    result = check_draft_hard_gates(
        "正文" * 2600,
        target_words=3000,
        chapter_outline="主角进入城门",
    )

    assert result["passed"] is True
    assert result["violations"] == []


def test_story_checker_detects_missing_scene_and_required_fact():
    result = check_story_integrity(
        "第一场正文",
        scene_plan=[{"scene_index": 1}, {"scene_index": 2}],
        scene_drafts=["第一场正文"],
        required_facts=["火种"],
    )

    assert result["passed"] is False
    assert "scene_count_mismatch" in result["violations"]
    assert "required_fact_missing" in result["violations"]


def test_story_checker_flags_explicit_canon_conflict_only():
    result = check_story_integrity(
        "北墙由黑曜石砌成，青石已经开裂。",
        canon_conflicts=[
            {"severity": "critical", "keywords": ["黑曜石", "青石"]}
        ],
    )

    assert result["passed"] is False
    assert result["findings"][0]["severity"] == "critical"
