from novel_agent.graph.scenes import assemble_scenes, build_scene_outcome, build_scene_plan


def test_build_scene_plan_uses_key_scenes_and_balances_target():
    plan = build_scene_plan(
        "备用大纲",
        3000,
        {"chapter_strategy": {"key_scenes": ["冲突", {"goal": "转折"}]}},
    )

    assert [scene["scene_index"] for scene in plan] == [1, 2]
    assert sum(scene["target_words"] for scene in plan) == 3000
    assert plan[0]["outline"] == "冲突"
    assert plan[1]["outline"] == "转折"


def test_build_scene_plan_falls_back_to_outline_sentences():
    plan = build_scene_plan("主角进城。敌人出现！", 2000)

    assert len(plan) == 2
    assert assemble_scenes([" 第一场 ", "", "第二场"]) == "第一场\n\n第二场"


class TestBuildSceneOutcome:
    """Phase 2: scene context 用结构化 outcome 替换 raw [-1200:]。"""

    def test_takes_last_paragraph(self):
        text = "第一段内容。\n\n第二段内容。\n\n最后结局段。"
        outcome = build_scene_outcome(text)
        assert "[前场结局]" in outcome
        assert "最后结局段" in outcome
        assert "第一段内容" not in outcome

    def test_takes_last_two_paragraphs(self):
        text = "第一段。\n\n倒数第二段。\n\n最后段。"
        outcome = build_scene_outcome(text)
        assert "倒数第二段" in outcome
        assert "最后段" in outcome
        assert "第一段" not in outcome

    def test_caps_at_max_chars(self):
        long_ending = "字" * 600
        text = f"开头。\n\n{long_ending}"
        outcome = build_scene_outcome(text, max_chars=400)
        assert len(outcome) <= 420  # 400 + label overhead

    def test_empty_content_returns_empty(self):
        assert build_scene_outcome("") == ""

    def test_single_paragraph(self):
        text = "只有一个段落。"
        outcome = build_scene_outcome(text)
        assert "只有一个段落" in outcome
