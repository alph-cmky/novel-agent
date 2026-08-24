from novel_agent.graph.scenes import assemble_scenes, build_scene_plan


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
