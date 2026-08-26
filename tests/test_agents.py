"""Tests for agent 层 — Editor / Orchestrator / Worldbuilding。

Agent 测试不调真实 LLM：patch ``run_with_tools`` 返回假 JSON，
只验证输入组装与输出解析。纯函数（Orchestrator 的 prompt helper）直接测。
"""

import asyncio
from unittest.mock import AsyncMock, patch

from novel_agent.agents.editor import EditorAgent
from novel_agent.agents.orchestrator import OrchestratorAgent
from novel_agent.agents.worldbuilding import WorldbuildingAgent


class TestEditorReview:
    @staticmethod
    def _capture_user_prompt(agent, **kwargs):
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=('{"overall_score": 85}', None)),
        ) as mocked:
            asyncio.run(agent.review(chapter_number=1, draft_content="正文", **kwargs))
        return mocked.call_args.args[0][1]["content"]

    def test_review_parses_report(self):
        agent = EditorAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=('{"overall_score": 85, "verdict": "pass"}', None)),
        ):
            report, _ = asyncio.run(agent.review(chapter_number=1, draft_content="正文"))
        assert report["overall_score"] == 85
        assert report["verdict"] == "pass"

    def test_narrative_mode_injected(self):
        user = self._capture_user_prompt(EditorAgent(), narrative_mode="unit_arc")
        assert "unit_arc" in user

    def test_no_narrative_mode_no_hint(self):
        user = self._capture_user_prompt(EditorAgent())
        assert "当前叙事模式" not in user

    def test_empty_output_returns_unavailable(self):
        """空输出（reasoning 模型偶发）→ unavailable 标记，而非假 0 分。"""
        agent = EditorAgent()
        with patch.object(agent, "run_with_tools", new=AsyncMock(return_value=("", None))):
            report, _ = asyncio.run(agent.review(chapter_number=1, draft_content="正文"))
        assert report.get("unavailable") is True
        assert report.get("overall_score") == 0
        assert report.get("verdict") == "manual_review"

    def test_unparseable_output_returns_unavailable(self):
        """非空但解析失败（JSON 语法错误/截断）→ unavailable，而非假 0 分。"""
        agent = EditorAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("这不是JSON", None)),
        ):
            report, _ = asyncio.run(agent.review(chapter_number=1, draft_content="正文"))
        assert report.get("unavailable") is True
        assert report.get("overall_score") == 0

    def test_editor_system_prompt_contains_deductive_rubric(self):
        """Editor 系统提示词必须包含严苛扣分阶梯（篇幅不足扣分、禁用词扣分、结尾升华扣分）。"""
        agent = EditorAgent()
        prompt = agent.system_prompt
        assert "严苛扣分阶梯" in prompt
        assert "字数/篇幅严重不足" in prompt
        assert "结尾总结升华" in prompt
        assert "出现公文禁用词" in prompt
        assert "rhythm" in prompt and "ai_flavor" in prompt and "dialogue" in prompt


class TestOrchestratorPromptHelpers:
    def test_analyze_includes_story_length_and_foreshadowings(self):
        agent = OrchestratorAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("{}", None)),
        ) as mocked:
            asyncio.run(
                agent.analyze(
                    chapter_number=4,
                    chapter_outline="大纲",
                    previous_chapters=[],
                    story_length="short",
                    context_packet={
                        "character_context": "",
                        "world_context": "",
                        "unresolved_foreshadowings": ["[第1章] 神秘信物"],
                    },
                )
            )
        prompt = mocked.call_args.args[0][1]["content"]
        assert "篇幅：短篇" in prompt
        assert "神秘信物" in prompt

    def test_analyze_uses_explicit_total_chapters_over_tail_slice(self):
        """长篇只传最近 5 章 tail 时，已完成章节数用显式 total 而非 len()。"""
        agent = OrchestratorAgent()
        tail = [{"chapter_number": i} for i in range(96, 101)]
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("{}", None)),
        ) as mocked:
            asyncio.run(
                agent.analyze(
                    chapter_number=101,
                    chapter_outline="大纲",
                    previous_chapters=tail,
                    total_chapters=100,
                )
            )
        prompt = mocked.call_args.args[0][1]["content"]
        assert "已完成章节数：100章" in prompt

    def test_analyze_parse_failure_fallback_is_natural_continuation(self):
        """Orchestrator 输出解析失败时的兜底不得回退到 cliffhanger（防 AI 网文味回流）。"""
        agent = OrchestratorAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("not json at all", None)),
        ):
            result = asyncio.run(
                agent.analyze(
                    chapter_number=1,
                    chapter_outline="大纲",
                    previous_chapters=[],
                )
            )
        assert result["chapter_strategy"]["ending_type"] == "natural_continuation"

    def test_mode_instruction_none(self):
        assert OrchestratorAgent._build_mode_instruction(None) == ""

    def test_mode_instruction_unit_arc(self):
        text = OrchestratorAgent._build_mode_instruction("unit_arc")
        assert "unit_arc" in text
        assert "unit_number" in text

    def test_mode_instruction_multi_perspective(self):
        text = OrchestratorAgent._build_mode_instruction("multi_perspective")
        assert "pov_config" in text

    def test_mode_instruction_linear_default(self):
        text = OrchestratorAgent._build_mode_instruction("linear")
        assert "linear" in text

    def test_mode_instructions_never_require_all_optional_fields(self):
        """Phase D: 模式指令不得要求输出全部可选字段（每章全字段 = token 浪费）。"""
        for mode in ("linear", "unit_arc", "hybrid", "multi_perspective", "ensemble"):
            text = OrchestratorAgent._build_mode_instruction(mode)
            assert "同时输出完整" not in text, f"{mode} still demands all fields"

    def test_scene_instruction_prunes_key_scenes_when_not_scene_first(self):
        """Phase D: 整章模式明确不要求 key_scenes/scene_composition。"""
        plain = OrchestratorAgent._build_scene_instruction(scene_first=False)
        scene = OrchestratorAgent._build_scene_instruction(scene_first=True)
        assert "不需要输出 key_scenes" in plain
        assert "必须将本章拆解为 3-4 个分镜场景" in scene
        assert "scene_composition" in scene

    def test_analyze_scene_first_reaches_user_prompt(self):
        """analyze(scene_first=True) 将拆场要求注入 user prompt。"""
        agent = OrchestratorAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("{}", None)),
        ) as mocked:
            asyncio.run(
                agent.analyze(
                    chapter_number=1,
                    chapter_outline="大纲",
                    previous_chapters=[],
                    scene_first=True,
                )
            )
        user = mocked.call_args.args[0][1]["content"]
        assert "scene_first 拆场模式" in user

        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=("{}", None)),
        ) as mocked:
            asyncio.run(
                agent.analyze(
                    chapter_number=1,
                    chapter_outline="大纲",
                    previous_chapters=[],
                )
            )
        user = mocked.call_args.args[0][1]["content"]
        assert "整章生成模式" in user

    def test_perspective_hint_first_person(self):
        text = OrchestratorAgent._build_perspective_hint("first_person")
        assert "第一人称" in text

    def test_perspective_hint_unknown_or_empty(self):
        assert OrchestratorAgent._build_perspective_hint("") == ""
        assert OrchestratorAgent._build_perspective_hint("bogus") == ""


class TestWorldbuildingExtract:
    @staticmethod
    def _capture_user_prompt(agent):
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=('{"new_entities": []}', None)),
        ) as mocked:
            asyncio.run(agent.extract(chapter_number=1, draft_content="正文"))
        return mocked.call_args.args[0][1]["content"]

    def test_open_foreshadowings_in_context(self):
        agent = WorldbuildingAgent(
            existing_foreshadowings=[
                {
                    "description": "神秘信物",
                    "planted_chapter": 1,
                    "status": "open",
                    "risk_level": "high",
                },
                {"description": "已解决", "planted_chapter": 1, "status": "resolved"},
            ],
        )
        user = self._capture_user_prompt(agent)
        assert "已有伏笔" in user
        assert "神秘信物" in user
        assert "已解决" not in user  # resolved 不进 context

    def test_no_foreshadowings_no_context(self):
        user = self._capture_user_prompt(WorldbuildingAgent())
        assert "已有伏笔" not in user

    def test_extract_parses_report(self):
        agent = WorldbuildingAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=('{"new_entities": [{"name": "林风"}]}', None)),
        ):
            report, _ = asyncio.run(agent.extract(chapter_number=1, draft_content="正文"))
        # parse_validated 会按 schema 补齐字段，这里只断言关键字段被保留
        assert report["new_entities"][0]["name"] == "林风"

    def test_extracts_complete_long_chapter(self):
        content = "前文" + ("铺垫" * 2500) + "尾部关键设定"
        agent = WorldbuildingAgent()
        with patch.object(
            agent,
            "run_with_tools",
            new=AsyncMock(return_value=('{"new_entities": []}', None)),
        ) as mocked:
            asyncio.run(agent.extract(chapter_number=1, draft_content=content))
        assert "尾部关键设定" in mocked.call_args.args[0][1]["content"]
