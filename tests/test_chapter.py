"""Tests for chapter graph nodes — candidate scores 维度补零。

锁住修复：``evolution_orchestrator_node`` 计算 best_scores 时，若
candidate editor dimensions 缺失某维度，需补 0 到 5 维，否则
``composite_score(best_scores)`` 的 ``dims_avg`` 分母 = 存在的维度数，
与 ``current_scores``（恒 5 维，来自 ``extract_scores``）不一致，
导致 ``is_new_best`` 判断错误。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from novel_agent.graph.chapter import (
    evolution_orchestrator_node,
    orchestrator_node,
    writer_node,
)


def _state(**overrides) -> dict:
    state = {
        "evolution_round": 1,
        "evolution_version": 2,
        "evolution_history": [
            {
                "v": 1,
                "editor": 70,
                "continuity": 70,
                "composite": 70.0,
                "dimensions": {
                    "rhythm": 70,
                    "ai_flavor": 70,
                    "dialogue": 70,
                    "logic": 70,
                    "writing": 70,
                },
                "delta": None,
                "focus": None,
            }
        ],
        "editor_report": {
            "overall_score": 80,
            "dimensions": {
                "rhythm": 80,
                "ai_flavor": 80,
                "dialogue": 80,
                "logic": 80,
                "writing": 80,
            },
        },
        "continuity_report": {"overall_score": 80},
        "evolution_candidates": [
            {
                "version": 1,
                "draft_content": "旧正文",
                "editor_report": {
                    "overall_score": 80,
                    "dimensions": {"rhythm": 100},
                },
                "continuity_report": {"overall_score": 80},
            }
        ],
        "evolution_best_candidate_version": 1,
        "evolution_max_rounds": 5,
        "draft_content": "新正文",
    }
    state.update(overrides)
    return state


def _run(state: dict) -> dict:
    # patch _config_for 避免真实 env/模型路由；patch Agent 类跳过 LLM 调用。
    with (
        patch("novel_agent.graph.chapter._config_for", return_value=None),
        patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
    ):
        mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
        return asyncio.run(evolution_orchestrator_node(state))


class TestBestScoresZeroFill:
    def test_missing_best_dimensions_zero_filled(self):
        """best dims 只有 1 维时，补零后 dims_avg 分母仍为 5。

        current composite = 80（5 维全 80）。
        修复后 best composite = 80*0.5 + 80*0.3 + (100/5)*0.2 = 68 < 80 → 更新 best。
        修复前 best composite = 80*0.5 + 80*0.3 + (100/1)*0.2 = 84 > 80 → 误判为不更新。
        """
        result = _run(_state())
        assert result["evolution_best_candidate_version"] == 2
        assert result["evolution_termination"] == ""

    def test_full_best_dimensions_unchanged(self):
        """best dims 完整 5 维时，补零不改变结果（对照）。"""
        state = _state(
            evolution_candidates=[
                {
                    "version": 1,
                    "draft_content": "旧正文",
                    "editor_report": {
                        "overall_score": 60,
                        "dimensions": {
                            "rhythm": 60,
                            "ai_flavor": 60,
                            "dialogue": 60,
                            "logic": 60,
                            "writing": 60,
                        },
                    },
                    "continuity_report": {"overall_score": 60},
                }
            ],
            evolution_best_candidate_version=1,
        )
        result = _run(state)
        # best composite = 60 < 80 → 仍更新 best
        assert result["evolution_best_candidate_version"] == 2


class TestWriterPromptProfileInGraph:
    """P0: graph 节点正确传递 writer_prompt_profile。

    回归测试：
    - orchestrator_node 不再向 OrchestratorAgent.analyze() 传 prompt_profile
      （旧代码会 TypeError: analyze() got an unexpected keyword argument）。
    - writer_node 从 state 读取 writer_prompt_profile 并传入 WriterAgent 构造。
    """

    @staticmethod
    def _orch_state(**overrides) -> dict:
        state = {
            "project_id": "",
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "story_length": "short",
            "character_context": "",
            "world_context": "",
            "skip_orchestrator": False,
        }
        state.update(overrides)
        return state

    def test_orchestrator_does_not_pass_prompt_profile(self):
        """orchestrator_node 不再向 analyze() 传 prompt_profile（修复 TypeError）。"""
        mock_orch = MagicMock()
        mock_orch._compressor = MagicMock()
        mock_orch._compressor.should_compress = MagicMock(return_value=False)
        mock_orch.analyze = AsyncMock(return_value={})
        with patch("novel_agent.graph.chapter.OrchestratorAgent", return_value=mock_orch):
            asyncio.run(orchestrator_node(self._orch_state()))
        kwargs = mock_orch.analyze.call_args.kwargs
        assert "prompt_profile" not in kwargs

    @staticmethod
    def _run_writer_node(state: dict) -> str:
        """Run writer_node, return prompt_profile passed to WriterAgent."""
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured["prompt_profile"] = kwargs.get("prompt_profile")
            mock_w = MagicMock()
            mock_w.write = AsyncMock(
                return_value=("正文" * 600, MagicMock(input_tokens=10, output_tokens=20))
            )
            mock_w.latest_trace = MagicMock(input_tokens=10, output_tokens=20)
            return mock_w

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", side_effect=_capture),
            patch(
                "novel_agent.graph.chapter.QualityService.check_draft_hard_gates",
                return_value={"passed": True, "violations": []},
            ),
            patch(
                "novel_agent.graph.chapter.QualityService.check_story_integrity",
                return_value={"passed": True, "violations": []},
            ),
        ):
            asyncio.run(writer_node(state, None))
        return captured.get("prompt_profile", "")

    def test_writer_node_default_v2(self):
        """state 不含 writer_prompt_profile → 传 v2。"""
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        assert self._run_writer_node(state) == "v2"

    def test_writer_node_explicit_v2(self):
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
            "writer_prompt_profile": "v2",
        }
        assert self._run_writer_node(state) == "v2"

    def test_writer_node_explicit_v1(self):
        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 1000,
            "persist_dir": "./novel-data",
            "project_id": "",
            "writer_prompt_profile": "v1",
        }
        assert self._run_writer_node(state) == "v1"


class TestNarrativeExtension:
    """Phase 1: writer_node 用 Narrative Extension 替换 compensation。

    回归测试：
    - actual < target → 触发 narrative_extension（不再调 write 全文重写）
    - extension 是 append，不是 replace
    - actual >= target → 不触发 extension
    - extension 后仍不足 → quality_gate_report['extension_failed'] = True
    - QualityService 最低门槛从 0.85 改为 0.5
    """

    @staticmethod
    def _run_writer_with_mocks(
        write_content: str = "正" * 3000,
        extension_content: str = "",
        target_words: int = 3000,
    ) -> dict:
        """Run writer_node with mocked WriterAgent, return result dict."""
        mock_writer = MagicMock()
        mock_writer.write = AsyncMock(
            return_value=(write_content, MagicMock(input_tokens=10, output_tokens=20))
        )
        mock_writer.write_stream = MagicMock()
        mock_writer.narrative_extension = AsyncMock(return_value=extension_content)
        mock_writer.latest_trace = MagicMock(input_tokens=10, output_tokens=20)

        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": target_words,
            "persist_dir": "./novel-data",
            "project_id": "",
        }

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", return_value=mock_writer),
        ):
            result = asyncio.run(writer_node(state, None))
        return result

    def test_short_content_triggers_extension(self):
        """actual < target → 调用 narrative_extension。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1500,
            extension_content="续" * 1500,
            target_words=3000,
        )
        assert "续" * 1500 in result["draft_content"]
        assert "短" * 1500 in result["draft_content"]

    def test_extension_appends_not_replaces(self):
        """Extension 是 append，原始内容保留。"""
        result = self._run_writer_with_mocks(
            write_content="原始内容",
            extension_content="续写内容",
            target_words=3000,
        )
        assert "原始内容" in result["draft_content"]
        assert "续写内容" in result["draft_content"]

    def test_target_length_no_extension(self):
        """actual >= target → 不触发 extension。"""
        mock_writer = MagicMock()
        mock_writer.write = AsyncMock(
            return_value=("正" * 3000, MagicMock(input_tokens=10, output_tokens=20))
        )
        mock_writer.narrative_extension = AsyncMock(return_value="不该出现")
        mock_writer.latest_trace = MagicMock(input_tokens=10, output_tokens=20)

        state = {
            "chapter_number": 1,
            "chapter_outline": "大纲",
            "target_chapter_words": 3000,
            "persist_dir": "./novel-data",
            "project_id": "",
        }
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.WriterAgent", return_value=mock_writer),
        ):
            result = asyncio.run(writer_node(state, None))
        mock_writer.narrative_extension.assert_not_called()
        assert "不该出现" not in result["draft_content"]

    def test_extension_still_short_sets_failed(self):
        """Extension 后仍不足 → extension_failed + passed=False + violation。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1000,
            extension_content="续" * 500,
            target_words=3000,
        )
        gate = result["quality_gate_report"]
        assert gate.get("extension_failed") is True
        assert gate["passed"] is False
        assert "length_target_unmet" in gate["violations"]

    def test_extension_success_no_failed_flag(self):
        """Extension 后达标 → 不设 extension_failed。"""
        result = self._run_writer_with_mocks(
            write_content="短" * 1500,
            extension_content="续" * 1500,
            target_words=3000,
        )
        assert "extension_failed" not in result["quality_gate_report"]

    def test_quality_gate_uses_half_target_not_85(self):
        """QualityService 硬性检查最低门槛从 0.85 改为 0.5。"""
        from novel_agent.services.quality import QualityService

        # 50% of target → 0.85 时代会 fail，0.5 时代 pass
        report = QualityService.check_draft_hard_gates(
            "字" * 1500,
            target_words=3000,
            chapter_outline="大纲",
        )
        assert report["passed"] is True

        # 40% of target → 0.5 以下，应该 fail
        report = QualityService.check_draft_hard_gates(
            "字" * 1200,
            target_words=3000,
            chapter_outline="大纲",
        )
        assert "minimum_length" in report["violations"]


class TestEvolutionContextMinimal:
    """Task 4: evolution enrichment 默认不调用 + 不传完整 draft。"""

    @staticmethod
    def _run_evo_first_round(state_extra: dict | None = None) -> dict:
        """Run evolution_orchestrator_node first-round branch."""
        state = {
            "evolution_round": 0,
            "evolution_version": 0,
            "evolution_history": [],
            "editor_report": {"overall_score": 70, "dimensions": {}},
            "continuity_report": {"overall_score": 70},
            "evolution_max_rounds": 5,
        }
        if state_extra:
            state.update(state_extra)
        with (
            patch("novel_agent.graph.chapter._config_for", return_value=None),
            patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
        ):
            mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
            result = asyncio.run(evolution_orchestrator_node(state))
        return result, mock_agent

    def test_first_round_enrichment_not_called_when_rule_plan_has_instruction(self):
        """规则计划有 primary_instruction 时不调用 LLM enrichment。"""
        result, mock_agent = self._run_evo_first_round()
        mock_agent.return_value.enrich_plan.assert_not_called()

    def test_first_round_enrichment_called_when_rule_plan_empty(self):
        """规则计划缺 primary_instruction 时才调用 LLM enrichment。"""
        from unittest.mock import patch as _patch

        with _patch(
            "novel_agent.graph.chapter.build_improvement_plan_rule",
            return_value={"primary_instruction": "", "focus_dimensions": []},
        ):
            result, mock_agent = self._run_evo_first_round({"skip_evolution_enrichment": False})
        mock_agent.return_value.enrich_plan.assert_called_once()

    @staticmethod
    def test_enrichment_draft_preview_capped():
        """enrich_plan 收到的 draft_preview 不超过 800 字。"""
        from unittest.mock import patch as _patch

        long_draft = "草" * 5000
        with _patch(
            "novel_agent.graph.chapter.build_improvement_plan_rule",
            return_value={"primary_instruction": "", "focus_dimensions": []},
        ):
            state = {
                "evolution_round": 0,
                "evolution_version": 0,
                "evolution_history": [],
                "editor_report": {"overall_score": 70, "dimensions": {}},
                "continuity_report": {"overall_score": 70},
                "evolution_max_rounds": 5,
                "draft_content": long_draft,
                "skip_evolution_enrichment": False,
            }
            with (
                patch("novel_agent.graph.chapter._config_for", return_value=None),
                patch("novel_agent.graph.chapter.EvolutionOrchestratorAgent") as mock_agent,
            ):
                mock_agent.return_value.enrich_plan = AsyncMock(return_value=None)
                asyncio.run(evolution_orchestrator_node(state))

        call_kwargs = mock_agent.return_value.enrich_plan.call_args.kwargs
        draft_preview = call_kwargs.get("draft_preview", "")
        assert len(draft_preview) <= 800


class TestContextMinimality:
    """Graph-level: verify agents receive minimal context, not full State.

    Regression guard against re-introducing full-context leaks.
    """

    @staticmethod
    def _editor_state(**overrides) -> dict:
        state = {
            "chapter_number": 1,
            "draft_content": "正文内容" * 100,
            "narrative_mode": None,
            "context_packet": {
                "character_context": "角色" * 5000,
                "world_context": "设定" * 5000,
                "recent_summary": "摘要" * 5000,
                "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
                "timeline_events": [{"e": f"事件{i}"} for i in range(30)],
                "timeline_findings": [{"f": f"发现{i}"} for i in range(10)],
            },
        }
        state.update(overrides)
        return state

    def test_editor_node_passes_style_report_and_minimal_context(self):
        """Editor receives StyleReport + for_editor projection, not full packet."""
        from novel_agent.graph.chapter import editor_node

        captured: dict = {}

        async def _mock_review(**kwargs):
            captured.update(kwargs)
            return {"overall_score": 80, "verdict": "pass"}, MagicMock()

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.EditorAgent") as mock_editor_cls,
        ):
            mock_editor_cls.return_value.review = _mock_review
            asyncio.run(editor_node(self._editor_state()))

        # style_report must be present (deterministic, 0 LLM)
        assert "style_report" in captured
        assert captured["style_report"] is not None
        assert "style_gate" in captured["style_report"]
        assert "paragraph_structure" in captured["style_report"]

        # context_packet must be the minimal for_editor projection
        ctx = captured.get("context_packet") or {}
        assert "world_context" not in ctx
        assert "timeline_events" not in ctx
        assert len(ctx.get("unresolved_foreshadowings", [])) <= 3

    def test_editor_node_returns_style_report_in_state(self):
        """editor_node output includes style_report for Evolution consumption."""
        from novel_agent.graph.chapter import editor_node

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter.EditorAgent") as mock_editor_cls,
        ):
            mock_editor_cls.return_value.review = AsyncMock(
                return_value=({"overall_score": 80, "verdict": "pass"}, MagicMock())
            )
            result = asyncio.run(editor_node(self._editor_state()))

        assert "style_report" in result
        assert result["style_report"]
        assert "style_gate" in result["style_report"]

    @staticmethod
    def _continuity_state(**overrides) -> dict:
        state = {
            "chapter_number": 1,
            "draft_content": "正文内容" * 100,
            "narrative_mode": None,
            "persist_dir": "./novel-data",
            "project_id": "",
            "context_packet": {
                "character_context": "角色" * 5000,
                "world_context": "设定" * 5000,
                "recent_summary": "摘要" * 5000,
                "unresolved_foreshadowings": [f"伏笔{i}" for i in range(20)],
                "timeline_events": [{"e": f"事件{i}"} for i in range(30)],
                "timeline_findings": [{"f": f"发现{i}"} for i in range(10)],
            },
        }
        state.update(overrides)
        return state

    def test_continuity_node_passes_minimal_context(self):
        """Continuity receives for_continuity projection, not full packet."""
        from novel_agent.graph.chapter import continuity_node

        captured: dict = {}

        async def _mock_audit(**kwargs):
            captured.update(kwargs)
            return {"overall_score": 90, "inconsistencies": []}, MagicMock()

        with (
            patch("novel_agent.graph.chapter._config_for", return_value=MagicMock()),
            patch("novel_agent.graph.chapter._get_chapter_store"),
            patch("novel_agent.graph.chapter.ContinuityAgent") as mock_cty_cls,
        ):
            mock_cty_cls.return_value.audit = _mock_audit
            asyncio.run(continuity_node(self._continuity_state()))

        ctx = captured.get("context_packet") or {}
        # for_continuity keeps timeline_events but not world_context or recent_summary
        assert "world_context" not in ctx
        assert "recent_summary" not in ctx
        assert len(ctx.get("timeline_events", [])) <= 10
        assert len(ctx.get("unresolved_foreshadowings", [])) <= 8
